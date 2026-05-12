# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
"""
model_family for dlrm_v3.
"""

import copy
import functools
import logging
import os
import time
import uuid
from threading import Event
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp
import torchrec
from checkpoint import (
    load_nonsparse_checkpoint,
    load_sparse_checkpoint,
)
from configs import HASH_SIZE
from datasets.dataset import Samples
from inference_modules import (
    get_hstu_model,
    HSTUSparseInferenceModule,
    move_sparse_output_to_device,
    set_is_inference,
)
from utils import Profiler
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig, SequenceEmbedding
from pyre_extensions import none_throws
from torch import quantization as quant
from torchrec.distributed.quant_embedding import QuantEmbeddingCollection
from torchrec.modules.embedding_configs import EmbeddingConfig, QuantConfig
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from torchrec.sparse.tensor_dict import maybe_td_to_kjt
from torchrec.test_utils import get_free_port

logger: logging.Logger = logging.getLogger(__name__)


def _get_dense_inference_dtype() -> torch.dtype:
    """
    Use bfloat16 only on GPUs with native support (Ampere+).
    """
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


class HSTUModelFamily:
    """
    High-level interface for the HSTU model family.

    Manages both sparse (embedding) and dense (transformer) components of the
    HSTU model, supporting distributed inference across multiple GPUs.

    Args:
        hstu_config: Configuration object for the HSTU model.
        table_config: Dictionary of embedding table configurations.
        output_trace: Whether to enable profiling trace output.
        sparse_quant: Whether to quantize sparse embeddings.
        compute_eval: Whether to compute evaluation metrics (includes labels).
        use_compile: Whether to apply torch.compile to the dense model.
    """

    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
        output_trace: bool = False,
        sparse_quant: bool = False,
        compute_eval: bool = False,
        use_compile: bool = False,
    ) -> None:
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.sparse: ModelFamilySparseDist = ModelFamilySparseDist(
            hstu_config=hstu_config,
            table_config=table_config,
            quant=sparse_quant,
        )

        assert torch.cuda.is_available(), "CUDA is required for this benchmark."
        ngpus = torch.cuda.device_count()
        self.world_size = int(os.environ.get("WORLD_SIZE", str(ngpus)))
        logger.warning(f"Using {self.world_size} GPU(s)...")
        dense_model_family_clazz = (
            ModelFamilyDenseDist
            if self.world_size > 1
            else ModelFamilyDenseSingleWorker
        )
        self.dense: Union[ModelFamilyDenseDist, ModelFamilyDenseSingleWorker] = (
            dense_model_family_clazz(
                hstu_config=hstu_config,
                table_config=table_config,
                output_trace=output_trace,
                compute_eval=compute_eval,
                use_compile=use_compile,
            )
        )

    def version(self) -> str:
        """Return the PyTorch version string."""
        return torch.__version__

    def name(self) -> str:
        """Return the model family name identifier."""
        return "model-family-hstu"

    def load(self, model_path: str) -> None:
        """
        Load model checkpoints from disk.

        Args:
            model_path: Base path to the model checkpoint directory.
        """
        self.sparse.load(model_path=model_path)
        self.dense.load(model_path=model_path)

    def predict(
        self, samples: Optional[Samples]
    ) -> Optional[
        Tuple[
            torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, float
        ]
    ]:
        """
        Run inference on a batch of samples.

        Processes samples through sparse embeddings, then dense forward pass.

        Args:
            samples: Input samples containing features. If None, signals shutdown.

        Returns:
            Tuple of (predictions, labels, weights, sparse_time, dense_time) or None.
        """
        with torch.no_grad():
            if samples is None:
                self.dense.predict(None, None, 0, None, 0, None)
                return None
            (
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
                dt_sparse,
            ) = self.sparse.predict(samples)
            out = self.dense.predict(
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            )
            (  # pyre-ignore [23]
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
                dt_dense,
            ) = out
            return (
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
                dt_sparse,
                dt_dense,
            )


def ec_patched_forward_wo_embedding_copy(
    ec_module: torchrec.EmbeddingCollection,
    features: KeyedJaggedTensor,  # can also take TensorDict as input
) -> Dict[str, JaggedTensor]:
    """
    Run the EmbeddingBagCollection forward pass. This method takes in a `KeyedJaggedTensor`
    and returns a `Dict[str, JaggedTensor]`, which is the result of the individual embeddings for each feature.

    Args:
        features (KeyedJaggedTensor): KJT of form [F X B X L].

    Returns:
        Dict[str, JaggedTensor]
    """
    features = maybe_td_to_kjt(features, None)
    feature_embeddings: Dict[str, JaggedTensor] = {}
    jt_dict: Dict[str, JaggedTensor] = features.to_dict()
    for i, emb_module in enumerate(ec_module.embeddings.values()):
        feature_names = ec_module._feature_names[i]
        embedding_names = ec_module._embedding_names_by_table[i]
        for j, embedding_name in enumerate(embedding_names):
            feature_name = feature_names[j]
            f = jt_dict[feature_name]
            indices = torch.clamp(f.values(), min=0, max=HASH_SIZE - 1)
            lookup = emb_module(
                input=indices
            )  # remove the dtype cast at https://github.com/meta-pytorch/torchrec/blob/0a2cebd5472a7edc5072b3c912ad8aaa4179b9d9/torchrec/modules/embedding_modules.py#L486
            feature_embeddings[embedding_name] = JaggedTensor(
                values=lookup,
                lengths=f.lengths(),
                weights=f.values() if ec_module._need_indices else None,
            )
    return feature_embeddings


class ModelFamilySparseDist:
    """
    Sparse Arch module manager.

    Handles loading and inference of sparse embedding lookups, optionally
    with quantization for memory efficiency.

    Args:
        hstu_config: HSTU model configuration.
        table_config: Embedding table configurations.
        quant: Whether to apply dynamic quantization to embeddings.
    """

    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
        quant: bool = False,
    ) -> None:
        super(ModelFamilySparseDist, self).__init__()
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.module: Optional[torch.nn.Module] = None
        self.quant: bool = quant

    def load(self, model_path: str) -> None:
        """
        Load sparse model checkpoint and optionally apply quantization.

        Args:
            model_path: Path to the model checkpoint directory.
        """
        logger.warning(f"Loading sparse module from {model_path}")

        sparse_arch: HSTUSparseInferenceModule = HSTUSparseInferenceModule(
            table_config=self.table_config,
            hstu_config=self.hstu_config,
        )
        load_sparse_checkpoint(model=sparse_arch._hstu_model, path=model_path)
        sparse_arch.eval()
        if self.quant:
            self.module = quant.quantize_dynamic(
                sparse_arch,
                qconfig_spec={
                    torchrec.EmbeddingCollection: QuantConfig(
                        activation=quant.PlaceholderObserver.with_args(
                            dtype=torch.float
                        ),
                        weight=quant.PlaceholderObserver.with_args(
                            dtype=torch.int8),
                    ),
                },
                mapping={
                    torchrec.EmbeddingCollection: QuantEmbeddingCollection,
                },
                inplace=False,
            )
        else:
            sparse_arch._hstu_model._embedding_collection.forward = (  # pyre-ignore[8]
                functools.partial(
                    ec_patched_forward_wo_embedding_copy,
                    sparse_arch._hstu_model._embedding_collection,
                )
            )
            self.module = sparse_arch
        logger.warning(f"sparse module is {self.module}")

    def predict(
        self, samples: Samples
    ) -> Tuple[
        Dict[str, SequenceEmbedding],
        Dict[str, torch.Tensor],
        int,
        torch.Tensor,
        int,
        torch.Tensor,
        float,
    ]:
        """
        Run sparse forward pass (embedding lookups).

        Args:
            samples: Input samples with feature tensors.

        Returns:
            Tuple of (seq_embeddings, payload_features, max_uih_len, uih_seq_lengths,
            max_num_candidates, num_candidates, elapsed_time).
        """
        with torch.profiler.record_function("sparse forward"):
            module: torch.nn.Module = none_throws(self.module)
            assert self.module is not None
            uih_features = samples.uih_features_kjt
            candidates_features = samples.candidates_features_kjt
            t0: float = time.time()
            (
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            ) = module(
                uih_features=uih_features,
                candidates_features=candidates_features,
            )
            dt_sparse: float = time.time() - t0
            return (
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
                dt_sparse,
            )


class ModelFamilyDenseDist:
    """
    Distributed dense module manager for multi-GPU inference.

    Spawns worker processes for each GPU to run dense forward passes in parallel,
    with samples distributed via inter-process queues.

    Optimization notes:
    - CPU tensors are passed through the queue (shared memory, zero-copy) rather than
      CUDA tensors (which require expensive CUDA IPC + deepcopy).
    - GPU transfer and torch.compile happen inside each worker process.

    Args:
        hstu_config: HSTU model configuration.
        table_config: Embedding table configurations.
        output_trace: Whether to enable profiling traces.
        compute_eval: Whether to compute evaluation metrics.
        use_compile: Whether to apply torch.compile to the dense model.
    """

    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
        output_trace: bool = False,
        compute_eval: bool = False,
        use_compile: bool = False,
    ) -> None:
        super(ModelFamilyDenseDist, self).__init__()
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.output_trace = output_trace
        self.compute_eval = compute_eval
        self.use_compile = use_compile
        self.dense_dtype = _get_dense_inference_dtype()

        ngpus = torch.cuda.device_count()
        self.world_size = int(os.environ.get("WORLD_SIZE", str(ngpus)))
        self.rank = 0
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(get_free_port())
        self.dist_backend = "nccl"

        ctx = mp.get_context("spawn")
        self.samples_q: List[mp.Queue] = [ctx.Queue()
                                          for _ in range(self.world_size)]
        self.result_q: List[mp.Queue] = [ctx.Queue()
                                         for _ in range(self.world_size)]

    def load(self, model_path: str) -> None:
        """
        Load dense model and spawn worker processes for distributed inference.

        Args:
            model_path: Path to the model checkpoint directory.
        """
        logger.warning(f"Loading dense module from {model_path}")

        ctx = mp.get_context("spawn")
        processes = []
        for rank in range(self.world_size):
            p = ctx.Process(
                target=self.distributed_setup,
                args=(
                    rank,
                    self.world_size,
                    model_path,
                ),
            )
            p.start()
            processes.append(p)

    def distributed_setup(self, rank: int, world_size: int,
                          model_path: str) -> None:
        """
        Initialize and run a dense worker process.

        Each worker loads the model, moves CPU tensors to GPU, runs dense forward,
        and returns results. CPU tensors are received from the queue (shared memory,
        no deepcopy needed), and GPU transfer happens inside the worker.

        Args:
            rank: Process rank (GPU index).
            world_size: Total number of worker processes.
            model_path: Path to model checkpoint.
        """
        set_is_inference(is_inference=not self.compute_eval)
        dense_dtype = self.dense_dtype
        model = get_hstu_model(
            table_config=self.table_config,
            hstu_config=self.hstu_config,
            table_device="cpu",
            max_hash_size=100,
            is_dense=True,
        ).to(dense_dtype)
        model.set_training_dtype(dense_dtype)
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(f"cuda:{rank}")
        load_nonsparse_checkpoint(
            model=model, device=device, optimizer=None, path=model_path
        )
        model = model.to(device)
        model.eval()

        if self.use_compile:
            try:
                logger.warning(f"[rank {rank}] Applying torch.compile to dense model...")
                # mode='default' uses TorchInductor without CUDA graphs.
                # The model has .item() calls which prevent CUDA graph capture,
                # so 'reduce-overhead' would silently fall back anyway; 'default'
                # is more robust and still fuses kernels via Inductor.
                model.main_forward = torch.compile(
                    model.main_forward,
                    mode="default",
                    dynamic=True,
                    fullgraph=False,
                )
                logger.warning(f"[rank {rank}] torch.compile applied successfully.")
            except Exception as e:
                logger.warning(f"[rank {rank}] torch.compile failed ({e}), running eager.")

        profiler = Profiler(rank) if self.output_trace else None
        transfer_stream = torch.cuda.Stream(device=device)

        with torch.no_grad():
            while True:
                item = self.samples_q[rank].get()
                # If -1 is received terminate all subprocesses
                if item == -1:
                    break
                if self.output_trace:
                    assert profiler is not None
                    profiler.step()
                with torch.profiler.record_function("get_item_from_queue"):
                    # Items are CPU tensors passed via shared memory – no deepcopy needed.
                    (
                        id,
                        seq_embeddings,
                        payload_features,
                        max_uih_len,
                        uih_seq_lengths,
                        max_num_candidates,
                        num_candidates,
                    ) = item
                    assert seq_embeddings is not None
                    # Move CPU tensors to GPU inside the worker (non-blocking).
                    seq_embeddings, payload_features, uih_seq_lengths, num_candidates = (
                        move_sparse_output_to_device(
                            seq_embeddings=seq_embeddings,
                            payload_features=payload_features,
                            uih_seq_lengths=uih_seq_lengths,
                            num_candidates=num_candidates,
                            device=device,
                            embedding_dtype=self.dense_dtype,
                            stream=transfer_stream,
                        )
                    )
                    # Ensure transfers are complete before the default stream uses them.
                    torch.cuda.current_stream(device).wait_stream(transfer_stream)
                with torch.profiler.record_function("dense forward"):
                    (
                        _,
                        _,
                        _,
                        mt_target_preds,
                        mt_target_labels,
                        mt_target_weights,
                    ) = model.main_forward(
                        seq_embeddings=seq_embeddings,
                        payload_features=payload_features,
                        max_uih_len=max_uih_len,
                        uih_seq_lengths=uih_seq_lengths,
                        max_num_candidates=max_num_candidates,
                        num_candidates=num_candidates,
                    )
                    assert mt_target_preds is not None
                    mt_target_preds = mt_target_preds.detach().to(device="cpu")
                    if mt_target_labels is not None:
                        mt_target_labels = mt_target_labels.detach().to(device="cpu")
                    if mt_target_weights is not None:
                        mt_target_weights = mt_target_weights.detach().to(device="cpu")
                    self.result_q[rank].put(
                        (id, mt_target_preds, mt_target_labels, mt_target_weights)
                    )

    def capture_output(
        self, id: uuid.UUID, rank: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Retrieve inference results from a worker process.

        Args:
            id: Unique identifier for the request.
            rank: Worker rank to retrieve from.

        Returns:
            Tuple of (predictions, labels, weights).
        """
        while True:
            recv_id, preds, labels, weights = self.result_q[rank].get()
            assert recv_id == id
            return preds, labels, weights

    def get_rank(self) -> int:
        """
        Get the next worker rank for load balancing.

        Returns:
            Rank index, cycling through available workers.
        """
        rank = self.rank
        self.rank = (self.rank + 1) % self.world_size
        return rank

    def predict(
        self,
        seq_embeddings: Optional[Dict[str, SequenceEmbedding]],
        payload_features: Optional[Dict[str, torch.Tensor]],
        max_uih_len: int,
        uih_seq_lengths: Optional[torch.Tensor],
        max_num_candidates: int,
        num_candidates: Optional[torch.Tensor],
    ) -> Optional[
        Tuple[torch.Tensor, Optional[torch.Tensor],
              Optional[torch.Tensor], float]
    ]:
        """
        Run distributed dense forward pass.

        Dispatches CPU tensors to a worker process (via shared memory) and
        collects results. GPU transfer happens inside the worker.

        Args:
            seq_embeddings: Sequence embeddings from sparse module (CPU tensors).
            payload_features: Additional feature tensors (CPU tensors).
            max_uih_len: Maximum UIH sequence length.
            uih_seq_lengths: Per-sample UIH lengths.
            max_num_candidates: Maximum candidates per sample.
            num_candidates: Per-sample candidate counts.

        Returns:
            Tuple of (predictions, labels, weights, elapsed_time) or None if shutdown.
        """
        id = uuid.uuid4()
        # If none is received terminate all subprocesses
        if seq_embeddings is None:
            for rank in range(self.world_size):
                self.samples_q[rank].put(-1)
            return None
        rank = self.get_rank()
        assert (
            payload_features is not None
            and num_candidates is not None
            and uih_seq_lengths is not None
        )
        t0: float = time.time()
        # Pass CPU tensors directly into the queue. Workers receive them via shared
        # memory (zero-copy), then do the GPU transfer internally – this removes the
        # CUDA-IPC overhead and the expensive deepcopy that was previously needed.
        self.samples_q[rank].put(
            (
                id,
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            )
        )
        (mt_target_preds, mt_target_labels, mt_target_weights) = self.capture_output(
            id, rank
        )
        dt_dense = time.time() - t0
        return (
            mt_target_preds,
            mt_target_labels,
            mt_target_weights,
            dt_dense,
        )


class ModelFamilyDenseSingleWorker:
    """
    Single-worker dense module manager for single-GPU inference.

    Simpler alternative to ModelFamilyDenseDist for single-GPU setups.

    Optimization notes:
    - torch.compile is applied to main_forward for kernel fusion.
    - A dedicated CUDA transfer stream overlaps CPU-to-GPU copies with prior work.

    Args:
        hstu_config: HSTU model configuration.
        table_config: Embedding table configurations.
        output_trace: Whether to enable profiling traces.
        compute_eval: Whether to compute evaluation metrics.
        use_compile: Whether to apply torch.compile to the dense model.
    """

    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
        output_trace: bool = False,
        compute_eval: bool = False,
        use_compile: bool = False,
    ) -> None:
        self.model: Optional[torch.nn.Module] = None
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.output_trace = output_trace
        self.use_compile = use_compile
        self.dense_dtype = _get_dense_inference_dtype()
        self.device: torch.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.profiler: Optional[Profiler] = (
            Profiler(rank=0) if self.output_trace else None
        )
        self.transfer_stream: Optional[torch.cuda.Stream] = None

    def load(self, model_path: str) -> None:
        """
        Load dense model for single-GPU inference.

        Applies torch.compile after loading to reduce kernel launch overhead.

        Args:
            model_path: Path to the model checkpoint directory.
        """
        logger.warning(f"Loading dense module from {model_path}")
        dense_dtype = self.dense_dtype
        self.model = (
            get_hstu_model(
                table_config=self.table_config,
                hstu_config=self.hstu_config,
                table_device="cpu",
                is_dense=True,
            )
            .to(self.device)
            .to(dense_dtype)
        )
        self.model.set_training_dtype(dense_dtype)
        load_nonsparse_checkpoint(
            model=self.model, device=self.device, optimizer=None, path=model_path
        )
        assert self.model is not None
        self.model.eval()

        if self.use_compile:
            try:
                logger.warning("Applying torch.compile to dense model main_forward...")
                # mode='default' uses TorchInductor without CUDA graphs.
                # The model has .item() calls which prevent CUDA graph capture,
                # so 'reduce-overhead' would silently fall back anyway; 'default'
                # is more robust and still fuses kernels via Inductor.
                self.model.main_forward = torch.compile(  # pyre-ignore [8]
                    self.model.main_forward,
                    mode="default",
                    dynamic=True,
                    fullgraph=False,
                )
                logger.warning("torch.compile applied successfully.")
            except Exception as e:
                logger.warning(f"torch.compile failed ({e}), running eager.")

        self.transfer_stream = torch.cuda.Stream(device=self.device)

    def predict(
        self,
        seq_embeddings: Optional[Dict[str, SequenceEmbedding]],
        payload_features: Optional[Dict[str, torch.Tensor]],
        max_uih_len: int,
        uih_seq_lengths: Optional[torch.Tensor],
        max_num_candidates: int,
        num_candidates: Optional[torch.Tensor],
    ) -> Optional[
        Tuple[
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            float,
        ]
    ]:
        """
        Run dense forward pass on single GPU.

        CPU-to-GPU transfers use a dedicated stream and non-blocking copies.
        The default stream waits for the transfer stream before starting the
        forward pass, enabling overlap with other CPU-side work.

        Args:
            seq_embeddings: Sequence embeddings from sparse module.
            payload_features: Additional feature tensors.
            max_uih_len: Maximum UIH sequence length.
            uih_seq_lengths: Per-sample UIH lengths.
            max_num_candidates: Maximum candidates per sample.
            num_candidates: Per-sample candidate counts.

        Returns:
            Tuple of (predictions, labels, weights, elapsed_time).
        """
        if self.output_trace:
            assert self.profiler is not None
            self.profiler.step()
        assert (
            payload_features is not None
            and uih_seq_lengths is not None
            and num_candidates is not None
            and seq_embeddings is not None
        )
        t0: float = time.time()
        with torch.profiler.record_function("dense forward"):
            seq_embeddings, payload_features, uih_seq_lengths, num_candidates = (
                move_sparse_output_to_device(
                    seq_embeddings=seq_embeddings,
                    payload_features=payload_features,
                    uih_seq_lengths=uih_seq_lengths,
                    num_candidates=num_candidates,
                    device=self.device,
                    embedding_dtype=self.dense_dtype,
                    stream=self.transfer_stream,
                )
            )
            # Ensure all non-blocking transfers finish before the compute stream uses them.
            torch.cuda.current_stream(self.device).wait_stream(self.transfer_stream)  # pyre-ignore [6]
            assert self.model is not None
            (
                _,
                _,
                _,
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
            ) = self.model.main_forward(  # pyre-ignore [29]
                seq_embeddings=seq_embeddings,
                payload_features=payload_features,
                max_uih_len=max_uih_len,
                uih_seq_lengths=uih_seq_lengths,
                max_num_candidates=max_num_candidates,
                num_candidates=num_candidates,
            )
            assert mt_target_preds is not None
            mt_target_preds = mt_target_preds.detach().to(device="cpu")
            if mt_target_labels is not None:
                mt_target_labels = mt_target_labels.detach().to(device="cpu")
            if mt_target_weights is not None:
                mt_target_weights = mt_target_weights.detach().to(device="cpu")
            dt_dense: float = time.time() - t0
            return mt_target_preds, mt_target_labels, mt_target_weights, dt_dense
