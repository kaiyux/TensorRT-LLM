"""High-level entrypoint to transform a model into an efficient inference model."""

import gc

import torch
from torch.fx import GraphModule

from ..compile import compile_and_capture
from ..custom_ops.attention_interface import AttentionRegistry
from ..distributed import common as dist_ad
from ..models.factory import ModelFactory
from ..shim.interface import AutoDeployConfig, CachedSequenceInterface
from ..utils.logger import ad_logger
from ._graph import move_to_device
from .export import torch_export_to_gm
from .library import (
    check_in_out_nodes,
    column_row_shard,
    ep_shard,
    fuse_allreduce_residual_rmsnorm,
    fuse_collectives,
    fuse_gemms,
    fuse_moe,
    identify_and_fuse_mha,
    insert_mha_with_kv_cache,
    insert_mla_with_kv_cache,
    match_moe_pattern,
    quantize,
    resize_kv_cache,
)


class InferenceOptimizer:
    def __init__(
        self,
        factory: ModelFactory,
        *,  # TODO: temporary until we have a better config system
        ad_config: AutoDeployConfig,
        visualize: bool = False,
    ):
        self.factory = factory
        self.attn_backend = ad_config.attn_backend
        self.mla_backend = ad_config.mla_backend
        # TODO (lliebenwein): let's split up the compile backend to separately handle cuda graph
        # and torch compile so we can follow the PyTorchConfig here and enable it separately.
        self.ad_config = ad_config
        if ad_config.use_cuda_graph or ad_config.torch_compile_enabled:
            compile_backend = "torch-opt"
        else:
            compile_backend = "torch-simple"
        self.compile_backend = compile_backend
        self.visualize = visualize

        # look up attention op
        self.attention_op = AttentionRegistry.get(self.attn_backend)
        self.mla_op = AttentionRegistry.get(self.mla_backend)

    def __call__(self, cm: CachedSequenceInterface) -> GraphModule:
        """Transform a model into an optimized inference model.

        Args:
            model: The model to transform.
            cp: The cache pool to use for caching.
            args: Example inputs to the model.
            dynamic_shapes: Dynamic shapes to use. Defaults to None.
            poe_config: The config for positional encoding. Defaults to None.
            quantization: The quantization method to use. Defaults to None.

        Returns:
            A GraphModule representing the optimized inference model.
        """
        ############################################################################################
        # INITIALIZE MODEL
        ############################################################################################
        model = self.factory.build_model(device="meta")

        ############################################################################################
        # EXPORT MODEL TO GRAPH MODULE
        ############################################################################################

        cm.info._set_example_sequence()
        egm = torch_export_to_gm(model, args=cm.args[:1], dynamic_shapes=cm.dynamic_shapes[:1])
        del model
        ad_logger.debug("original graph: " + str(egm))
        local_rank, world_size = dist_ad.get_rank_world_size()

        ############################################################################################
        # RUN PATTERN MATCHER TRANSFORMATIONS TO STANDARDIZE GRAPH REPRESENTATION
        ############################################################################################

        # quantization
        egm = quantize(egm, self.factory.get_quant_config())

        # Match MoE pattern
        egm = match_moe_pattern(egm)

        # identify MHA patterns
        egm = identify_and_fuse_mha(egm, self.factory.get_positional_embedding_config())

        ############################################################################################
        # RUN TRANSFORMATIONS ON STANDARDIZED GRAPH REPRESENTATION
        ############################################################################################

        input_node = check_in_out_nodes(egm)

        # insert MHA with KV cache
        egm = insert_mha_with_kv_cache(
            egm, cm, self.attention_op, self.factory.get_cache_config(), input_node
        )

        # insert MLA with KV cache
        egm = insert_mla_with_kv_cache(
            egm, cm, self.mla_op, self.factory.get_cache_config(), input_node
        )

        # run TP sharding across ranks
        egm = column_row_shard(egm, local_rank, world_size)

        # run EP sharding across ranks
        egm = ep_shard(egm, local_rank, world_size)

        ############################################################################################
        # SETUP CACHES AND LOAD WEIGHTS
        ############################################################################################

        # initialize caches, load weights, and map to correct device
        cm.initialize_caches()

        # load weights
        self.factory.load_or_random_init(egm, mmap=True, map_location=cm.device)
        move_to_device(egm, cm.device)

        ############################################################################################
        # RUN POST-LOAD FUSION AND OPTIMIZATIONS
        ############################################################################################

        # run MoE fusion
        egm = fuse_moe(egm)

        # run GEMM fusion
        egm = fuse_gemms(egm)

        # check if we can fuse allreduce, residual and rmsnorm
        egm = fuse_allreduce_residual_rmsnorm(egm)

        # check if we can fuse collectives
        egm = fuse_collectives(egm)

        # visualize the final graph
        if self.visualize:
            try:
                from .library import visualize_namespace

                visualize_namespace(egm, args=cm.args, dynamic_shapes=cm.dynamic_shapes)
                ad_logger.warning(
                    "Please run `pip install -r examples/auto_deploy/requirements.txt` to visualize"
                    " the graph."
                )
            except ImportError:
                pass

        ############################################################################################
        # RESIZE CACHE
        ############################################################################################
        # Free memory ratio is hardcoded to 0.8 for now to ensure we have enough memory for graph capture.
        resize_kv_cache(egm, cm, free_mem_ratio=0.8)

        ############################################################################################
        # COMPILE MODEL
        ############################################################################################

        cm.info._set_generate_only_batch()
        compiler_kwargs = {
            "cuda_graph_batch_sizes": self.ad_config.cuda_graph_batch_sizes,
            "num_batched_inputs": 1,  # TODO (lucaslie): improve once we have a config system...
        }
        egm_compiled = compile_and_capture(
            egm,
            self.compile_backend,
            args=cm.args,
            dynamic_shapes=cm.dynamic_shapes,
            compiler_kwargs=compiler_kwargs,
        )
        cm.info.reset()

        torch.cuda.empty_cache()
        gc.collect()
        return egm_compiled
