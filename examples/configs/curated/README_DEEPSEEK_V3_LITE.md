# DeepSeek V3 Lite Optimization Configurations

This document provides comprehensive documentation for the curated TensorRT-LLM optimization configurations for DeepSeek V3 Lite model deployment on NVIDIA B200 GPUs.

## Table of Contents

- [Profiling Summary](#profiling-summary)
- [Available Configurations](#available-configurations)
- [Usage Examples](#usage-examples)
- [Optimization Details](#optimization-details)
- [Configuration Comparison Table](#configuration-comparison-table)
- [Expected Performance](#expected-performance)

---

## Profiling Summary

The DeepSeek V3 Lite configurations are optimized based on detailed profiling conducted on NVIDIA B200 hardware with NVFP4 quantization.

### Hardware Configuration

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA B200 |
| HBM | 178GB |
| Quantization | NVFP4 |
| Tensor Parallel | 1 GPU |

### Profiling Results

#### Prefill Phase

| Metric | Value |
|--------|-------|
| Input Sequence Length (ISL) | 512 tokens |
| Batch Size | 32 |
| Throughput | **22,222 tok/s** |
| Time to First Token (TTFT) | 404ms |
| Bottleneck | **Compute-bound** |
| MoE GPU Time | ~49% |

#### Decode Phase

| Metric | Value |
|--------|-------|
| Input Sequence Length (ISL) | 128 tokens |
| Output Sequence Length (OSL) | 512 tokens |
| Batch Size | 8 |
| Throughput | **1,311 tok/s** |
| Time Per Output Token (TPOT) | 5.4ms |
| Bottleneck | **Memory-bound** |
| Primary Constraint | PagedKV Cache Bandwidth |

### Key Insights

1. **Prefill is compute-bound**: MoE (Mixture of Experts) layers consume ~49% of GPU time during prefill, making CUTLASS MoE backend optimizations critical.

2. **Decode is memory-bound**: PagedKV cache access dominates decode time, making FP8 KV cache and memory optimizations essential.

3. **Dual optimization strategy**: Different phases require different optimization approaches - compute optimization for prefill, memory optimization for decode.

---

## Available Configurations

### 1. Latency Configuration (`deepseek-v3-lite-latency.yaml`)

**Use Case**: Applications requiring the lowest possible latency, such as interactive chat, real-time assistants, and latency-sensitive APIs.

**Target Workload**: Short-to-medium length requests with small concurrent user counts.

**Key Characteristics**:
- Small batch sizes (4-8) for minimal queuing latency
- CUDA graphs enabled for fast kernel dispatch
- No speculative decoding (adds overhead for short requests)
- Overlap scheduler disabled for simpler, more predictable latency
- Stream interval of 1 for immediate response streaming

### 2. Throughput Configuration (`deepseek-v3-lite-throughput.yaml`)

**Use Case**: Batch processing, high-load servers, and scenarios prioritizing total tokens/second over individual request latency.

**Target Workload**: High concurrent request counts, batch inference workloads.

**Key Characteristics**:
- Large batch sizes (up to 128) for maximum GPU utilization
- MTP speculative decoding (1 layer) for decode acceleration
- Overlap scheduler enabled for additional throughput (~4%)
- Higher KV cache memory allocation (90%)
- Extended GEMM backend options including cutedsl

### 3. Balanced Configuration (`deepseek-v3-lite-balanced.yaml`)

**Use Case**: General-purpose deployment with mixed workloads requiring good latency and throughput.

**Target Workload**: Production deployments with variable traffic patterns.

**Key Characteristics**:
- Medium batch sizes (32) balancing both metrics
- MTP speculative decoding (2 layers) for moderate speedup
- All performance optimizations enabled
- Moderate stream interval for good responsiveness
- Best choice when workload characteristics are unknown

---

## Usage Examples

### Using with `trtllm-serve`

#### Latency-Optimized Serving

```bash
trtllm-serve \
    --model deepseek-ai/DeepSeek-V3-Lite \
    --extra-llm-api-options examples/configs/curated/deepseek-v3-lite-latency.yaml
```

#### Throughput-Optimized Serving

```bash
trtllm-serve \
    --model deepseek-ai/DeepSeek-V3-Lite \
    --extra-llm-api-options examples/configs/curated/deepseek-v3-lite-throughput.yaml
```

#### Balanced Serving (Recommended for General Use)

```bash
trtllm-serve \
    --model deepseek-ai/DeepSeek-V3-Lite \
    --extra-llm-api-options examples/configs/curated/deepseek-v3-lite-balanced.yaml
```

### Using with `trtllm-bench`

#### Benchmarking Latency Configuration

```bash
trtllm-bench \
    --model deepseek-ai/DeepSeek-V3-Lite \
    --extra-llm-api-options examples/configs/curated/deepseek-v3-lite-latency.yaml \
    --dataset synthetic \
    --num-requests 100 \
    --input-seq-len 512 \
    --output-seq-len 128
```

#### Benchmarking Throughput Configuration

```bash
trtllm-bench \
    --model deepseek-ai/DeepSeek-V3-Lite \
    --extra-llm-api-options examples/configs/curated/deepseek-v3-lite-throughput.yaml \
    --dataset synthetic \
    --num-requests 1000 \
    --concurrency 64 \
    --input-seq-len 512 \
    --output-seq-len 512
```

#### Comparing Configurations

```bash
# Run benchmark sweep across all configurations
for config in latency throughput balanced; do
    echo "Testing deepseek-v3-lite-${config}.yaml"
    trtllm-bench \
        --model deepseek-ai/DeepSeek-V3-Lite \
        --extra-llm-api-options examples/configs/curated/deepseek-v3-lite-${config}.yaml \
        --dataset synthetic \
        --num-requests 500 \
        --output-file results_${config}.json
done
```

### Custom Configuration Override

You can override specific parameters while using a base configuration:

```bash
trtllm-serve \
    --model deepseek-ai/DeepSeek-V3-Lite \
    --extra-llm-api-options examples/configs/curated/deepseek-v3-lite-balanced.yaml \
    --max-batch-size 64  # Override the default batch size
```

---

## Optimization Details

### 1. CUDA Graphs with Padding

**What it does**: Pre-compiles GPU kernel execution sequences into reusable graphs, eliminating CPU-side kernel launch overhead.

**Configuration**:
```yaml
cuda_graph_config:
  enable_padding: true
  max_batch_size: 32
  batch_sizes: [1, 2, 4, 8, 16, 24, 32]
```

**Benefits**:
- **22% decode speedup** due to reduced kernel launch latency
- Padding enables efficient graph reuse across varying batch sizes
- Pre-defined batch sizes optimize for common request patterns

**When to use**: Always recommended for decode-heavy workloads. The overhead of graph compilation is amortized over many requests.

### 2. FP8 KV Cache

**What it does**: Stores key-value cache in FP8 format instead of FP16/BF16, reducing memory bandwidth requirements.

**Configuration**:
```yaml
kv_cache_config:
  dtype: fp8
  free_gpu_memory_fraction: 0.85-0.90
```

**Benefits**:
- 2x memory bandwidth reduction for KV cache access
- Enables larger batch sizes or longer sequences
- Critical for memory-bound decode phase

**Memory Allocation by Config**:
| Configuration | Memory Fraction | Purpose |
|---------------|-----------------|---------|
| Latency | 85% | Reserve memory for fast allocation |
| Balanced | 88% | Moderate buffer |
| Throughput | 90% | Maximize KV cache capacity |

### 3. CUTLASS MoE Backend

**What it does**: Uses CUTLASS (CUDA Templates for Linear Algebra Subroutines) for optimized MoE expert computations.

**Configuration**:
```yaml
moe_config:
  backend: CUTLASS
  use_low_precision_moe_combine: true
```

**Benefits**:
- Optimized for SM100 Blackwell architecture (B200)
- Better performance for compute-bound MoE layers
- Efficient expert selection and computation

**Why CUTLASS for DeepSeek V3 Lite**: The MoE layers consume ~49% of prefill GPU time, making backend optimization critical. CUTLASS provides the best performance on modern NVIDIA architectures.

### 4. Low Precision MoE Combine

**What it does**: Performs expert output combination in lower precision, reducing computational overhead.

**Configuration**:
```yaml
moe_config:
  use_low_precision_moe_combine: true
```

**Benefits**:
- Improved throughput for NVFP4 quantized models
- Reduced memory bandwidth for expert combination
- Minimal accuracy impact with proper quantization

### 5. MTP Speculative Decoding

**What it does**: Uses Multi-Token Prediction (MTP) to speculate multiple tokens ahead, improving decode throughput.

**Configuration** (Throughput):
```yaml
speculative_config:
  decoding_type: MTP
  num_nextn_predict_layers: 1
```

**Configuration** (Balanced):
```yaml
speculative_config:
  decoding_type: MTP
  num_nextn_predict_layers: 2
```

**Layer Configuration Guidelines**:
| Layers | Use Case | Trade-off |
|--------|----------|-----------|
| 0 (disabled) | Latency-critical, short outputs | Lowest per-request overhead |
| 1 | High throughput, large batches | Good speedup, minimal memory |
| 2 | Balanced workloads | Better speedup, moderate memory |
| 3+ | Long outputs, lower batches | Best speedup, higher memory |

**Why disabled for latency config**: MTP adds per-request overhead that hurts latency for short requests. For latency-optimized scenarios, the overhead outweighs benefits.

### 6. Overlap Scheduler

**What it does**: Overlaps compute operations to improve GPU utilization and hide latency.

**Configuration**:
```yaml
# Throughput/Balanced - enabled
disable_overlap_scheduler: false

# Latency - disabled
disable_overlap_scheduler: true
```

**Benefits**:
- ~4% additional throughput when enabled
- Better GPU utilization for high-batch workloads

**Why disabled for latency**: The overlap introduces slight unpredictability in timing. For latency-critical workloads, simpler scheduling provides more consistent response times.

### 7. NVFP4 GEMM Backend Selection

**What it does**: Configures which GEMM backends are allowed for NVFP4 quantized operations.

**Configuration**:
```yaml
nvfp4_gemm_config:
  allowed_backends: ['cutlass', 'cublaslt', 'cuda_core']
  # Throughput also includes: 'cutedsl'
```

**Backend Options**:
| Backend | Characteristics |
|---------|-----------------|
| `cutlass` | High performance, optimized templates |
| `cublaslt` | cuBLAS library, stable and reliable |
| `cutedsl` | Extreme performance, longer launch time |
| `cuda_core` | Fallback for unsupported shapes |

**Why cutedsl only for throughput**: The longer kernel launch time is acceptable when optimizing for sustained throughput rather than individual request latency.

---

## Configuration Comparison Table

| Parameter | Latency | Balanced | Throughput |
|-----------|---------|----------|------------|
| **Batch Configuration** |
| `max_batch_size` | 8 | 32 | 128 |
| `max_num_tokens` | 8,192 | 12,288 | 16,384 |
| **CUDA Graphs** |
| `enable_padding` | true | true | true |
| `max_batch_size` | 8 | 32 | 128 |
| `batch_sizes` | [1,2,4,8] | [1,2,4,8,16,24,32] | [1,2,4,8,16,32,64,128] |
| **KV Cache** |
| `dtype` | fp8 | fp8 | fp8 |
| `free_gpu_memory_fraction` | 0.85 | 0.88 | 0.90 |
| **MoE** |
| `backend` | CUTLASS | CUTLASS | CUTLASS |
| `use_low_precision_moe_combine` | true | true | true |
| **Scheduler** |
| `disable_overlap_scheduler` | true | false | false |
| `stream_interval` | 1 | 5 | 10 |
| **Speculative Decoding** |
| `decoding_type` | disabled | MTP | MTP |
| `num_nextn_predict_layers` | - | 2 | 1 |
| **NVFP4 Backends** |
| `allowed_backends` | cutlass, cublaslt, cuda_core | cutlass, cublaslt, cuda_core | cutlass, cublaslt, cutedsl, cuda_core |

---

## Expected Performance

### Latency Configuration

**Optimized for**: Time to First Token (TTFT) and Time Per Output Token (TPOT)

| Metric | Expected Performance |
|--------|---------------------|
| TTFT (ISL=512) | ~400-450ms |
| TPOT | ~5-6ms |
| P99 Latency Consistency | Excellent |
| Throughput | Moderate |

**Best for**:
- Interactive chatbots
- Real-time code completion
- Latency-sensitive APIs
- Small concurrent user counts (<10)

### Throughput Configuration

**Optimized for**: Maximum tokens processed per second

| Metric | Expected Performance |
|--------|---------------------|
| Decode Throughput | Up to 40% improvement over baseline |
| Batch Efficiency | Excellent at high loads |
| GPU Utilization | >85% sustained |
| Latency | Higher variance |

**Best for**:
- Batch processing pipelines
- High-traffic production servers
- Offline inference workloads
- Cost optimization (tokens per GPU-hour)

### Balanced Configuration

**Optimized for**: Good performance across all metrics

| Metric | Expected Performance |
|--------|---------------------|
| TTFT | Good (~450-500ms) |
| TPOT | Good (~6-8ms) |
| Throughput | Good (70-80% of max) |
| Consistency | Good |

**Best for**:
- Production deployments with mixed traffic
- When workload characteristics are unknown
- Starting point for optimization
- Multi-tenant deployments

### Performance Improvement Summary

| Optimization | Impact | Applies To |
|--------------|--------|------------|
| CUDA Graphs | **22% decode speedup** | All configs |
| FP8 KV Cache | 2x memory efficiency | All configs |
| CUTLASS MoE | Best SM100 performance | All configs |
| MTP (1 layer) | Decode acceleration | Throughput |
| MTP (2 layers) | Better decode acceleration | Balanced |
| Overlap Scheduler | **~4% throughput boost** | Balanced, Throughput |

---

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**:
- Reduce `max_batch_size`
- Lower `free_gpu_memory_fraction`
- Use latency config for smaller memory footprint

**High Latency Variance**:
- Switch to latency config
- Ensure `disable_overlap_scheduler: true`
- Reduce `stream_interval`

**Low Throughput**:
- Switch to throughput config
- Increase `max_batch_size`
- Enable overlap scheduler

### Configuration Tuning Tips

1. **Start with Balanced**: Use balanced config as baseline, then tune based on observed metrics.

2. **Monitor GPU Utilization**: Low utilization suggests room for larger batches.

3. **Watch Memory Usage**: If close to capacity, reduce batch size or memory fraction.

4. **Profile Your Workload**: Use `trtllm-bench` to measure your specific input/output length distributions.

---

## Additional Resources

- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [DeepSeek V3 Model Documentation](https://github.com/deepseek-ai/DeepSeek-V3)
- [NVIDIA B200 Specifications](https://www.nvidia.com/en-us/data-center/b200/)
- [Mixture of Experts (MoE) Optimization Guide](https://nvidia.github.io/TensorRT-LLM/advanced/moe.html)

---

*Last Updated: 2024*
*Configurations tested on: NVIDIA B200 (178GB HBM) with TensorRT-LLM*
