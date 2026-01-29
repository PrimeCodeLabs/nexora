# Nexora: Multi-GPU LLM Inference Simulator

A simulator for understanding multi-GPU LLM inference overhead. Built to answer: *Why does TP=4 (4×24GB GPUs) often feel worse than a single 96GB GPU?* and *What optimizations actually help?*

**No GPUs required** - runs on any system including laptops.

## What This Tool Is / Is Not

**This tool IS:**
- A *model* for reasoning about tensor-parallel communication overhead
- Useful for comparing hardware configurations (PCIe vs NVLink, TP=2 vs TP=4)
- A way to explore optimization tradeoffs (overlap, precision, grouping)

**This tool is NOT:**
- A benchmark - it doesn't measure real hardware
- Exact tok/s predictions - real systems have many additional factors
- A replacement for profiling actual deployments

The simulator helps you understand *why* multi-GPU setups underperform and *where* to focus optimization efforts, but actual performance depends on your specific model, batch size, and system.

## Who This Is For

- **ML/Infra engineers** evaluating multi-GPU deployment options
- **Hardware decision-makers** comparing PCIe vs NVLink tradeoffs
- **Performance engineers** understanding where communication overhead comes from
- **Anyone curious** about why "more GPUs" doesn't always mean "faster"

## Installation (No Rust Required)

### Option 1: Download Prebuilt Binary

Download the binary for your platform and verify the checksum:

**macOS (Apple Silicon)**
```bash
# Download binary and checksum
curl -LO https://github.com/PrimeCodeLabs/nexora/releases/latest/download/nexora-macos-arm64
curl -LO https://github.com/PrimeCodeLabs/nexora/releases/latest/download/nexora-macos-arm64.sha256

# Verify checksum
shasum -a 256 -c nexora-macos-arm64.sha256

# Make executable and run
chmod +x nexora-macos-arm64
./nexora-macos-arm64 --help
```

**macOS (Intel)**
```bash
curl -LO https://github.com/PrimeCodeLabs/nexora/releases/latest/download/nexora-macos-x86_64
curl -LO https://github.com/PrimeCodeLabs/nexora/releases/latest/download/nexora-macos-x86_64.sha256
shasum -a 256 -c nexora-macos-x86_64.sha256
chmod +x nexora-macos-x86_64
./nexora-macos-x86_64 --help
```

**Linux (x86_64)**
```bash
curl -LO https://github.com/PrimeCodeLabs/nexora/releases/latest/download/nexora-linux-x86_64
curl -LO https://github.com/PrimeCodeLabs/nexora/releases/latest/download/nexora-linux-x86_64.sha256
sha256sum -c nexora-linux-x86_64.sha256
chmod +x nexora-linux-x86_64
./nexora-linux-x86_64 --help
```

**Windows (PowerShell)**
```powershell
Invoke-WebRequest -Uri https://github.com/PrimeCodeLabs/nexora/releases/latest/download/nexora-windows-x86_64.exe -OutFile nexora.exe
Invoke-WebRequest -Uri https://github.com/PrimeCodeLabs/nexora/releases/latest/download/nexora-windows-x86_64.exe.sha256 -OutFile nexora.exe.sha256
# Verify: compare output of these two commands
(Get-FileHash nexora.exe -Algorithm SHA256).Hash.ToLower()
Get-Content nexora.exe.sha256
.\nexora.exe --help
```

### Option 2: Build from Source (requires Rust)

```bash
cargo build --release
# Binary at: ./target/release/nexora
```

## Hardware Presets (Recommended)

Instead of guessing bandwidth and latency values, use hardware presets that model real-world configurations:

### PCIe (Consumer/Workstation)

| Preset | Bandwidth | Latency | Use Case |
|--------|-----------|---------|----------|
| `pcie4_good` | 18 GB/s | 25 µs | PCIe 4.0 with direct GPU-to-GPU P2P |
| `pcie4_bad` | 12 GB/s | 45 µs | PCIe 4.0 through CPU/switch (common!) |
| `pcie5_good` | 28 GB/s | 18 µs | PCIe 5.0 with good topology |
| `pcie5_bad` | 18 GB/s | 35 µs | PCIe 5.0 with suboptimal routing |

### NVIDIA NVLink (Datacenter)

| Preset | Bandwidth | Latency | Use Case |
|--------|-----------|---------|----------|
| `nvlink` | 150 GB/s | 3 µs | NVLink 3.0 (A100) |
| `nvlink4` | 225 GB/s | 2 µs | NVLink 4.0 (H100) |
| `nvlink5` | 450 GB/s | 1.5 µs | NVLink 5.0 (B200) |

### AMD / InfiniBand

| Preset | Bandwidth | Latency | Use Case |
|--------|-----------|---------|----------|
| `infinity_fabric` | 200 GB/s | 3 µs | AMD Infinity Fabric (MI300) |
| `ib_hdr` | 25 GB/s | 1.5 µs | InfiniBand HDR (multi-node) |
| `ib_ndr` | 50 GB/s | 1 µs | InfiniBand NDR (multi-node) |

**Why presets matter:** Most consumer/workstation multi-GPU setups have `pcie4_bad` characteristics - the GPUs communicate through the CPU or a PCIe switch, not directly. This adds significant latency that dominates small-message transfers.

**Note on InfiniBand:** IB presets model multi-node communication. Despite lower bandwidth than NVLink, the ultra-low RDMA latency makes IB competitive for small message sizes.

```bash
# Recommended: start with a preset
./nexora --preset pcie4_bad --tp 4

# Only override manually if you have measured your specific hardware
./nexora --bw-gbps 15.0 --latency-ns 35000 --tp 4
```

## Quick Start

```bash
# See TP=4 overhead on typical PCIe hardware
./nexora --preset pcie4_bad --tp 4

# Compare with NVLink (much lower overhead)
./nexora --preset nvlink --tp 4

# Explore optimizations
./nexora --preset pcie4_bad --tp 4 --overlap 0.3 --max-risk 0.5
```

## Example Output

```
=== Multi-GPU LLM Inference Simulator ===

Hardware Preset: PCIe4 Bad Topology
Configuration:
  Layers:        32
  TP degree:     4
  Compute/layer: 80.00 us
  Comm ops/layer: 2
  AR bytes:      32768 B
  Bandwidth:     12.0 GB/s
  Latency:       45.00 us
  Algorithm:     Ring
  Group size:    1
  Precision:     16 bits
  Overlap:       0%

--- Baseline Results (TP=4) ---
  Time/token:    11.69 ms
  Throughput:    85.6 tok/s
  Compute:       2.56 ms (21.9%)
  Comm:          8.90 ms (76.2%)
  Collectives:   64
  Risk:          0.00

--- TP Comparison (vs TP=1 baseline) ---
  TP=1:  2.59 ms  (baseline)
  TP=2:  5.83 ms  ratio=2.25x  overhead=+124.6%
  TP=4:  11.69 ms  ratio=4.50x  overhead=+350.4%

  *** Comm-bound regime: TP=4 is 4.5x slower than TP=1 ***

=== RUN SUMMARY ===
  preset:       pcie4_bad
  bw/lat:       12.0 GB/s / 45000 ns
  algo/ops:     Ring / 2 ops/layer
  ar/ag_bytes:  32768 / 0 B
  tp:           4
  t_tp1:        2594953 ns
  t_tp4:        11687479 ns
  ratio:        4.50x
  overhead_pct: 350.4%
```

This shows the *communication-bound regime* where TP=4 is actually 4.5x slower than TP=1 due to PCIe overhead. With optimizations enabled, the optimizer can find configurations that significantly reduce this penalty.

## Usage Examples

### Comparing Hardware Configurations

```bash
# PCIe 4.0 "bad" topology (most common consumer setup)
./nexora --preset pcie4_bad --tp 4

# PCIe 4.0 "good" topology (direct P2P enabled)
./nexora --preset pcie4_good --tp 4

# NVLink generations
./nexora --preset nvlink --tp 8    # A100
./nexora --preset nvlink4 --tp 8   # H100
./nexora --preset nvlink5 --tp 8   # B200

# AMD MI300
./nexora --preset infinity_fabric --tp 8

# Multi-node InfiniBand
./nexora --preset ib_ndr --tp 8
```

### Exploring Optimizations

```bash
# Enable compute/comm overlap (30%)
./nexora --preset pcie4_bad --tp 4 --overlap 0.3

# Allow optimizer to search with higher risk tolerance
./nexora --preset pcie4_bad --tp 4 --max-risk 0.5 --overlap 0.3

# Skip optimization, just show baseline
./nexora --preset pcie4_bad --tp 4 --no-opt
```

### TTFT (Time To First Token) Simulation

```bash
# Basic TTFT simulation
./nexora --preset pcie4_bad --ttft

# Custom prompt/output sizes
./nexora --preset pcie4_bad --ttft --prompt-tokens 1024 --output-tokens 256

# Cache-aware decode (shows context scaling)
./nexora --preset pcie4_bad --ttft --cache-aware
```

### Parameter Sweep (Sensitivity Analysis)

```bash
# Sweep bandwidth and latency values, output CSV
./nexora --sweep --sweep-bw "12,16,20,24" --sweep-lat "20000,30000,45000"

# Save to file for analysis
./nexora --sweep > sweep_results.csv
```

## CLI Reference

Run `./nexora --help` for full documentation. Key flags organized by category:

**Hardware / Topology** (high impact):
- `--preset` - Hardware profile (recommended)
- `--tp` - Tensor parallelism degree
- `--algo` - Collective algorithm (ring/tree)

**Model Structure**:
- `--layers` - Number of transformer layers
- `--comm-ops` - Collectives per layer (typically 2-4)
- `--ar-bytes`, `--ag-bytes` - Payload sizes

**Runtime / Scheduling** (optimization levers):
- `--group` - Layer grouping (sync every N layers)
- `--bits` - Communication precision (16/8/4)
- `--overlap` - Comm/compute overlap fraction

**Optimization Guardrails**:
- `--max-risk` - Risk budget [0, 1]
- `--max-group` - Maximum group size for search

## Simulation Model

### Token Time Calculation

```
token_time = compute_time + visible_comm_time + kv_overhead

compute_time = layers × compute_ns
visible_comm_time = raw_comm × (1 - overlap)
raw_comm = num_collectives × (latency × algo_steps + transfer_time)
```

### Algorithm Overhead

- **Ring all-reduce**: `(tp-1)` latency steps, `2×(tp-1)/tp` data factor
- **Tree all-reduce**: `log2(tp)` latency steps

### Why Presets Matter

With `pcie4_bad` defaults:
- Latency per collective: 45 µs × 3 steps (Ring, TP=4) = 135 µs
- 64 collectives per token = 8.6 ms of just latency
- This dominates the 2.6 ms of compute time

## Testing

```bash
cargo test              # Run all tests
cargo test invariants   # Run specific module
```

Tests validate monotonicity invariants, TP=1 independence, and reality checks (e.g., PCIe4 bad must show ≥100% TP=4 overhead).

## License

MIT
