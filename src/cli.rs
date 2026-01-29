//! Command-line interface for the GPU inference simulator.
//!
//! Parameters are organized into categories:
//!
//! ## Hardware / Topology
//! - `--tp`: Tensor parallelism degree (most impactful for comm overhead)
//! - `--preset`: Hardware profile (sets bw + latency; overrides manual values)
//! - `--algo`: Collective algorithm (ring/tree)
//! - `--bw-gbps`, `--latency-ns`: Manual hardware params (ignored if preset set)
//!
//! ## Model Structure
//! - `--layers`: Number of transformer layers
//! - `--comm-ops`: Collectives per layer (2-4 typical; high impact)
//! - `--ar-bytes`, `--ag-bytes`: Payload sizes (or `--msg-bytes` fallback)
//! - `--ctx-len`: Context length for decode KV-cache overhead
//! - `--kv-bytes-base`: KV read bytes per token per layer
//!
//! ## Runtime / Scheduling (Optimizations)
//! - `--group`: Layer grouping (sync every N layers)
//! - `--bits`: Communication precision (16/8/4)
//! - `--overlap`: Comm/compute overlap fraction [0, 0.95]
//! - `--ttft`, `--cache-aware`: Simulation mode flags
//!
//! ## Optimization Guardrails
//! - `--max-risk`: Risk budget for optimizer [0, 1]
//! - `--max-group`: Maximum group size for search
//!
//! ## Impact Summary
//! Parameters with highest impact on results:
//! 1. `--preset` / `--bw-gbps` / `--latency-ns` (hardware regime)
//! 2. `--tp` (communication overhead scaling)
//! 3. `--comm-ops` (collectives per layer)
//! 4. `--overlap` and `--group` (optimization levers)

use clap::Parser;

use crate::model::{CollectiveAlgo, CommPrecision, HardwarePreset, PrefillConfig, SimConfig};

#[derive(Parser, Debug)]
#[command(name = "nexora")]
#[command(author, version, about = "Multi-GPU LLM inference simulator for understanding tensor-parallel overhead")]
pub struct Cli {
    // =========================================================================
    // Hardware / Topology
    // =========================================================================

    /// Tensor parallelism degree (1 = single GPU, 4 = 4-way TP)
    /// [HIGH IMPACT: determines communication overhead]
    #[arg(long, default_value = "4", help_heading = "Hardware / Topology")]
    pub tp: u32,

    /// Hardware preset: pcie4_good, pcie4_bad, pcie5_good, pcie5_bad, nvlink
    /// [HIGH IMPACT: sets bandwidth and latency]
    #[arg(long, help_heading = "Hardware / Topology")]
    pub preset: Option<String>,

    /// Collective algorithm: ring or tree
    #[arg(long, default_value = "ring", help_heading = "Hardware / Topology")]
    pub algo: String,

    /// Inter-GPU bandwidth in GB/s (ignored if --preset is set)
    #[arg(long, default_value = "18.0", help_heading = "Hardware / Topology")]
    pub bw_gbps: f64,

    /// Fixed latency per collective step in nanoseconds (ignored if --preset is set)
    #[arg(long, default_value = "25000", help_heading = "Hardware / Topology")]
    pub latency_ns: u64,

    // =========================================================================
    // Model Structure
    // =========================================================================

    /// Number of transformer layers
    #[arg(long, default_value = "32", help_heading = "Model Structure")]
    pub layers: u32,

    /// Number of collective operations per layer (typically 2-4)
    /// [HIGH IMPACT: directly multiplies communication overhead]
    #[arg(long, default_value = "2", help_heading = "Model Structure")]
    pub comm_ops: u32,

    /// All-reduce payload per operation in bytes
    #[arg(long, default_value = "32768", help_heading = "Model Structure")]
    pub ar_bytes: u64,

    /// All-gather payload per operation in bytes (0 = AR only)
    #[arg(long, default_value = "0", help_heading = "Model Structure")]
    pub ag_bytes: u64,

    /// Legacy: message payload in bytes (fallback for --ar-bytes)
    #[arg(long, help_heading = "Model Structure")]
    pub msg_bytes: Option<u64>,

    /// Context length for decode (prompt + generated tokens so far)
    #[arg(long, default_value = "512", help_heading = "Model Structure")]
    pub ctx_len: u32,

    /// Base KV-cache read bytes per layer per context token
    #[arg(long, default_value = "128", help_heading = "Model Structure")]
    pub kv_bytes_base: u64,

    /// Disable KV sharding (all KV on single GPU; reduces cross-GPU traffic)
    #[arg(long, default_value = "false", help_heading = "Model Structure")]
    pub no_kv_shard: bool,

    /// Compute time per layer in nanoseconds (internal model parameter)
    #[arg(long, default_value = "80000", help_heading = "Model Structure")]
    pub compute_ns: u64,

    // =========================================================================
    // Runtime / Scheduling
    // =========================================================================

    /// Layer group size: sync every N layers (reduces collective count)
    /// [HIGH IMPACT: trades latency for throughput]
    #[arg(long, default_value = "1", help_heading = "Runtime / Scheduling")]
    pub group: u32,

    /// Communication precision bits (16, 8, or 4)
    #[arg(long, default_value = "16", help_heading = "Runtime / Scheduling")]
    pub bits: u32,

    /// Overlap fraction: portion of comm hidden under compute [0, 0.95]
    /// [HIGH IMPACT: can significantly reduce visible comm latency]
    #[arg(long, default_value = "0.0", help_heading = "Runtime / Scheduling")]
    pub overlap: f64,

    /// Enable TTFT (Time To First Token) simulation mode
    #[arg(long, default_value = "false", help_heading = "Runtime / Scheduling")]
    pub ttft: bool,

    /// Enable context-length-aware decode simulation
    #[arg(long, default_value = "false", help_heading = "Runtime / Scheduling")]
    pub cache_aware: bool,

    // =========================================================================
    // Optimization Guardrails
    // =========================================================================

    /// Maximum acceptable risk budget for optimizer [0, 1]
    #[arg(long, default_value = "0.3", help_heading = "Optimization Guardrails")]
    pub max_risk: f64,

    /// Maximum layer group size for optimization search
    #[arg(long, default_value = "8", help_heading = "Optimization Guardrails")]
    pub max_group: u32,

    /// Skip optimization, just run single simulation
    #[arg(long, default_value = "false", help_heading = "Optimization Guardrails")]
    pub no_opt: bool,

    // =========================================================================
    // TTFT / Prefill (used with --ttft)
    // =========================================================================

    /// Number of prompt tokens for prefill stage
    #[arg(long, default_value = "512", help_heading = "TTFT / Prefill")]
    pub prompt_tokens: u32,

    /// Alias for --prompt-tokens
    #[arg(long, help_heading = "TTFT / Prefill")]
    pub prompt_len: Option<u32>,

    /// Number of output tokens to generate
    #[arg(long, default_value = "128", help_heading = "TTFT / Prefill")]
    pub output_tokens: u32,

    /// Prefill compute scaling factor (attention overhead)
    #[arg(long, default_value = "1.5", help_heading = "TTFT / Prefill")]
    pub prefill_factor: f64,

    /// KV-cache write overhead per token per layer (ns)
    #[arg(long, default_value = "100", help_heading = "TTFT / Prefill")]
    pub kv_write_ns: u64,

    /// KV-cache read overhead per context token (ns) for decode
    #[arg(long, default_value = "10", help_heading = "TTFT / Prefill")]
    pub kv_read_ns: u64,

    // =========================================================================
    // Sweep / Benchmark Modes
    // =========================================================================

    /// Enable parameter sweep mode (CSV output)
    #[arg(long, default_value = "false", help_heading = "Modes")]
    pub sweep: bool,

    /// Bandwidth values to sweep (comma-separated, GB/s)
    #[arg(long, default_value = "12,16,20,24", help_heading = "Modes")]
    pub sweep_bw: String,

    /// Latency values to sweep (comma-separated, ns)
    #[arg(long, default_value = "20000,30000,45000", help_heading = "Modes")]
    pub sweep_lat: String,

    /// Context lengths to sweep (comma-separated)
    #[arg(long, default_value = "256,512,1024,2048", help_heading = "Modes")]
    pub sweep_ctx: String,

    /// Run micro-benchmark timing loop
    #[arg(long, default_value = "false", help_heading = "Modes")]
    pub bench: bool,

    /// Number of iterations for micro-benchmark
    #[arg(long, default_value = "1000000", help_heading = "Modes")]
    pub bench_iters: u64,

    /// Show detailed breakdown of overhead sources
    #[arg(long, default_value = "false", help_heading = "Modes")]
    pub verbose: bool,
}

impl Cli {
    /// Convert CLI args to a SimConfig for the baseline run.
    pub fn to_config(&self) -> SimConfig {
        // Start from preset if specified
        let mut config = if let Some(preset_str) = &self.preset {
            if let Some(preset) = HardwarePreset::from_str(preset_str) {
                SimConfig::from_preset(preset)
            } else {
                SimConfig::default()
            }
        } else {
            SimConfig {
                bw_gbps: self.bw_gbps,
                latency_ns: self.latency_ns,
                ..Default::default()
            }
        };

        let algo = CollectiveAlgo::from_str(&self.algo).unwrap_or(CollectiveAlgo::Ring);
        let precision = CommPrecision::from_bits(self.bits).unwrap_or(CommPrecision::FP16);

        // Override with CLI args
        config.layers = self.layers;
        config.tp = self.tp;
        config.compute_ns = self.compute_ns;
        config.comm_ops_per_layer = self.comm_ops;
        config.ar_bytes = self.msg_bytes.unwrap_or(self.ar_bytes);
        config.ag_bytes = self.ag_bytes;
        config.algo = algo;
        config.group_size = self.group;
        config.precision = precision;
        config.overlap = self.overlap.clamp(0.0, 0.95);
        config.ctx_len = self.ctx_len;
        config.kv_bytes_base = self.kv_bytes_base;
        config.kv_sharded = !self.no_kv_shard;

        // If preset not specified, use CLI bw/latency
        if self.preset.is_none() {
            config.bw_gbps = self.bw_gbps;
            config.latency_ns = self.latency_ns;
        }

        config
    }

    /// Convert CLI args to a PrefillConfig for TTFT simulation.
    pub fn to_prefill_config(&self) -> PrefillConfig {
        PrefillConfig {
            base: self.to_config(),
            prompt_tokens: self.prompt_len.unwrap_or(self.prompt_tokens),
            prefill_compute_factor: self.prefill_factor,
            kv_cache_write_ns: self.kv_write_ns,
        }
    }

    /// Parse sweep bandwidth values.
    pub fn parse_sweep_bw(&self) -> Vec<f64> {
        self.sweep_bw
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect()
    }

    /// Parse sweep latency values.
    pub fn parse_sweep_lat(&self) -> Vec<u64> {
        self.sweep_lat
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect()
    }

    /// Parse sweep context lengths.
    pub fn parse_sweep_ctx(&self) -> Vec<u32> {
        self.sweep_ctx
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect()
    }
}

/// Format nanoseconds as human-readable string.
pub fn format_ns(ns: u64) -> String {
    if ns >= 1_000_000_000 {
        format!("{:.2} s", ns as f64 / 1e9)
    } else if ns >= 1_000_000 {
        format!("{:.2} ms", ns as f64 / 1e6)
    } else if ns >= 1_000 {
        format!("{:.2} us", ns as f64 / 1e3)
    } else {
        format!("{} ns", ns)
    }
}

/// Format tokens per second.
pub fn format_tps(tps: f64) -> String {
    if tps >= 1000.0 {
        format!("{:.1}K tok/s", tps / 1000.0)
    } else {
        format!("{:.1} tok/s", tps)
    }
}

/// Format percentage.
pub fn format_pct(value: f64, total: f64) -> String {
    if total > 0.0 {
        format!("{:.1}%", 100.0 * value / total)
    } else {
        "0.0%".to_string()
    }
}
