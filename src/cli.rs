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
use serde::Serialize;

use crate::model::{CollectiveAlgo, CommPrecision, HardwarePreset, ModelPreset, PrefillConfig, SimConfig, SimResult};

/// Complete JSON output for simulation results.
#[derive(Debug, Clone, Serialize)]
pub struct JsonOutput {
    pub preset: Option<String>,
    pub config: SimConfig,
    pub baseline: SimResult,
    pub baseline_risk: f64,
    pub tp_comparison: Option<TpComparison>,
    pub optimized: Option<OptimizedResult>,
    pub insights: Vec<String>,
}

/// TP comparison results.
#[derive(Debug, Clone, Serialize)]
pub struct TpComparison {
    pub tp1_ns: u64,
    pub tp1_tok_per_sec: f64,
    pub tp2_ns: u64,
    pub tp2_ratio: f64,
    pub tp2_overhead_pct: f64,
    pub tpn_ns: u64,
    pub tpn_ratio: f64,
    pub tpn_overhead_pct: f64,
}

/// Optimized configuration results.
#[derive(Debug, Clone, Serialize)]
pub struct OptimizedResult {
    pub config: OptimizedConfig,
    pub result: SimResult,
    pub risk: f64,
    pub speedup_ratio: f64,
    pub speedup_pct: f64,
    pub configs_evaluated: u32,
    pub configs_rejected_risk: u32,
}

/// Optimized configuration summary.
#[derive(Debug, Clone, Serialize)]
pub struct OptimizedConfig {
    pub group_size: u32,
    pub precision_bits: u32,
    pub overlap: f64,
}

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

    /// Pipeline parallelism degree (1 = no PP, 4 = 4-stage pipeline)
    #[arg(long, default_value = "1", help_heading = "Hardware / Topology")]
    pub pp: u32,

    /// Number of micro-batches for pipeline parallelism (reduces bubble overhead)
    #[arg(long, default_value = "1", help_heading = "Hardware / Topology")]
    pub micro_batches: u32,

    /// Hardware preset: pcie4_good, pcie4_bad, pcie5_good, pcie5_bad, nvlink, nvlink4, nvlink5, infinity_fabric, ib_hdr, ib_ndr
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

    /// HBM memory bandwidth per GPU in GB/s (auto from preset if not specified)
    #[arg(long, help_heading = "Hardware / Topology")]
    pub memory_bw_gbps: Option<f64>,

    // =========================================================================
    // Model Structure
    // =========================================================================

    /// Model preset: llama-7b, llama-13b, llama-70b, mixtral-8x7b, mixtral-8x22b
    /// [HIGH IMPACT: sets layers, hidden_dim, compute_ns]
    #[arg(long, help_heading = "Model Structure")]
    pub model: Option<String>,

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

    /// Batch size: tokens processed in parallel
    /// [HIGH IMPACT: larger batches amortize latency overhead]
    #[arg(long, default_value = "1", help_heading = "Model Structure")]
    pub batch_size: u32,

    /// Number of MoE experts (0 = dense model, 8 = typical Mixtral)
    #[arg(long, default_value = "0", help_heading = "Model Structure")]
    pub moe_experts: u32,

    /// Number of active experts per token (typical: 2 for top-k routing)
    #[arg(long, default_value = "2", help_heading = "Model Structure")]
    pub active_experts: u32,

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

    /// Compare multiple hardware presets side-by-side (comma-separated)
    /// Example: --compare "pcie4_bad,nvlink,nvlink4"
    #[arg(long, help_heading = "Modes")]
    pub compare: Option<String>,

    /// Show ASCII visualization of results
    #[arg(long, default_value = "false", help_heading = "Modes")]
    pub ascii: bool,

    // =========================================================================
    // Output Format
    // =========================================================================

    /// Output JSON format instead of text
    #[arg(long, default_value = "false", help_heading = "Output")]
    pub json: bool,

    /// Serve interactive HTML visualization on localhost
    #[arg(long, help_heading = "Output")]
    pub serve: Option<u16>,
}

impl Cli {
    /// Convert CLI args to a SimConfig for the baseline run.
    pub fn to_config(&self) -> SimConfig {
        // Start from model preset if specified
        let mut config = if let Some(model_str) = &self.model {
            if let Some(model) = ModelPreset::from_str(model_str) {
                SimConfig::from_model_preset(model)
            } else {
                SimConfig::default()
            }
        } else {
            SimConfig::default()
        };

        // Apply hardware preset if specified (overrides bw/latency/memory_bw)
        if let Some(preset_str) = &self.preset {
            if let Some(preset) = HardwarePreset::from_str(preset_str) {
                config.bw_gbps = preset.bandwidth_gbps();
                config.latency_ns = preset.latency_ns();
                // Use preset memory BW unless explicitly overridden
                config.memory_bw_gbps = self.memory_bw_gbps.unwrap_or(preset.memory_bw_gbps());
            }
        } else {
            // Use manual bw/latency if no hardware preset
            config.bw_gbps = self.bw_gbps;
            config.latency_ns = self.latency_ns;
            // Use explicit memory_bw or default
            if let Some(mem_bw) = self.memory_bw_gbps {
                config.memory_bw_gbps = mem_bw;
            }
        }

        let algo = CollectiveAlgo::from_str(&self.algo).unwrap_or(CollectiveAlgo::Ring);
        let precision = CommPrecision::from_bits(self.bits).unwrap_or(CommPrecision::FP16);

        // Override model params only if explicitly different from default
        // (allows CLI to override model preset)
        if self.model.is_none() || self.layers != 32 {
            config.layers = self.layers;
        }
        if self.model.is_none() || self.compute_ns != 80000 {
            config.compute_ns = self.compute_ns;
        }
        if self.model.is_none() || self.comm_ops != 2 {
            config.comm_ops_per_layer = self.comm_ops;
        }
        if self.model.is_none() || self.ar_bytes != 32768 {
            config.ar_bytes = self.msg_bytes.unwrap_or(self.ar_bytes);
        }

        // Always apply these settings
        config.tp = self.tp;
        config.pp = self.pp;
        config.micro_batches = self.micro_batches;
        config.ag_bytes = self.ag_bytes;
        config.algo = algo;
        config.group_size = self.group;
        config.precision = precision;
        config.overlap = self.overlap.clamp(0.0, 0.95);
        config.ctx_len = self.ctx_len;
        config.kv_bytes_base = self.kv_bytes_base;
        config.kv_sharded = !self.no_kv_shard;
        config.batch_size = self.batch_size;
        config.moe_experts = self.moe_experts;
        config.active_experts = self.active_experts;

        config
    }

    /// Get the model preset if specified.
    pub fn model_preset(&self) -> Option<ModelPreset> {
        self.model.as_ref().and_then(|s| ModelPreset::from_str(s))
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

    /// Parse compare preset names.
    pub fn parse_compare_presets(&self) -> Vec<HardwarePreset> {
        self.compare
            .as_ref()
            .map(|s| {
                s.split(',')
                    .filter_map(|name| HardwarePreset::from_str(name.trim()))
                    .collect()
            })
            .unwrap_or_default()
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

// ============================================================================
// ASCII Visualization
// ============================================================================

const BAR_WIDTH: usize = 40;

/// Create an ASCII bar of specified width based on fraction (0.0 to 1.0)
pub fn ascii_bar(fraction: f64, width: usize) -> String {
    let filled = (fraction.clamp(0.0, 1.0) * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("[{}{}]", "=".repeat(filled), " ".repeat(empty))
}

/// Print a labeled ASCII bar for a time component
pub fn print_time_bar(label: &str, ns: u64, total_ns: u64, time_str: &str) {
    let fraction = if total_ns > 0 {
        ns as f64 / total_ns as f64
    } else {
        0.0
    };
    let pct = fraction * 100.0;
    println!("  {:8} {} {:>5.1}%  {}",
        label,
        ascii_bar(fraction, BAR_WIDTH),
        pct,
        time_str
    );
}

/// Print a comparison bar showing relative performance
pub fn print_comparison_bar(label: &str, ns: u64, baseline_ns: u64, max_ns: u64, time_str: &str) {
    let fraction = if max_ns > 0 {
        ns as f64 / max_ns as f64
    } else {
        0.0
    };
    let overhead = if baseline_ns > 0 {
        (ns as f64 - baseline_ns as f64) / baseline_ns as f64 * 100.0
    } else {
        0.0
    };

    if ns == baseline_ns {
        println!("  {:6} {} {}  (baseline)",
            label,
            ascii_bar(fraction, BAR_WIDTH),
            time_str
        );
    } else {
        println!("  {:6} {} {}  {:+.0}%",
            label,
            ascii_bar(fraction, BAR_WIDTH),
            time_str,
            overhead
        );
    }
}
