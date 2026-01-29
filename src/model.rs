//! Core simulation model for multi-GPU LLM inference latency.
//!
//! Models decoding token time as repeated steps across transformer layers,
//! accounting for compute time, communication overhead, and various optimizations.
//!
//! # Allocation-Free Hot Path
//!
//! The [`simulate_token`] function is designed to be completely allocation-free.
//! All computation uses stack-based values, and the result is returned by value.
//! This allows extremely high throughput simulation (>50M calls/sec on modern CPUs).
//!
//! # Realistic Communication Model
//!
//! Real tensor-parallel inference has multiple collectives per layer:
//! - Attention: all-reduce after output projection
//! - MLP: all-reduce after down projection (and possibly all-gather)
//! - KV-cache: may require cross-GPU reads for sharded caches
//!
//! The model supports:
//! - `comm_ops_per_layer`: number of collective ops (typically 2-4)
//! - Separate AR (all-reduce) and AG (all-gather) byte sizes
//! - Context-length dependent KV-cache overhead

/// Collective algorithm choice for tensor-parallel communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectiveAlgo {
    Ring,
    Tree,
}

impl CollectiveAlgo {
    /// Parse from string, case-insensitive.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ring" => Some(Self::Ring),
            "tree" => Some(Self::Tree),
            _ => None,
        }
    }

    /// Algorithm-specific overhead factor.
    /// Ring: 2*(p-1)/p factor for all-reduce
    /// Tree: log2(p) steps with higher per-step bandwidth
    #[inline]
    pub fn overhead_factor(&self, tp: u32) -> f64 {
        if tp <= 1 {
            return 0.0;
        }
        match self {
            Self::Ring => {
                // Ring all-reduce: 2*(p-1)/p data transfers
                2.0 * (tp as f64 - 1.0) / tp as f64
            }
            Self::Tree => {
                // Tree: log2(p) steps, but each transfers full data
                (tp as f64).log2()
            }
        }
    }

    /// All-gather factor (different from all-reduce).
    /// Each GPU sends its shard to all others.
    #[inline]
    pub fn allgather_factor(&self, tp: u32) -> f64 {
        if tp <= 1 {
            return 0.0;
        }
        // All-gather: (p-1)/p of total data transferred by each GPU
        (tp as f64 - 1.0) / tp as f64
    }
}

/// Communication precision for reduced-precision collectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommPrecision {
    FP16,
    FP8,
    FP4,
}

impl CommPrecision {
    pub fn bits(&self) -> u32 {
        match self {
            Self::FP16 => 16,
            Self::FP8 => 8,
            Self::FP4 => 4,
        }
    }

    pub fn from_bits(bits: u32) -> Option<Self> {
        match bits {
            16 => Some(Self::FP16),
            8 => Some(Self::FP8),
            4 => Some(Self::FP4),
            _ => None,
        }
    }

    /// Scale factor relative to FP16 baseline.
    #[inline]
    pub fn scale_factor(&self) -> f64 {
        self.bits() as f64 / 16.0
    }
}

// ============================================================================
// Hardware Profile Presets
// ============================================================================

/// Hardware profile preset for realistic parameter regimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwarePreset {
    /// PCIe 4.0 x16 with good topology (direct P2P)
    Pcie4Good,
    /// PCIe 4.0 x16 with bad topology (through CPU/switch)
    Pcie4Bad,
    /// PCIe 5.0 x16 with good topology
    Pcie5Good,
    /// PCIe 5.0 x16 with bad topology
    Pcie5Bad,
    /// NVLink 3.0 (A100)
    NvLink,
    /// NVLink 4.0 (H100)
    NvLink4,
    /// NVLink 5.0 (B200)
    NvLink5,
    /// AMD Infinity Fabric (MI300)
    InfinityFabric,
    /// InfiniBand HDR (200 Gb/s)
    IbHdr,
    /// InfiniBand NDR (400 Gb/s)
    IbNdr,
    /// Custom (use CLI params)
    Custom,
}

impl HardwarePreset {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "pcie4_good" | "pcie4good" => Some(Self::Pcie4Good),
            "pcie4_bad" | "pcie4bad" => Some(Self::Pcie4Bad),
            "pcie5_good" | "pcie5good" => Some(Self::Pcie5Good),
            "pcie5_bad" | "pcie5bad" => Some(Self::Pcie5Bad),
            "nvlink" | "nvlink3" => Some(Self::NvLink),
            "nvlink4" => Some(Self::NvLink4),
            "nvlink5" => Some(Self::NvLink5),
            "infinity_fabric" | "infinityfabric" | "mi300" => Some(Self::InfinityFabric),
            "ib_hdr" | "ibhdr" | "infiniband_hdr" => Some(Self::IbHdr),
            "ib_ndr" | "ibndr" | "infiniband_ndr" => Some(Self::IbNdr),
            "custom" => Some(Self::Custom),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Pcie4Good => "PCIe4 Good Topology",
            Self::Pcie4Bad => "PCIe4 Bad Topology",
            Self::Pcie5Good => "PCIe5 Good Topology",
            Self::Pcie5Bad => "PCIe5 Bad Topology",
            Self::NvLink => "NVLink 3.0 (A100)",
            Self::NvLink4 => "NVLink 4.0 (H100)",
            Self::NvLink5 => "NVLink 5.0 (B200)",
            Self::InfinityFabric => "AMD Infinity Fabric (MI300)",
            Self::IbHdr => "InfiniBand HDR",
            Self::IbNdr => "InfiniBand NDR",
            Self::Custom => "Custom",
        }
    }

    /// Get bandwidth in GB/s for this preset.
    pub fn bandwidth_gbps(&self) -> f64 {
        match self {
            Self::Pcie4Good => 18.0,       // ~56% of theoretical 32 GB/s
            Self::Pcie4Bad => 12.0,        // ~38% due to switch/CPU hop
            Self::Pcie5Good => 28.0,       // ~44% of theoretical 64 GB/s
            Self::Pcie5Bad => 18.0,        // ~28% with bad topology
            Self::NvLink => 150.0,         // NVLink 3.0 per direction (A100)
            Self::NvLink4 => 225.0,        // NVLink 4.0 per direction (H100: 450/2)
            Self::NvLink5 => 450.0,        // NVLink 5.0 per direction (B200: 900/2)
            Self::InfinityFabric => 200.0, // AMD Infinity Fabric (MI300)
            Self::IbHdr => 25.0,           // InfiniBand HDR: 200 Gb/s = 25 GB/s
            Self::IbNdr => 50.0,           // InfiniBand NDR: 400 Gb/s = 50 GB/s
            Self::Custom => 20.0,
        }
    }

    /// Get per-collective latency in nanoseconds.
    pub fn latency_ns(&self) -> u64 {
        match self {
            Self::Pcie4Good => 25_000,     // 25 µs
            Self::Pcie4Bad => 45_000,      // 45 µs (extra hop)
            Self::Pcie5Good => 18_000,     // 18 µs
            Self::Pcie5Bad => 35_000,      // 35 µs
            Self::NvLink => 3_000,         // 3 µs
            Self::NvLink4 => 2_000,        // 2 µs (H100)
            Self::NvLink5 => 1_500,        // 1.5 µs (B200)
            Self::InfinityFabric => 3_000, // 3 µs (similar to NVLink 3)
            Self::IbHdr => 1_500,          // 1.5 µs (RDMA)
            Self::IbNdr => 1_000,          // 1 µs (RDMA)
            Self::Custom => 5_000,
        }
    }
}

// ============================================================================
// Simulation Configuration
// ============================================================================

/// Configuration for a single simulation run.
/// All times in nanoseconds, bandwidth in GB/s.
#[derive(Debug, Clone)]
pub struct SimConfig {
    /// Number of transformer layers.
    pub layers: u32,
    /// Tensor parallelism degree (1 = no TP, 4 = 4-way split).
    pub tp: u32,
    /// Compute time per layer in nanoseconds.
    pub compute_ns: u64,

    // Communication parameters (realistic multi-collective model)

    /// Number of collective operations per layer (typically 2-4).
    /// - 2: minimal (attention AR + MLP AR)
    /// - 4: with all-gather ops
    pub comm_ops_per_layer: u32,
    /// All-reduce payload per operation in bytes.
    pub ar_bytes: u64,
    /// All-gather payload per operation in bytes (0 = no AG ops).
    pub ag_bytes: u64,
    /// Effective inter-GPU bandwidth in GB/s.
    pub bw_gbps: f64,
    /// Fixed latency per collective operation in nanoseconds.
    pub latency_ns: u64,
    /// Collective algorithm choice.
    pub algo: CollectiveAlgo,
    /// Layer grouping: perform one collective per G layers.
    pub group_size: u32,
    /// Communication precision.
    pub precision: CommPrecision,
    /// Overlap fraction: portion of comm hidden under compute [0, 0.95].
    pub overlap: f64,

    // KV-cache parameters for context-aware simulation

    /// Current context length (for decode with KV-cache).
    pub ctx_len: u32,
    /// Base KV read bytes per layer per context token (linear scaling).
    pub kv_bytes_base: u64,
    /// Whether KV-cache is sharded across GPUs (adds cross-GPU traffic).
    pub kv_sharded: bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        // Realistic defaults for a ~7B model decode on PCIe 4.0
        Self {
            layers: 32,
            tp: 4,
            compute_ns: 80_000,       // 80 µs per layer (decode is memory-bound)
            comm_ops_per_layer: 2,    // AR after attention + AR after MLP
            ar_bytes: 32_768,         // 32 KB per AR (hidden_dim * 2 bytes * batch)
            ag_bytes: 0,              // No AG by default
            bw_gbps: 18.0,            // Realistic PCIe 4.0
            latency_ns: 25_000,       // 25 µs per collective
            algo: CollectiveAlgo::Ring,
            group_size: 1,
            precision: CommPrecision::FP16,
            overlap: 0.0,
            ctx_len: 512,
            kv_bytes_base: 128,       // Bytes per token per layer for KV read (linear)
            kv_sharded: true,
        }
    }
}

impl SimConfig {
    /// Create config from a hardware preset.
    pub fn from_preset(preset: HardwarePreset) -> Self {
        Self {
            bw_gbps: preset.bandwidth_gbps(),
            latency_ns: preset.latency_ns(),
            ..Default::default()
        }
    }
}

/// Result of a simulation run.
#[derive(Debug, Clone, Copy)]
pub struct SimResult {
    /// Total time per token in nanoseconds.
    pub ns_per_token: u64,
    /// Tokens per second (simulated).
    pub tok_per_sec: f64,
    /// Total compute time contribution in nanoseconds.
    pub compute_total_ns: u64,
    /// Total communication time contribution in nanoseconds (after overlap).
    pub comm_total_ns: u64,
    /// KV-cache overhead in nanoseconds.
    pub kv_overhead_ns: u64,
    /// Number of collective operations performed.
    pub num_collectives: u32,
}

/// Compute communication time for a single collective operation.
/// Returns time in nanoseconds.
#[inline]
pub fn comm_time_ns(
    payload_bytes: u64,
    bw_gbps: f64,
    latency_ns: u64,
    algo: CollectiveAlgo,
    tp: u32,
    is_allgather: bool,
) -> u64 {
    if tp <= 1 {
        return 0;
    }

    // Bandwidth in bytes/ns = GB/s
    let bw_bytes_per_ns = bw_gbps;

    // Algorithm overhead factor for data movement
    let algo_factor = if is_allgather {
        algo.allgather_factor(tp)
    } else {
        algo.overhead_factor(tp)
    };

    // Transfer time = (payload * algo_factor) / bandwidth
    let transfer_ns = (payload_bytes as f64 * algo_factor) / bw_bytes_per_ns;

    // Latency scales with algorithm steps:
    // Ring: (tp-1) steps, Tree: log2(tp) steps
    let latency_steps = match algo {
        CollectiveAlgo::Ring => (tp - 1) as u64,
        CollectiveAlgo::Tree => ((tp as f64).log2().ceil() as u64).max(1),
    };
    let total_latency_ns = latency_ns * latency_steps;

    // Total = algorithm latency + transfer time
    total_latency_ns + transfer_ns.ceil() as u64
}

/// Compute KV-cache read overhead for decode.
/// Uses simple linear scaling: bytes = kv_bytes_base * ctx_len * layers.
#[inline]
pub fn kv_cache_overhead_ns(
    ctx_len: u32,
    layers: u32,
    kv_bytes_base: u64,
    tp: u32,
    kv_sharded: bool,
    bw_gbps: f64,
    latency_ns: u64,
) -> u64 {
    if ctx_len == 0 || kv_bytes_base == 0 {
        return 0;
    }

    // Total KV bytes to read: base * ctx_len * layers (linear scaling)
    let kv_bytes = kv_bytes_base as f64
        * ctx_len as f64
        * layers as f64;

    // If KV is sharded and TP>1, add cross-GPU traffic overhead
    let cross_gpu_factor = if kv_sharded && tp > 1 {
        // Each GPU may need to read from other GPUs for attention
        // Simplified: add (tp-1)/tp overhead factor
        1.0 + (tp - 1) as f64 / tp as f64 * 0.3
    } else {
        1.0
    };

    let effective_bytes = kv_bytes * cross_gpu_factor;

    // Memory bandwidth limited (simplified model)
    // For cross-GPU, add latency per "chunk" of data
    let transfer_ns = effective_bytes / bw_gbps;

    // Add latency penalty for sharded access
    let latency_overhead = if kv_sharded && tp > 1 {
        latency_ns / 4 // Partial latency for KV access
    } else {
        0
    };

    (transfer_ns.ceil() as u64) + latency_overhead
}

/// Run the simulation for a single token decode step.
/// Returns SimResult with timing breakdown.
///
/// INVARIANT: No heap allocations in this function (hot path).
#[inline]
pub fn simulate_token(config: &SimConfig) -> SimResult {
    // Total compute time: all layers
    let compute_total_ns = config.compute_ns * config.layers as u64;

    // For TP=1, no communication overhead (but still KV cache cost)
    if config.tp <= 1 {
        let kv_overhead = kv_cache_overhead_ns(
            config.ctx_len,
            config.layers,
            config.kv_bytes_base,
            1,
            false, // Not sharded for TP=1
            config.bw_gbps * 5.0, // Local memory is faster
            0,
        );

        let ns_per_token = compute_total_ns + kv_overhead;
        return SimResult {
            ns_per_token,
            tok_per_sec: 1e9 / ns_per_token as f64,
            compute_total_ns,
            comm_total_ns: 0,
            kv_overhead_ns: kv_overhead,
            num_collectives: 0,
        };
    }

    // Number of collective operations with layer grouping
    let group_size = config.group_size.max(1);
    let groups = (config.layers + group_size - 1) / group_size;

    // Each group has comm_ops_per_layer * layers_in_group collectives
    // But with grouping, we fuse them into fewer larger collectives
    let ops_per_group = config.comm_ops_per_layer;
    let num_collectives = groups * ops_per_group;

    // Payload per collective: sum across grouped layers, scaled by precision
    let layers_in_group = group_size.min(config.layers);

    // Calculate AR time
    let ar_payload = config.ar_bytes * layers_in_group as u64;
    let ar_scaled = (ar_payload as f64 * config.precision.scale_factor()).ceil() as u64;
    let ar_time = comm_time_ns(
        ar_scaled,
        config.bw_gbps,
        config.latency_ns,
        config.algo,
        config.tp,
        false,
    );

    // Calculate AG time (if applicable)
    let ag_time = if config.ag_bytes > 0 {
        let ag_payload = config.ag_bytes * layers_in_group as u64;
        let ag_scaled = (ag_payload as f64 * config.precision.scale_factor()).ceil() as u64;
        comm_time_ns(
            ag_scaled,
            config.bw_gbps,
            config.latency_ns,
            config.algo,
            config.tp,
            true,
        )
    } else {
        0
    };

    // Time per collective = AR time + AG time (if both exist)
    // If AG is disabled (ag_bytes=0), all ops are AR
    // Otherwise, assume AR ops and AG ops alternate
    let (ar_ops, ag_ops) = if config.ag_bytes == 0 {
        (ops_per_group, 0)
    } else {
        ((ops_per_group + 1) / 2, ops_per_group / 2)
    };
    let comm_per_group = ar_ops as u64 * ar_time + ag_ops as u64 * ag_time;

    // Total raw communication time
    let raw_comm_total_ns = comm_per_group * groups as u64;

    // Apply overlap: fraction of comm hidden under compute
    let overlap_clamped = config.overlap.clamp(0.0, 0.95);
    let visible_comm_ns = (raw_comm_total_ns as f64 * (1.0 - overlap_clamped)).ceil() as u64;

    // KV-cache overhead (sharded across GPUs)
    let kv_overhead = kv_cache_overhead_ns(
        config.ctx_len,
        config.layers,
        config.kv_bytes_base,
        config.tp,
        config.kv_sharded,
        config.bw_gbps,
        config.latency_ns,
    );

    // Total token time
    let ns_per_token = compute_total_ns + visible_comm_ns + kv_overhead;

    SimResult {
        ns_per_token,
        tok_per_sec: if ns_per_token > 0 { 1e9 / ns_per_token as f64 } else { 0.0 },
        compute_total_ns,
        comm_total_ns: visible_comm_ns,
        kv_overhead_ns: kv_overhead,
        num_collectives,
    }
}

/// Estimate "risk" of a configuration (toy model).
/// Higher group sizes and lower precision increase risk of accuracy degradation.
/// Returns a value in [0, 1].
#[inline]
pub fn risk_estimate(group_size: u32, precision: CommPrecision, overlap: f64) -> f64 {
    // Base risk from grouping: increases sharply beyond threshold
    let group_risk = if group_size <= 1 {
        0.0
    } else if group_size <= 4 {
        0.05 * (group_size as f64 - 1.0)
    } else {
        // Sharp increase beyond 4
        0.15 + 0.15 * ((group_size as f64 - 4.0).ln() + 1.0)
    };

    // Risk from reduced precision (FP4 is risky without careful implementation)
    let precision_risk = match precision {
        CommPrecision::FP16 => 0.0,
        CommPrecision::FP8 => 0.12,
        CommPrecision::FP4 => 0.40, // Higher risk for FP4
    };

    // High overlap with low precision is especially risky
    let interaction_risk = if precision == CommPrecision::FP4 && overlap < 0.5 {
        0.1 // Penalty for FP4 without overlap to hide latency
    } else {
        0.0
    };

    // Combined risk (capped at 1.0)
    (group_risk + precision_risk + interaction_risk).min(1.0)
}

// ============================================================================
// TTFT (Time To First Token) / Prefill Modeling
// ============================================================================

/// Configuration for prefill (prompt processing) stage.
/// Extends SimConfig with prompt-specific parameters.
#[derive(Debug, Clone)]
pub struct PrefillConfig {
    /// Base simulation config (shared with decode).
    pub base: SimConfig,
    /// Number of prompt tokens to process.
    pub prompt_tokens: u32,
    /// Compute time scaling factor for prefill (typically > 1 due to attention).
    /// Prefill compute scales with sequence length for attention.
    pub prefill_compute_factor: f64,
    /// KV-cache write overhead per token per layer (nanoseconds).
    /// Models the cost of writing key-value cache entries.
    pub kv_cache_write_ns: u64,
}

impl Default for PrefillConfig {
    fn default() -> Self {
        Self {
            base: SimConfig::default(),
            prompt_tokens: 512,
            prefill_compute_factor: 1.5, // Attention is more expensive than MLP alone
            kv_cache_write_ns: 100,      // Small overhead per KV write
        }
    }
}

/// Result of prefill simulation.
#[derive(Debug, Clone, Copy)]
pub struct PrefillResult {
    /// Time to first token (TTFT) in nanoseconds.
    pub ttft_ns: u64,
    /// Compute time for prefill in nanoseconds.
    pub compute_ns: u64,
    /// Communication time for prefill in nanoseconds.
    pub comm_ns: u64,
    /// KV-cache write overhead in nanoseconds.
    pub kv_cache_ns: u64,
    /// Number of collectives during prefill.
    pub num_collectives: u32,
}

/// Result of a full request (prefill + decode).
#[derive(Debug, Clone, Copy)]
pub struct FullRequestResult {
    /// Prefill stage result.
    pub prefill: PrefillResult,
    /// Per-token decode result.
    pub decode: SimResult,
    /// Total time for the full request in nanoseconds.
    pub total_ns: u64,
    /// Effective tokens per second (total output tokens / total time).
    pub effective_tok_per_sec: f64,
}

/// Simulate the prefill (prompt processing) stage.
/// Models TTFT including:
/// - Compute time scaled by sequence length and prefill factor
/// - Communication for TP (scales with prompt length)
/// - KV-cache write overhead
///
/// INVARIANT: No heap allocations in this function.
#[inline]
pub fn simulate_prefill(config: &PrefillConfig) -> PrefillResult {
    let base = &config.base;

    // Prefill compute: process all prompt tokens through all layers
    // Scale by prefill_compute_factor (attention is O(n^2) but we approximate linearly)
    let base_compute = base.compute_ns * base.layers as u64;
    let prefill_compute = (base_compute as f64
        * config.prompt_tokens as f64
        * config.prefill_compute_factor)
        .ceil() as u64;

    // KV-cache write overhead: per token, per layer
    let kv_cache_ns = config.kv_cache_write_ns * config.prompt_tokens as u64 * base.layers as u64;

    // For TP=1, no communication
    if base.tp <= 1 {
        let ttft_ns = prefill_compute + kv_cache_ns;
        return PrefillResult {
            ttft_ns,
            compute_ns: prefill_compute,
            comm_ns: 0,
            kv_cache_ns,
            num_collectives: 0,
        };
    }

    // Communication during prefill: same collectives but with larger activations
    let group_size = base.group_size.max(1);
    let groups = (base.layers + group_size - 1) / group_size;
    let ops_per_group = base.comm_ops_per_layer;
    let num_collectives = groups * ops_per_group;

    // Prefill message size scales with prompt tokens
    let layers_in_group = group_size.min(base.layers);
    let ar_payload = base.ar_bytes * layers_in_group as u64 * config.prompt_tokens as u64;
    let ar_scaled = (ar_payload as f64 * base.precision.scale_factor()).ceil() as u64;

    // For prefill, the large batch allows better bandwidth utilization
    // so latency is relatively smaller portion
    let ar_time = comm_time_ns(
        ar_scaled,
        base.bw_gbps,
        base.latency_ns,
        base.algo,
        base.tp,
        false,
    );

    let ag_time = if base.ag_bytes > 0 {
        let ag_payload = base.ag_bytes * layers_in_group as u64 * config.prompt_tokens as u64;
        let ag_scaled = (ag_payload as f64 * base.precision.scale_factor()).ceil() as u64;
        comm_time_ns(ag_scaled, base.bw_gbps, base.latency_ns, base.algo, base.tp, true)
    } else {
        0
    };

    let ar_ops = (ops_per_group + 1) / 2;
    let ag_ops = ops_per_group / 2;
    let comm_per_group = ar_ops as u64 * ar_time + ag_ops as u64 * ag_time;
    let raw_comm_ns = comm_per_group * groups as u64;

    let overlap_clamped = base.overlap.clamp(0.0, 0.95);
    let visible_comm_ns = (raw_comm_ns as f64 * (1.0 - overlap_clamped)).ceil() as u64;

    let ttft_ns = prefill_compute + visible_comm_ns + kv_cache_ns;

    PrefillResult {
        ttft_ns,
        compute_ns: prefill_compute,
        comm_ns: visible_comm_ns,
        kv_cache_ns,
        num_collectives,
    }
}

/// Simulate a full request: prefill + N decode steps.
///
/// INVARIANT: No heap allocations in this function.
#[inline]
pub fn simulate_full_request(
    prefill_config: &PrefillConfig,
    output_tokens: u32,
) -> FullRequestResult {
    let prefill = simulate_prefill(prefill_config);

    // For decode, context grows with each token
    // Use average context length for simplicity
    let avg_ctx = prefill_config.prompt_tokens + output_tokens / 2;
    let decode_config = SimConfig {
        ctx_len: avg_ctx,
        ..prefill_config.base.clone()
    };
    let decode = simulate_token(&decode_config);

    // Total time = TTFT + (output_tokens * decode_time)
    let decode_total_ns = decode.ns_per_token * output_tokens as u64;
    let total_ns = prefill.ttft_ns + decode_total_ns;

    // Effective throughput: output tokens / total time
    let effective_tok_per_sec = if total_ns > 0 {
        output_tokens as f64 * 1e9 / total_ns as f64
    } else {
        0.0
    };

    FullRequestResult {
        prefill,
        decode,
        total_ns,
        effective_tok_per_sec,
    }
}

/// Configuration for decode stage with KV-cache considerations.
/// Extends SimConfig with context-length scaling.
#[derive(Debug, Clone)]
pub struct DecodeWithCacheConfig {
    /// Base simulation config.
    pub base: SimConfig,
    /// Current context length (prompt + generated so far).
    pub context_length: u32,
    /// KV-cache read overhead per attention head per layer (ns).
    /// Models memory bandwidth for reading cached keys/values.
    pub kv_cache_read_ns_per_token: u64,
}

impl Default for DecodeWithCacheConfig {
    fn default() -> Self {
        Self {
            base: SimConfig::default(),
            context_length: 512,
            kv_cache_read_ns_per_token: 10, // Small per-token read overhead
        }
    }
}

/// Simulate a single decode step with KV-cache overhead.
/// The overhead scales linearly with context length.
///
/// INVARIANT: No heap allocations.
#[inline]
pub fn simulate_token_with_cache(config: &DecodeWithCacheConfig) -> SimResult {
    // Use the built-in ctx_len support
    let cfg = SimConfig {
        ctx_len: config.context_length,
        kv_bytes_base: config.kv_cache_read_ns_per_token as u64 * 10, // Scale factor
        ..config.base.clone()
    };
    simulate_token(&cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tp1_no_comm() {
        let config = SimConfig {
            tp: 1,
            layers: 10,
            compute_ns: 1000,
            ctx_len: 0,
            kv_bytes_base: 0,
            ..Default::default()
        };
        let result = simulate_token(&config);
        assert_eq!(result.comm_total_ns, 0);
        assert_eq!(result.num_collectives, 0);
        assert_eq!(result.ns_per_token, 10_000);
    }

    #[test]
    fn test_tp1_independent_of_comm_params() {
        let base = SimConfig {
            tp: 1,
            layers: 10,
            compute_ns: 1000,
            ctx_len: 0,
            kv_bytes_base: 0,
            ..Default::default()
        };

        // Vary algo, group, bits, overlap - should not affect TP=1
        let configs = vec![
            SimConfig { algo: CollectiveAlgo::Tree, ..base.clone() },
            SimConfig { group_size: 5, ..base.clone() },
            SimConfig { precision: CommPrecision::FP4, ..base.clone() },
            SimConfig { overlap: 0.9, ..base.clone() },
            SimConfig { comm_ops_per_layer: 4, ..base.clone() },
        ];

        let base_result = simulate_token(&base);
        for cfg in &configs {
            let result = simulate_token(cfg);
            assert_eq!(result.ns_per_token, base_result.ns_per_token);
        }
    }

    #[test]
    fn test_comm_ops_increases_comm() {
        let base = SimConfig {
            tp: 4,
            layers: 32,
            ctx_len: 0,
            kv_bytes_base: 0,
            ..Default::default()
        };

        let r1 = simulate_token(&SimConfig { comm_ops_per_layer: 1, ..base.clone() });
        let r2 = simulate_token(&SimConfig { comm_ops_per_layer: 2, ..base.clone() });
        let r4 = simulate_token(&SimConfig { comm_ops_per_layer: 4, ..base.clone() });

        assert!(r2.comm_total_ns > r1.comm_total_ns);
        assert!(r4.comm_total_ns > r2.comm_total_ns);
    }

    #[test]
    fn test_comm_time_increases_with_payload() {
        let t1 = comm_time_ns(1000, 10.0, 100, CollectiveAlgo::Ring, 4, false);
        let t2 = comm_time_ns(2000, 10.0, 100, CollectiveAlgo::Ring, 4, false);
        assert!(t2 >= t1);
    }

    #[test]
    fn test_comm_time_increases_with_latency() {
        let t1 = comm_time_ns(1000, 10.0, 100, CollectiveAlgo::Ring, 4, false);
        let t2 = comm_time_ns(1000, 10.0, 200, CollectiveAlgo::Ring, 4, false);
        assert!(t2 >= t1);
    }

    #[test]
    fn test_comm_time_decreases_with_bandwidth() {
        let t1 = comm_time_ns(1000, 10.0, 100, CollectiveAlgo::Ring, 4, false);
        let t2 = comm_time_ns(1000, 20.0, 100, CollectiveAlgo::Ring, 4, false);
        assert!(t2 <= t1);
    }

    #[test]
    fn test_ring_vs_tree_differ() {
        let ring = comm_time_ns(10000, 10.0, 100, CollectiveAlgo::Ring, 4, false);
        let tree = comm_time_ns(10000, 10.0, 100, CollectiveAlgo::Tree, 4, false);
        assert_ne!(ring, tree);
    }

    #[test]
    fn test_precision_scale() {
        assert_eq!(CommPrecision::FP16.scale_factor(), 1.0);
        assert_eq!(CommPrecision::FP8.scale_factor(), 0.5);
        assert_eq!(CommPrecision::FP4.scale_factor(), 0.25);
    }

    #[test]
    fn test_risk_increases_with_group() {
        let r1 = risk_estimate(1, CommPrecision::FP16, 0.0);
        let r4 = risk_estimate(4, CommPrecision::FP16, 0.0);
        let r8 = risk_estimate(8, CommPrecision::FP16, 0.0);
        assert!(r4 > r1);
        assert!(r8 > r4);
        // Sharp increase after 4
        assert!(r8 - r4 > r4 - r1);
    }

    #[test]
    fn test_risk_increases_with_lower_precision() {
        let r16 = risk_estimate(1, CommPrecision::FP16, 0.0);
        let r8 = risk_estimate(1, CommPrecision::FP8, 0.0);
        let r4 = risk_estimate(1, CommPrecision::FP4, 0.0);
        assert!(r8 > r16);
        assert!(r4 > r8);
    }

    #[test]
    fn test_preset_values() {
        let pcie4_bad = HardwarePreset::Pcie4Bad;
        assert!(pcie4_bad.bandwidth_gbps() < HardwarePreset::Pcie4Good.bandwidth_gbps());
        assert!(pcie4_bad.latency_ns() > HardwarePreset::Pcie4Good.latency_ns());
    }

    // TTFT / Prefill tests

    #[test]
    fn test_prefill_tp1_no_comm() {
        let config = PrefillConfig {
            base: SimConfig { tp: 1, layers: 10, compute_ns: 1000, ..Default::default() },
            prompt_tokens: 100,
            prefill_compute_factor: 1.0,
            kv_cache_write_ns: 10,
        };
        let result = simulate_prefill(&config);
        assert_eq!(result.comm_ns, 0);
        assert_eq!(result.num_collectives, 0);
    }

    #[test]
    fn test_prefill_scales_with_prompt_tokens() {
        let base_config = PrefillConfig {
            base: SimConfig { tp: 1, ..Default::default() },
            prompt_tokens: 100,
            prefill_compute_factor: 1.0,
            kv_cache_write_ns: 0,
        };

        let r1 = simulate_prefill(&base_config);
        let r2 = simulate_prefill(&PrefillConfig { prompt_tokens: 200, ..base_config.clone() });

        // Should scale linearly with prompt tokens
        assert_eq!(r2.compute_ns, r1.compute_ns * 2);
    }

    #[test]
    fn test_full_request_combines_prefill_and_decode() {
        let prefill_config = PrefillConfig {
            base: SimConfig {
                tp: 1,
                layers: 10,
                compute_ns: 1000,
                ctx_len: 0,
                kv_bytes_base: 0,
                ..Default::default()
            },
            prompt_tokens: 100,
            prefill_compute_factor: 1.0,
            kv_cache_write_ns: 0,
        };

        let output_tokens = 50;
        let result = simulate_full_request(&prefill_config, output_tokens);

        // Verify prefill is included
        assert!(result.prefill.ttft_ns > 0);

        // Verify decode is included
        assert!(result.decode.ns_per_token > 0);
    }

    #[test]
    fn test_ctx_len_increases_decode_time() {
        let base = SimConfig::default();

        let r1 = simulate_token(&SimConfig { ctx_len: 100, ..base.clone() });
        let r2 = simulate_token(&SimConfig { ctx_len: 1000, ..base.clone() });
        let r3 = simulate_token(&SimConfig { ctx_len: 4000, ..base.clone() });

        assert!(r2.ns_per_token > r1.ns_per_token);
        assert!(r3.ns_per_token > r2.ns_per_token);
    }
}
