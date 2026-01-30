//! Optimization search over simulation parameters.
//!
//! Searches over group size, precision bits, and overlap to find
//! the best configuration minimizing ns/token subject to risk constraints.

use serde::Serialize;

use crate::model::{risk_estimate, simulate_token, CommPrecision, SimConfig, SimResult};

/// Optimization search parameters.
#[derive(Debug, Clone, Serialize)]
pub struct OptParams {
    /// Maximum group size to search.
    pub max_group: u32,
    /// Maximum acceptable risk.
    pub max_risk: f64,
    /// Base overlap value from config.
    pub base_overlap: f64,
    /// Quality penalty weight (trades off latency vs quality).
    pub quality_weight: f64,
}

impl Default for OptParams {
    fn default() -> Self {
        Self {
            max_group: 8,
            max_risk: 0.3,
            base_overlap: 0.0,
            quality_weight: 0.0,
        }
    }
}

/// Result of optimization search.
#[derive(Debug, Clone, Serialize)]
pub struct OptResult {
    /// Best configuration found.
    pub best_config: SimConfig,
    /// Simulation result for best config.
    pub best_result: SimResult,
    /// Risk of best config.
    pub best_risk: f64,
    /// Number of configurations evaluated.
    pub configs_evaluated: u32,
    /// Number of configs rejected due to risk.
    pub configs_rejected_risk: u32,
}

/// Search for optimal configuration.
///
/// Searches over:
/// - group_size: 1..=max_group
/// - bits: 16, 8, 4
/// - overlap: [0.0, base_overlap, min(0.75, base_overlap * 1.7)]
///
/// Returns the configuration minimizing ns/token subject to risk <= max_risk.
///
/// Realistic constraints:
/// - FP4 requires either high overlap (>=0.5) or high risk budget (>=0.5)
/// - Group size > 4 has sharply increasing risk
/// - Quality penalty can be applied to objective
pub fn optimize(base_config: &SimConfig, params: &OptParams) -> OptResult {
    let precisions = [CommPrecision::FP16, CommPrecision::FP8, CommPrecision::FP4];

    // Overlap candidates: include 0, base, and an aggressive option
    let overlap_candidates: [f64; 4] = [
        0.0,
        params.base_overlap,
        (params.base_overlap * 1.5).min(0.6),
        (params.base_overlap * 2.0).min(0.75),
    ];

    let mut best_config = base_config.clone();
    let mut best_result = simulate_token(base_config);
    let mut best_risk = risk_estimate(base_config.group_size, base_config.precision, base_config.overlap);
    let mut best_score = f64::MAX;
    let mut configs_evaluated = 0u32;
    let mut configs_rejected_risk = 0u32;

    // Grid search
    for group_size in 1..=params.max_group {
        for &precision in &precisions {
            for &overlap in &overlap_candidates {
                // Check if FP4 is allowed for this overlap
                if precision == CommPrecision::FP4 {
                    // FP4 requires high overlap or high risk budget
                    if overlap < 0.5 && params.max_risk < 0.5 {
                        configs_rejected_risk += 1;
                        continue;
                    }
                }

                // Check risk constraint
                let risk = risk_estimate(group_size, precision, overlap);
                if risk > params.max_risk {
                    configs_rejected_risk += 1;
                    continue;
                }

                configs_evaluated += 1;

                let config = SimConfig {
                    group_size,
                    precision,
                    overlap,
                    ..base_config.clone()
                };

                let result = simulate_token(&config);

                // Score = latency + quality_weight * risk
                let score = result.ns_per_token as f64
                    + params.quality_weight * risk * result.ns_per_token as f64;

                if score < best_score {
                    best_config = config;
                    best_result = result;
                    best_risk = risk;
                    best_score = score;
                }
            }
        }
    }

    OptResult {
        best_config,
        best_result,
        best_risk,
        configs_evaluated,
        configs_rejected_risk,
    }
}

/// Sweep parameters and output results for sensitivity analysis.
/// Returns a vector of sweep results.
pub fn parameter_sweep(
    base_config: &SimConfig,
    opt_params: &OptParams,
    bw_values: &[f64],
    lat_values: &[u64],
    ctx_values: &[u32],
) -> Vec<SweepResult> {
    let mut results = Vec::with_capacity(bw_values.len() * lat_values.len() * ctx_values.len());

    for &bw in bw_values {
        for &lat in lat_values {
            for &ctx in ctx_values {
                let config = SimConfig {
                    bw_gbps: bw,
                    latency_ns: lat,
                    ctx_len: ctx,
                    ..base_config.clone()
                };

                // TP=1 baseline
                let tp1_config = SimConfig { tp: 1, ..config.clone() };
                let tp1_result = simulate_token(&tp1_config);

                // TP=2
                let tp2_config = SimConfig { tp: 2, ..config.clone() };
                let tp2_result = simulate_token(&tp2_config);

                // TP=4
                let tp4_config = SimConfig { tp: 4, ..config.clone() };
                let tp4_result = simulate_token(&tp4_config);

                // Optimized TP=4
                let opt_result = optimize(&tp4_config, opt_params);

                // Calculate overheads
                let tp2_overhead = if tp1_result.ns_per_token > 0 {
                    100.0 * (tp2_result.ns_per_token as f64 - tp1_result.ns_per_token as f64)
                        / tp1_result.ns_per_token as f64
                } else {
                    0.0
                };

                let tp4_overhead = if tp1_result.ns_per_token > 0 {
                    100.0 * (tp4_result.ns_per_token as f64 - tp1_result.ns_per_token as f64)
                        / tp1_result.ns_per_token as f64
                } else {
                    0.0
                };

                let opt_speedup = if opt_result.best_result.ns_per_token > 0 {
                    tp4_result.ns_per_token as f64 / opt_result.best_result.ns_per_token as f64
                } else {
                    1.0
                };

                results.push(SweepResult {
                    bw_gbps: bw,
                    latency_ns: lat,
                    ctx_len: ctx,
                    tp1_ns: tp1_result.ns_per_token,
                    tp2_ns: tp2_result.ns_per_token,
                    tp4_ns: tp4_result.ns_per_token,
                    tp2_overhead_pct: tp2_overhead,
                    tp4_overhead_pct: tp4_overhead,
                    opt_ns: opt_result.best_result.ns_per_token,
                    opt_speedup,
                    opt_group: opt_result.best_config.group_size,
                    opt_bits: opt_result.best_config.precision.bits(),
                    opt_overlap: opt_result.best_config.overlap,
                    opt_risk: opt_result.best_risk,
                });
            }
        }
    }

    results
}

/// Single result from parameter sweep.
#[derive(Debug, Clone, Serialize)]
pub struct SweepResult {
    pub bw_gbps: f64,
    pub latency_ns: u64,
    pub ctx_len: u32,
    pub tp1_ns: u64,
    pub tp2_ns: u64,
    pub tp4_ns: u64,
    pub tp2_overhead_pct: f64,
    pub tp4_overhead_pct: f64,
    pub opt_ns: u64,
    pub opt_speedup: f64,
    pub opt_group: u32,
    pub opt_bits: u32,
    pub opt_overlap: f64,
    pub opt_risk: f64,
}

impl SweepResult {
    /// CSV header.
    pub fn csv_header() -> &'static str {
        "bw_gbps,latency_ns,ctx_len,tp1_ns,tp2_ns,tp4_ns,tp2_overhead%,tp4_overhead%,opt_ns,opt_speedup,opt_group,opt_bits,opt_overlap,opt_risk"
    }

    /// Format as CSV row.
    pub fn to_csv(&self) -> String {
        format!(
            "{:.1},{},{},{},{},{},{:.1},{:.1},{},{:.2},{},{},{:.2},{:.3}",
            self.bw_gbps,
            self.latency_ns,
            self.ctx_len,
            self.tp1_ns,
            self.tp2_ns,
            self.tp4_ns,
            self.tp2_overhead_pct,
            self.tp4_overhead_pct,
            self.opt_ns,
            self.opt_speedup,
            self.opt_group,
            self.opt_bits,
            self.opt_overlap,
            self.opt_risk
        )
    }
}

/// Compare TP configurations and return overhead percentages.
pub fn compare_tp_configs(base: &SimConfig) -> (f64, f64) {
    let tp1 = simulate_token(&SimConfig { tp: 1, ..base.clone() });
    let tp2 = simulate_token(&SimConfig { tp: 2, ..base.clone() });
    let tp4 = simulate_token(&SimConfig { tp: 4, ..base.clone() });

    let tp2_overhead = if tp1.ns_per_token > 0 {
        100.0 * (tp2.ns_per_token as f64 - tp1.ns_per_token as f64) / tp1.ns_per_token as f64
    } else {
        0.0
    };

    let tp4_overhead = if tp1.ns_per_token > 0 {
        100.0 * (tp4.ns_per_token as f64 - tp1.ns_per_token as f64) / tp1.ns_per_token as f64
    } else {
        0.0
    };

    (tp2_overhead, tp4_overhead)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_finds_better_than_baseline() {
        let config = SimConfig {
            tp: 4,
            layers: 32,
            compute_ns: 80_000,
            ar_bytes: 32_768,
            bw_gbps: 12.0,
            latency_ns: 45_000,
            ctx_len: 0,
            kv_bytes_base: 0,
            ..Default::default()
        };

        let baseline = simulate_token(&config);

        let params = OptParams {
            max_group: 8,
            max_risk: 0.5,
            base_overlap: 0.3,
            quality_weight: 0.0,
        };

        let opt = optimize(&config, &params);

        // Optimizer should find something at least as good as baseline
        assert!(opt.best_result.ns_per_token <= baseline.ns_per_token);
    }

    #[test]
    fn test_optimize_respects_risk() {
        let config = SimConfig {
            tp: 4,
            ..Default::default()
        };

        let params = OptParams {
            max_group: 8,
            max_risk: 0.15, // Low risk tolerance
            base_overlap: 0.0,
            quality_weight: 0.0,
        };

        let opt = optimize(&config, &params);

        // Best config should respect risk constraint
        let risk = risk_estimate(opt.best_config.group_size, opt.best_config.precision, opt.best_config.overlap);
        assert!(risk <= params.max_risk + 0.01); // Small tolerance for floating point
    }

    #[test]
    fn test_fp4_requires_conditions() {
        let config = SimConfig {
            tp: 4,
            ..Default::default()
        };

        // Low risk, low overlap: FP4 should not be chosen
        let params_low = OptParams {
            max_group: 4,
            max_risk: 0.3,
            base_overlap: 0.0,
            quality_weight: 0.0,
        };
        let opt_low = optimize(&config, &params_low);
        assert_ne!(opt_low.best_config.precision, CommPrecision::FP4);

        // High risk: FP4 can be chosen
        let params_high = OptParams {
            max_group: 4,
            max_risk: 0.6,
            base_overlap: 0.5,
            quality_weight: 0.0,
        };
        let opt_high = optimize(&config, &params_high);
        // FP4 should at least be considered now
        assert!(opt_high.configs_evaluated > 0);
    }

    #[test]
    fn test_sweep_produces_results() {
        let config = SimConfig {
            tp: 4,
            ctx_len: 0,
            kv_bytes_base: 0,
            ..Default::default()
        };

        let params = OptParams::default();

        let bw_values = vec![12.0, 20.0];
        let lat_values = vec![25000, 45000];
        let ctx_values = vec![512];

        let results = parameter_sweep(&config, &params, &bw_values, &lat_values, &ctx_values);

        assert_eq!(results.len(), 4);

        // Higher bandwidth should reduce overhead
        let low_bw = results.iter().find(|r| r.bw_gbps == 12.0 && r.latency_ns == 25000).unwrap();
        let high_bw = results.iter().find(|r| r.bw_gbps == 20.0 && r.latency_ns == 25000).unwrap();
        assert!(high_bw.tp4_overhead_pct <= low_bw.tp4_overhead_pct);
    }

    #[test]
    fn test_tp2_less_overhead_than_tp4() {
        // Under most conditions, TP=2 should have less overhead than TP=4
        let config = SimConfig {
            tp: 4,
            ctx_len: 0,
            kv_bytes_base: 0,
            ..Default::default()
        };

        let (tp2_overhead, tp4_overhead) = compare_tp_configs(&config);
        assert!(tp2_overhead < tp4_overhead);
    }
}
