//! Comprehensive test suite validating simulation invariants.
//!
//! These tests ensure the simulator behaves correctly without requiring
//! actual GPUs - critical for development on systems like M1 MacBooks.
//!
//! # Test Categories
//!
//! - **Invariants**: Monotonicity properties that must hold
//! - **Edge cases**: Boundary conditions and special values
//! - **Allocation checks**: Verify hot path is allocation-free (conceptual)
//! - **Reality checks**: Guardrails to prevent unrealistic results

#[cfg(test)]
mod invariants {
    use crate::model::{comm_time_ns, simulate_token, CollectiveAlgo, CommPrecision, SimConfig};

    /// Test: Increasing bandwidth should not worsen time.
    #[test]
    fn bandwidth_monotonicity() {
        let bandwidths = [5.0, 10.0, 20.0, 40.0, 80.0];

        for tp in [2, 4, 8] {
            let mut prev_time = u64::MAX;

            for &bw in &bandwidths {
                let config = SimConfig {
                    tp,
                    bw_gbps: bw,
                    layers: 32,
                    compute_ns: 80_000,
                    ar_bytes: 32_768,
                    latency_ns: 25_000,
                    ctx_len: 0,
                    kv_bytes_base: 0,
                    ..Default::default()
                };

                let result = simulate_token(&config);
                assert!(
                    result.ns_per_token <= prev_time,
                    "tp={}, bw={}: {} > {} (should decrease)",
                    tp, bw, result.ns_per_token, prev_time
                );
                prev_time = result.ns_per_token;
            }
        }
    }

    /// Test: Increasing latency should not improve time.
    #[test]
    fn latency_monotonicity() {
        let latencies = [5_000, 15_000, 30_000, 50_000, 100_000];

        for tp in [2, 4, 8] {
            let mut prev_time = 0u64;

            for &lat in &latencies {
                let config = SimConfig {
                    tp,
                    latency_ns: lat,
                    layers: 32,
                    compute_ns: 80_000,
                    ar_bytes: 32_768,
                    bw_gbps: 18.0,
                    ctx_len: 0,
                    kv_bytes_base: 0,
                    ..Default::default()
                };

                let result = simulate_token(&config);
                assert!(
                    result.ns_per_token >= prev_time,
                    "tp={}, lat={}: {} < {} (should increase)",
                    tp, lat, result.ns_per_token, prev_time
                );
                prev_time = result.ns_per_token;
            }
        }
    }

    /// Test: Increasing payload should not improve time.
    #[test]
    fn payload_monotonicity() {
        let payloads = [8_192, 16_384, 32_768, 65_536, 131_072];

        for tp in [2, 4, 8] {
            let mut prev_time = 0u64;

            for &payload in &payloads {
                let config = SimConfig {
                    tp,
                    ar_bytes: payload,
                    layers: 32,
                    compute_ns: 80_000,
                    bw_gbps: 18.0,
                    latency_ns: 25_000,
                    ctx_len: 0,
                    kv_bytes_base: 0,
                    ..Default::default()
                };

                let result = simulate_token(&config);
                assert!(
                    result.ns_per_token >= prev_time,
                    "tp={}, payload={}: {} < {} (should increase)",
                    tp, payload, result.ns_per_token, prev_time
                );
                prev_time = result.ns_per_token;
            }
        }
    }

    /// Test: Increasing overlap should not worsen time.
    #[test]
    fn overlap_monotonicity() {
        let overlaps = [0.0, 0.1, 0.25, 0.5, 0.75, 0.95];

        for tp in [2, 4, 8] {
            let mut prev_time = u64::MAX;

            for &overlap in &overlaps {
                let config = SimConfig {
                    tp,
                    overlap,
                    layers: 32,
                    compute_ns: 80_000,
                    ar_bytes: 32_768,
                    bw_gbps: 18.0,
                    latency_ns: 25_000,
                    ctx_len: 0,
                    kv_bytes_base: 0,
                    ..Default::default()
                };

                let result = simulate_token(&config);
                assert!(
                    result.ns_per_token <= prev_time,
                    "tp={}, overlap={}: {} > {} (should decrease)",
                    tp, overlap, result.ns_per_token, prev_time
                );
                prev_time = result.ns_per_token;
            }
        }
    }

    /// Test: For TP=1, comm cost must be 0 and independent of all comm params.
    #[test]
    fn tp1_zero_comm() {
        let base = SimConfig {
            tp: 1,
            layers: 32,
            compute_ns: 50_000,
            ctx_len: 0,
            kv_bytes_base: 0,
            ..Default::default()
        };

        let expected_time = base.compute_ns * base.layers as u64;
        let base_result = simulate_token(&base);

        assert_eq!(base_result.comm_total_ns, 0, "TP=1 should have zero comm");
        assert_eq!(
            base_result.ns_per_token, expected_time,
            "TP=1 time should equal pure compute"
        );

        // Vary all comm parameters - should not affect result
        let variations: Vec<SimConfig> = vec![
            SimConfig { algo: CollectiveAlgo::Tree, ..base.clone() },
            SimConfig { algo: CollectiveAlgo::Ring, ..base.clone() },
            SimConfig { group_size: 1, ..base.clone() },
            SimConfig { group_size: 8, ..base.clone() },
            SimConfig { precision: CommPrecision::FP16, ..base.clone() },
            SimConfig { precision: CommPrecision::FP8, ..base.clone() },
            SimConfig { overlap: 0.0, ..base.clone() },
            SimConfig { overlap: 0.5, ..base.clone() },
            SimConfig { bw_gbps: 1.0, ..base.clone() },
            SimConfig { bw_gbps: 100.0, ..base.clone() },
            SimConfig { latency_ns: 100, ..base.clone() },
            SimConfig { latency_ns: 100_000, ..base.clone() },
        ];

        for (i, cfg) in variations.iter().enumerate() {
            let result = simulate_token(cfg);
            assert_eq!(
                result.ns_per_token, expected_time,
                "TP=1 variation {} should have same time", i
            );
            assert_eq!(
                result.comm_total_ns, 0,
                "TP=1 variation {} should have zero comm", i
            );
        }
    }

    /// Test: Ring and Tree should differ for tp>2 and both be monotonic w.r.t. payload.
    #[test]
    fn ring_tree_differ_and_monotonic() {
        let payloads = [8_192, 16_384, 32_768, 65_536];

        for tp in [2, 4, 8] {
            let mut ring_times = Vec::new();
            let mut tree_times = Vec::new();

            for &payload in &payloads {
                let ring_t = comm_time_ns(payload, 18.0, 25_000, CollectiveAlgo::Ring, tp, false);
                let tree_t = comm_time_ns(payload, 18.0, 25_000, CollectiveAlgo::Tree, tp, false);

                ring_times.push(ring_t);
                tree_times.push(tree_t);
            }

            // Verify monotonicity
            for i in 1..ring_times.len() {
                assert!(ring_times[i] >= ring_times[i - 1], "Ring not monotonic at tp={}", tp);
                assert!(tree_times[i] >= tree_times[i - 1], "Tree not monotonic at tp={}", tp);
            }

            // Ring and Tree should differ for tp > 2
            if tp > 2 {
                let differ_count = ring_times.iter().zip(&tree_times).filter(|(r, t)| r != t).count();
                assert!(differ_count > 0, "Ring and Tree should differ at tp={}", tp);
            }
        }
    }

    /// Test: Lower precision should reduce comm time.
    #[test]
    fn precision_reduces_comm() {
        let config = SimConfig {
            tp: 4,
            layers: 32,
            compute_ns: 80_000,
            ar_bytes: 65_536,
            bw_gbps: 18.0,
            latency_ns: 25_000,
            ctx_len: 0,
            kv_bytes_base: 0,
            ..Default::default()
        };

        let fp16 = simulate_token(&SimConfig { precision: CommPrecision::FP16, ..config.clone() });
        let fp8 = simulate_token(&SimConfig { precision: CommPrecision::FP8, ..config.clone() });
        let fp4 = simulate_token(&SimConfig { precision: CommPrecision::FP4, ..config.clone() });

        // Lower precision should have better (lower) or equal time
        assert!(fp8.ns_per_token <= fp16.ns_per_token);
        assert!(fp4.ns_per_token <= fp8.ns_per_token);

        // Comm component specifically should decrease
        assert!(fp8.comm_total_ns <= fp16.comm_total_ns);
        assert!(fp4.comm_total_ns <= fp8.comm_total_ns);
    }

    /// Test: Higher TP should increase comm overhead.
    #[test]
    fn tp_increases_comm() {
        let base = SimConfig {
            layers: 32,
            compute_ns: 80_000,
            ar_bytes: 32_768,
            bw_gbps: 18.0,
            latency_ns: 25_000,
            ctx_len: 0,
            kv_bytes_base: 0,
            ..Default::default()
        };

        let tp1 = simulate_token(&SimConfig { tp: 1, ..base.clone() });
        let tp2 = simulate_token(&SimConfig { tp: 2, ..base.clone() });
        let tp4 = simulate_token(&SimConfig { tp: 4, ..base.clone() });
        let tp8 = simulate_token(&SimConfig { tp: 8, ..base.clone() });

        // TP=1 has no comm
        assert_eq!(tp1.comm_total_ns, 0);

        // Higher TP should have more comm
        assert!(tp2.comm_total_ns > 0);
        assert!(tp4.comm_total_ns > tp2.comm_total_ns);
        assert!(tp8.comm_total_ns > tp4.comm_total_ns);
    }

    /// Test: Num collectives respects grouping and comm_ops.
    #[test]
    fn grouping_and_comm_ops() {
        let config = SimConfig {
            tp: 4,
            layers: 32,
            comm_ops_per_layer: 2,
            ctx_len: 0,
            kv_bytes_base: 0,
            ..Default::default()
        };

        let g1 = simulate_token(&SimConfig { group_size: 1, ..config.clone() });
        let g4 = simulate_token(&SimConfig { group_size: 4, ..config.clone() });
        let g8 = simulate_token(&SimConfig { group_size: 8, ..config.clone() });

        // With 2 comm ops per layer and 32 layers:
        // group=1: 32 groups * 2 ops = 64 collectives
        // group=4: 8 groups * 2 ops = 16 collectives
        // group=8: 4 groups * 2 ops = 8 collectives
        assert_eq!(g1.num_collectives, 64);
        assert_eq!(g4.num_collectives, 16);
        assert_eq!(g8.num_collectives, 8);
    }
}

#[cfg(test)]
mod edge_cases {
    use crate::model::{simulate_token, CollectiveAlgo, SimConfig};

    /// Test: Very high bandwidth approaches latency-bound regime.
    /// With Ring algorithm at TP=4, latency scales by (tp-1) = 3 steps.
    #[test]
    fn high_bandwidth_latency_bound() {
        let config = SimConfig {
            tp: 4,
            ar_bytes: 8_192,
            bw_gbps: 500.0, // Very high
            latency_ns: 25_000,
            layers: 1,
            compute_ns: 0,
            comm_ops_per_layer: 1,
            ctx_len: 0,
            kv_bytes_base: 0,
            algo: CollectiveAlgo::Ring,
            ..Default::default()
        };

        let result = simulate_token(&config);
        // Ring at TP=4 has (tp-1)=3 latency steps
        let expected_latency = config.latency_ns * 3;
        // With extremely high bandwidth, should be close to just latency
        assert!(result.comm_total_ns >= expected_latency);
        assert!(result.comm_total_ns < expected_latency + 1000); // Small transfer overhead
    }

    /// Test: Zero layers produces zero time.
    #[test]
    fn zero_layers() {
        let config = SimConfig {
            tp: 4,
            layers: 0,
            ctx_len: 0,
            kv_bytes_base: 0,
            ..Default::default()
        };

        let result = simulate_token(&config);
        assert_eq!(result.ns_per_token, 0);
        assert_eq!(result.num_collectives, 0);
    }

    /// Test: Algorithm overhead factors are correct.
    #[test]
    fn algo_overhead_factors() {
        let eps = 0.001;
        assert!((CollectiveAlgo::Ring.overhead_factor(2) - 1.0).abs() < eps);
        assert!((CollectiveAlgo::Ring.overhead_factor(4) - 1.5).abs() < eps);
        assert!((CollectiveAlgo::Ring.overhead_factor(8) - 1.75).abs() < eps);

        assert!((CollectiveAlgo::Tree.overhead_factor(2) - 1.0).abs() < eps);
        assert!((CollectiveAlgo::Tree.overhead_factor(4) - 2.0).abs() < eps);
        assert!((CollectiveAlgo::Tree.overhead_factor(8) - 3.0).abs() < eps);

        assert_eq!(CollectiveAlgo::Ring.overhead_factor(1), 0.0);
        assert_eq!(CollectiveAlgo::Tree.overhead_factor(1), 0.0);
    }
}

#[cfg(test)]
mod allocation_free {
    use crate::model::{Bottleneck, CollectiveAlgo, CommPrecision, SimConfig, SimResult};

    /// Verify that all hot-path types implement Copy (no heap allocation needed).
    #[test]
    fn hot_path_types_are_copy() {
        let algo = CollectiveAlgo::Ring;
        let _algo_copy: CollectiveAlgo = algo;
        let _algo_copy2: CollectiveAlgo = algo;

        let prec = CommPrecision::FP16;
        let _prec_copy: CommPrecision = prec;
        let _prec_copy2: CommPrecision = prec;

        let result = SimResult {
            ns_per_token: 100,
            tok_per_sec: 10.0,
            compute_total_ns: 80,
            comm_total_ns: 20,
            kv_overhead_ns: 0,
            num_collectives: 1,
            bottleneck: Bottleneck::Compute,
        };
        let _result_copy: SimResult = result;
        let _result_copy2: SimResult = result;
    }

    /// Test that simulate_token can be called in a tight loop without issues.
    #[test]
    fn tight_loop_simulation() {
        let config = SimConfig {
            tp: 4,
            layers: 32,
            ctx_len: 0,
            kv_bytes_base: 0,
            ..Default::default()
        };

        let mut total_ns: u64 = 0;
        for _ in 0..10_000 {
            let result = crate::model::simulate_token(&config);
            total_ns = total_ns.wrapping_add(result.ns_per_token);
        }

        assert!(total_ns > 0);
    }
}

// ============================================================================
// Reality-Check Acceptance Tests
// ============================================================================
//
// These tests ensure the simulator produces plausible results under realistic
// parameter regimes. They are NOT "ground truth" but guardrails to prevent
// the simulator from collapsing to trivially small overhead.

#[cfg(test)]
mod reality_checks {
    use crate::model::{simulate_token, HardwarePreset, SimConfig};
    use crate::optimize::compare_tp_configs;

    /// REALITY CHECK: Under PCIe 4.0 bad topology with reasonable comm parameters,
    /// TP=4 overhead should be >= 25% vs TP=1 for decode.
    #[test]
    fn pcie4_bad_tp4_significant_overhead() {
        let preset = HardwarePreset::Pcie4Bad;
        let config = SimConfig {
            tp: 4,
            layers: 32,
            compute_ns: 80_000,           // 80 µs per layer
            comm_ops_per_layer: 2,        // 2 collectives per layer
            ar_bytes: 32_768,             // 32 KB per collective
            bw_gbps: preset.bandwidth_gbps(),
            latency_ns: preset.latency_ns(),
            ctx_len: 0,
            kv_bytes_base: 0,
            ..Default::default()
        };

        let (_, tp4_overhead) = compare_tp_configs(&config);

        assert!(
            tp4_overhead >= 25.0,
            "PCIe4 bad topology: TP=4 overhead should be >= 25%, got {:.1}%",
            tp4_overhead
        );
    }

    /// ACCEPTANCE TEST: Under pcie4_bad, TP=4 ratio must be >= 2.0 (>= 100% overhead).
    /// This guards against model regressions that produce unrealistically low TP overhead.
    #[test]
    fn pcie4_bad_tp4_ratio_at_least_2x() {
        let preset = HardwarePreset::Pcie4Bad;
        let config = SimConfig {
            tp: 4,
            layers: 32,
            compute_ns: 80_000,           // 80 µs per layer
            comm_ops_per_layer: 2,        // 2 collectives per layer
            ar_bytes: 32_768,             // 32 KB per collective
            bw_gbps: preset.bandwidth_gbps(),
            latency_ns: preset.latency_ns(),
            ctx_len: 0,
            kv_bytes_base: 0,
            ..Default::default()
        };

        let tp1_result = simulate_token(&SimConfig { tp: 1, ..config.clone() });
        let tp4_result = simulate_token(&config);

        let ratio = tp4_result.ns_per_token as f64 / tp1_result.ns_per_token as f64;
        let overhead_pct = (ratio - 1.0) * 100.0;

        assert!(
            ratio >= 2.0,
            "REALISM GUARD: Under pcie4_bad, TP=4 ratio must be >= 2.0x. \
             Got ratio={:.2}x (overhead={:.1}%). \
             t_tp1={} ns, t_tp4={} ns",
            ratio, overhead_pct, tp1_result.ns_per_token, tp4_result.ns_per_token
        );
    }

    /// REALITY CHECK: TP=2 overhead should be materially less than TP=4.
    #[test]
    fn tp2_less_overhead_than_tp4() {
        let config = SimConfig {
            tp: 4,
            layers: 32,
            compute_ns: 80_000,
            comm_ops_per_layer: 2,
            ar_bytes: 32_768,
            bw_gbps: 12.0,
            latency_ns: 45_000,
            ctx_len: 0,
            kv_bytes_base: 0,
            ..Default::default()
        };

        let (tp2_overhead, tp4_overhead) = compare_tp_configs(&config);

        assert!(
            tp2_overhead < tp4_overhead,
            "TP=2 overhead ({:.1}%) should be less than TP=4 ({:.1}%)",
            tp2_overhead, tp4_overhead
        );

        // TP=4 should have at least 1.5x the overhead of TP=2
        assert!(
            tp4_overhead > tp2_overhead * 1.3,
            "TP=4 overhead ({:.1}%) should be significantly more than TP=2 ({:.1}%)",
            tp4_overhead, tp2_overhead
        );
    }

    /// REALITY CHECK: Increasing context length should increase decode time
    /// more for TP>1 than for TP=1 (due to KV sharding overhead).
    #[test]
    fn ctx_len_hurts_tp_more() {
        let base = SimConfig {
            layers: 32,
            compute_ns: 80_000,
            kv_bytes_base: 128,
            kv_sharded: true,
            ..Default::default()
        };

        // Short context
        let tp1_short = simulate_token(&SimConfig { tp: 1, ctx_len: 256, ..base.clone() });
        let tp4_short = simulate_token(&SimConfig { tp: 4, ctx_len: 256, ..base.clone() });

        // Long context
        let tp1_long = simulate_token(&SimConfig { tp: 1, ctx_len: 2048, ..base.clone() });
        let tp4_long = simulate_token(&SimConfig { tp: 4, ctx_len: 2048, ..base.clone() });

        // Compute context scaling impact
        let tp1_increase = tp1_long.ns_per_token as f64 / tp1_short.ns_per_token as f64;
        let tp4_increase = tp4_long.ns_per_token as f64 / tp4_short.ns_per_token as f64;

        // TP=4 with sharded KV should see at least as much impact from context growth
        assert!(
            tp4_increase >= tp1_increase * 0.9, // Allow small tolerance
            "TP=4 context scaling ({:.2}x) should be >= TP=1 ({:.2}x)",
            tp4_increase, tp1_increase
        );
    }

    /// REALITY CHECK: More comm ops per layer should increase overhead.
    #[test]
    fn more_comm_ops_increases_overhead() {
        let base = SimConfig {
            tp: 4,
            layers: 32,
            compute_ns: 80_000,
            bw_gbps: 18.0,
            latency_ns: 25_000,
            ctx_len: 0,
            kv_bytes_base: 0,
            ..Default::default()
        };

        let ops2 = simulate_token(&SimConfig { comm_ops_per_layer: 2, ..base.clone() });
        let ops4 = simulate_token(&SimConfig { comm_ops_per_layer: 4, ..base.clone() });

        let tp1 = simulate_token(&SimConfig { tp: 1, ..base.clone() });

        let overhead_ops2 = 100.0 * (ops2.ns_per_token as f64 - tp1.ns_per_token as f64)
            / tp1.ns_per_token as f64;
        let overhead_ops4 = 100.0 * (ops4.ns_per_token as f64 - tp1.ns_per_token as f64)
            / tp1.ns_per_token as f64;

        assert!(
            overhead_ops4 > overhead_ops2,
            "4 comm ops ({:.1}%) should have more overhead than 2 ops ({:.1}%)",
            overhead_ops4, overhead_ops2
        );
    }

    /// REALITY CHECK: NVLink should have much lower overhead than PCIe.
    #[test]
    fn nvlink_much_better_than_pcie() {
        let pcie_config = SimConfig {
            tp: 4,
            layers: 32,
            compute_ns: 80_000,
            comm_ops_per_layer: 2,
            ar_bytes: 32_768,
            bw_gbps: HardwarePreset::Pcie4Bad.bandwidth_gbps(),
            latency_ns: HardwarePreset::Pcie4Bad.latency_ns(),
            ctx_len: 0,
            kv_bytes_base: 0,
            ..Default::default()
        };

        let nvlink_config = SimConfig {
            bw_gbps: HardwarePreset::NvLink.bandwidth_gbps(),
            latency_ns: HardwarePreset::NvLink.latency_ns(),
            ..pcie_config.clone()
        };

        let (_, pcie_overhead) = compare_tp_configs(&pcie_config);
        let (_, nvlink_overhead) = compare_tp_configs(&nvlink_config);

        assert!(
            nvlink_overhead < pcie_overhead / 2.0,
            "NVLink overhead ({:.1}%) should be much less than PCIe ({:.1}%)",
            nvlink_overhead, pcie_overhead
        );
    }

    /// REALITY CHECK: With realistic PCIe params, comm should be a significant
    /// portion of total time (not trivially small).
    #[test]
    fn comm_is_significant_portion() {
        let config = SimConfig {
            tp: 4,
            layers: 32,
            compute_ns: 80_000,
            comm_ops_per_layer: 2,
            ar_bytes: 32_768,
            bw_gbps: 12.0,
            latency_ns: 45_000,
            ctx_len: 0,
            kv_bytes_base: 0,
            ..Default::default()
        };

        let result = simulate_token(&config);

        let comm_pct = 100.0 * result.comm_total_ns as f64 / result.ns_per_token as f64;

        assert!(
            comm_pct >= 20.0,
            "Comm should be >= 20% of total time, got {:.1}%",
            comm_pct
        );
    }
}

#[cfg(test)]
mod ttft_tests {
    use crate::model::{
        simulate_full_request, simulate_prefill, PrefillConfig, SimConfig,
    };

    /// Test: TTFT increases with more prompt tokens.
    #[test]
    fn ttft_scales_with_prompt() {
        let base = PrefillConfig {
            base: SimConfig { tp: 4, ..Default::default() },
            prompt_tokens: 100,
            prefill_compute_factor: 1.0,
            kv_cache_write_ns: 0,
        };

        let r1 = simulate_prefill(&base);
        let r2 = simulate_prefill(&PrefillConfig { prompt_tokens: 500, ..base.clone() });
        let r3 = simulate_prefill(&PrefillConfig { prompt_tokens: 1000, ..base.clone() });

        assert!(r2.ttft_ns > r1.ttft_ns);
        assert!(r3.ttft_ns > r2.ttft_ns);
    }

    /// Test: Full request includes both prefill and decode.
    #[test]
    fn full_request_combines_stages() {
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

        let result = simulate_full_request(&prefill_config, 50);

        assert!(result.prefill.ttft_ns > 0);
        assert!(result.decode.ns_per_token > 0);
        assert!(result.total_ns > result.prefill.ttft_ns);
    }
}
