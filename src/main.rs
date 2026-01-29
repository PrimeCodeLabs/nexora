//! Multi-GPU LLM Inference Simulator
//!
//! Simulates latency/throughput for tensor-parallel inference,
//! exploring why TP=4 (4x24GB) can feel worse than a single 96GB GPU,
//! and how optimizations like overlap, reduced precision, and layer grouping help.

mod cli;
mod model;
mod optimize;

#[cfg(test)]
mod tests;

use clap::Parser;
use cli::{format_ns, format_pct, format_tps, Cli};
use model::{
    risk_estimate, simulate_full_request, simulate_prefill, simulate_token,
    simulate_token_with_cache, DecodeWithCacheConfig, HardwarePreset, SimConfig,
};
use optimize::{optimize, parameter_sweep, OptParams, SweepResult};

fn main() {
    let args = Cli::parse();

    if args.bench {
        run_benchmark(&args);
        return;
    }

    if args.sweep {
        run_sweep(&args);
        return;
    }

    if args.ttft {
        run_ttft_simulation(&args);
        return;
    }

    run_simulation(&args);
}

fn run_simulation(args: &Cli) {
    let config = args.to_config();

    println!("=== Multi-GPU LLM Inference Simulator ===\n");

    // Show preset if used
    if let Some(preset_str) = &args.preset {
        if let Some(preset) = HardwarePreset::from_str(preset_str) {
            println!("Hardware Preset: {}", preset.name());
        }
    }

    println!("Configuration:");
    println!("  Layers:        {}", config.layers);
    println!("  TP degree:     {}", config.tp);
    println!("  Compute/layer: {}", format_ns(config.compute_ns));
    println!("  Comm ops/layer: {}", config.comm_ops_per_layer);
    println!("  AR bytes:      {} B", config.ar_bytes);
    if config.ag_bytes > 0 {
        println!("  AG bytes:      {} B", config.ag_bytes);
    }
    println!("  Bandwidth:     {:.1} GB/s", config.bw_gbps);
    println!("  Latency:       {}", format_ns(config.latency_ns));
    println!("  Algorithm:     {:?}", config.algo);
    println!("  Group size:    {}", config.group_size);
    println!("  Precision:     {} bits", config.precision.bits());
    println!("  Overlap:       {:.0}%", config.overlap * 100.0);
    if config.ctx_len > 0 {
        println!("  Context len:   {}", config.ctx_len);
    }
    println!();

    // Baseline simulation
    let baseline = simulate_token(&config);
    let baseline_risk = risk_estimate(config.group_size, config.precision, config.overlap);

    println!("--- Baseline Results (TP={}) ---", config.tp);
    println!("  Time/token:    {}", format_ns(baseline.ns_per_token));
    println!("  Throughput:    {}", format_tps(baseline.tok_per_sec));
    println!("  Compute:       {} ({})",
        format_ns(baseline.compute_total_ns),
        format_pct(baseline.compute_total_ns as f64, baseline.ns_per_token as f64)
    );
    println!("  Comm:          {} ({})",
        format_ns(baseline.comm_total_ns),
        format_pct(baseline.comm_total_ns as f64, baseline.ns_per_token as f64)
    );
    if baseline.kv_overhead_ns > 0 {
        println!("  KV overhead:   {} ({})",
            format_ns(baseline.kv_overhead_ns),
            format_pct(baseline.kv_overhead_ns as f64, baseline.ns_per_token as f64)
        );
    }
    println!("  Collectives:   {}", baseline.num_collectives);
    println!("  Risk:          {:.2}", baseline_risk);
    println!();

    // Compare with TP=1 and TP=2
    let tp1_result = simulate_token(&SimConfig { tp: 1, ..config.clone() });

    if config.tp > 1 {
        let tp2_result = simulate_token(&SimConfig { tp: 2, ..config.clone() });

        // Calculate ratios and overhead percentages
        let tp2_ratio = tp2_result.ns_per_token as f64 / tp1_result.ns_per_token as f64;
        let tp2_overhead_pct = (tp2_ratio - 1.0) * 100.0;

        let tp4_ratio = baseline.ns_per_token as f64 / tp1_result.ns_per_token as f64;
        let tp4_overhead_pct = (tp4_ratio - 1.0) * 100.0;

        println!("--- TP Comparison (vs TP=1 baseline) ---");
        println!("  TP=1:  {}  (baseline)", format_ns(tp1_result.ns_per_token));
        println!("  TP=2:  {}  ratio={:.2}x  overhead={:+.1}%",
            format_ns(tp2_result.ns_per_token), tp2_ratio, tp2_overhead_pct);
        if config.tp >= 4 {
            println!("  TP=4:  {}  ratio={:.2}x  overhead={:+.1}%",
                format_ns(baseline.ns_per_token), tp4_ratio, tp4_overhead_pct);
        }
        println!();

        // Highlight the pain point
        if tp4_overhead_pct > 100.0 {
            println!("  *** Comm-bound regime: TP=4 is {:.1}x slower than TP=1 ***", tp4_ratio);
            println!();
        } else if tp4_overhead_pct > 25.0 {
            println!("  *** Significant TP overhead: {:.1}% ***", tp4_overhead_pct);
            println!();
        }
    }

    // Run optimizer if not disabled
    if !args.no_opt && config.tp > 1 {
        let opt_params = OptParams {
            max_group: args.max_group,
            max_risk: args.max_risk,
            base_overlap: args.overlap,
            quality_weight: 0.0,
        };

        let opt_result = optimize(&config, &opt_params);

        println!("--- Optimization Search ---");
        println!("  Configs evaluated: {}", opt_result.configs_evaluated);
        println!("  Rejected (risk):   {}", opt_result.configs_rejected_risk);
        println!();

        println!("--- Best Configuration Found ---");
        println!("  Group size:    {}", opt_result.best_config.group_size);
        println!("  Precision:     {} bits", opt_result.best_config.precision.bits());
        println!("  Overlap:       {:.0}%", opt_result.best_config.overlap * 100.0);
        println!("  Risk:          {:.2}", opt_result.best_risk);
        println!();

        println!("--- Optimized Results ---");
        println!("  Time/token:    {}", format_ns(opt_result.best_result.ns_per_token));
        println!("  Throughput:    {}", format_tps(opt_result.best_result.tok_per_sec));
        println!("  Compute:       {} ({})",
            format_ns(opt_result.best_result.compute_total_ns),
            format_pct(opt_result.best_result.compute_total_ns as f64, opt_result.best_result.ns_per_token as f64)
        );
        println!("  Comm:          {} ({})",
            format_ns(opt_result.best_result.comm_total_ns),
            format_pct(opt_result.best_result.comm_total_ns as f64, opt_result.best_result.ns_per_token as f64)
        );
        println!("  Collectives:   {}", opt_result.best_result.num_collectives);
        println!();

        // Speedup calculations
        let opt_speedup_ratio = baseline.ns_per_token as f64 / opt_result.best_result.ns_per_token as f64;
        let opt_speedup_pct = (opt_speedup_ratio - 1.0) * 100.0;

        // TP overhead calculations
        let tp_ratio = baseline.ns_per_token as f64 / tp1_result.ns_per_token as f64;
        let tp_overhead_pct = (tp_ratio - 1.0) * 100.0;

        // Print compact run summary
        print_run_summary(
            args.preset.as_deref(),
            config.bw_gbps,
            config.latency_ns,
            config.tp,
            tp1_result.ns_per_token,
            baseline.ns_per_token,
            opt_result.best_result.ns_per_token,
            tp_ratio,
            tp_overhead_pct,
            opt_result.best_config.group_size,
            opt_result.best_config.precision.bits(),
            opt_result.best_config.overlap,
            opt_result.best_risk,
            opt_speedup_ratio,
            opt_speedup_pct,
            &format!("{:?}", config.algo).to_lowercase(),
            config.comm_ops_per_layer,
            config.ar_bytes,
            config.ag_bytes,
        );

        // Key takeaway
        println!();
        if opt_speedup_ratio > 1.2 {
            println!("KEY INSIGHT: Significant gains possible with optimizations.");
            println!("  Recommendation: group={}, {}bit, {:.0}% overlap",
                opt_result.best_config.group_size,
                opt_result.best_config.precision.bits(),
                opt_result.best_config.overlap * 100.0);
        } else if opt_speedup_ratio > 1.05 {
            println!("KEY INSIGHT: Modest gains available. Consider NVLink for better scaling.");
        } else {
            println!("KEY INSIGHT: Compute-dominated regime. TP overhead is manageable.");
        }
    } else if config.tp > 1 {
        // No optimizer, but still print summary for TP comparison
        let tp_ratio = baseline.ns_per_token as f64 / tp1_result.ns_per_token as f64;
        let tp_overhead_pct = (tp_ratio - 1.0) * 100.0;

        println!("=== RUN SUMMARY ===");
        println!("  preset:       {}", args.preset.as_deref().unwrap_or("custom"));
        println!("  bw/lat:       {:.1} GB/s / {} ns", config.bw_gbps, config.latency_ns);
        println!("  algo/ops:     {:?} / {} ops/layer", config.algo, config.comm_ops_per_layer);
        println!("  ar/ag_bytes:  {} / {} B", config.ar_bytes, config.ag_bytes);
        println!("  tp:           {}", config.tp);
        println!("  t_tp1:        {} ns", tp1_result.ns_per_token);
        println!("  t_tp{}:        {} ns", config.tp, baseline.ns_per_token);
        println!("  ratio:        {:.2}x", tp_ratio);
        println!("  overhead_pct: {:.1}%", tp_overhead_pct);
    }
}

/// Print a compact run summary block.
#[allow(clippy::too_many_arguments)]
fn print_run_summary(
    preset: Option<&str>,
    bw_gbps: f64,
    latency_ns: u64,
    tp: u32,
    t_tp1_ns: u64,
    t_tpx_baseline_ns: u64,
    t_tpx_optimized_ns: u64,
    tp_ratio: f64,
    tp_overhead_pct: f64,
    opt_group: u32,
    opt_bits: u32,
    opt_overlap: f64,
    opt_risk: f64,
    opt_speedup_ratio: f64,
    opt_speedup_pct: f64,
    algo: &str,
    comm_ops: u32,
    ar_bytes: u64,
    ag_bytes: u64,
) {
    println!("=== RUN SUMMARY ===");
    println!("  preset:       {}", preset.unwrap_or("custom"));
    println!("  bw/lat:       {:.1} GB/s / {} ns", bw_gbps, latency_ns);
    println!("  algo/ops:     {} / {} ops/layer", algo, comm_ops);
    println!("  ar/ag_bytes:  {} / {} B", ar_bytes, ag_bytes);
    println!("  tp:           {}", tp);
    println!("  t_tp1:        {} ns", t_tp1_ns);
    println!("  t_tp{}_base:   {} ns", tp, t_tpx_baseline_ns);
    println!("  t_tp{}_opt:    {} ns", tp, t_tpx_optimized_ns);
    println!("  ratio:        {:.2}x", tp_ratio);
    println!("  overhead_pct: {:.1}%", tp_overhead_pct);
    println!("  optimizer:    group={}, bits={}, overlap={:.0}%, risk={:.2}",
        opt_group, opt_bits, opt_overlap * 100.0, opt_risk);
    println!("  speedup:      {:.2}x ({:+.1}%)", opt_speedup_ratio, opt_speedup_pct);
}

fn run_sweep(args: &Cli) {
    let config = args.to_config();
    let bw_values = args.parse_sweep_bw();
    let lat_values = args.parse_sweep_lat();
    let ctx_values = args.parse_sweep_ctx();

    let opt_params = OptParams {
        max_group: args.max_group,
        max_risk: args.max_risk,
        base_overlap: args.overlap,
        quality_weight: 0.0,
    };

    let results = parameter_sweep(&config, &opt_params, &bw_values, &lat_values, &ctx_values);

    // Output as CSV
    println!("{}", SweepResult::csv_header());
    for result in &results {
        println!("{}", result.to_csv());
    }
}

fn run_ttft_simulation(args: &Cli) {
    let prefill_config = args.to_prefill_config();
    let base_config = args.to_config();

    println!("=== TTFT (Time To First Token) Simulation ===\n");

    if let Some(preset_str) = &args.preset {
        if let Some(preset) = HardwarePreset::from_str(preset_str) {
            println!("Hardware Preset: {}", preset.name());
        }
    }

    println!("Configuration:");
    println!("  Layers:        {}", base_config.layers);
    println!("  TP degree:     {}", base_config.tp);
    println!("  Compute/layer: {}", format_ns(base_config.compute_ns));
    println!("  Comm ops/layer: {}", base_config.comm_ops_per_layer);
    println!("  Bandwidth:     {:.1} GB/s", base_config.bw_gbps);
    println!("  Latency:       {}", format_ns(base_config.latency_ns));
    println!();
    println!("Prefill Settings:");
    println!("  Prompt tokens:   {}", args.prompt_len.unwrap_or(args.prompt_tokens));
    println!("  Output tokens:   {}", args.output_tokens);
    println!("  Prefill factor:  {:.1}x", args.prefill_factor);
    println!();

    // Simulate prefill (TTFT)
    let prefill_result = simulate_prefill(&prefill_config);

    println!("--- Prefill Stage (TTFT) ---");
    println!("  TTFT:          {}", format_ns(prefill_result.ttft_ns));
    println!("  Compute:       {} ({})",
        format_ns(prefill_result.compute_ns),
        format_pct(prefill_result.compute_ns as f64, prefill_result.ttft_ns as f64)
    );
    println!("  Comm:          {} ({})",
        format_ns(prefill_result.comm_ns),
        format_pct(prefill_result.comm_ns as f64, prefill_result.ttft_ns as f64)
    );
    if prefill_result.kv_cache_ns > 0 {
        println!("  KV-cache:      {} ({})",
            format_ns(prefill_result.kv_cache_ns),
            format_pct(prefill_result.kv_cache_ns as f64, prefill_result.ttft_ns as f64)
        );
    }
    println!("  Collectives:   {}", prefill_result.num_collectives);
    println!();

    // Simulate decode
    let decode_result = simulate_token(&base_config);

    println!("--- Decode Stage (per token) ---");
    println!("  Time/token:    {}", format_ns(decode_result.ns_per_token));
    println!("  Throughput:    {}", format_tps(decode_result.tok_per_sec));
    println!();

    // If cache-aware, show how decode degrades with context
    if args.cache_aware {
        println!("--- Cache-Aware Decode (context scaling) ---");

        let prompt_len = args.prompt_len.unwrap_or(args.prompt_tokens);
        let contexts = [
            prompt_len,
            prompt_len + args.output_tokens / 4,
            prompt_len + args.output_tokens / 2,
            prompt_len + args.output_tokens,
        ];

        for &ctx in &contexts {
            let cache_config = DecodeWithCacheConfig {
                base: base_config.clone(),
                context_length: ctx,
                kv_cache_read_ns_per_token: args.kv_read_ns,
            };
            let result = simulate_token_with_cache(&cache_config);
            println!("  Context {:>5}: {} ({})",
                ctx,
                format_ns(result.ns_per_token),
                format_tps(result.tok_per_sec)
            );
        }
        println!();
    }

    // Full request simulation
    let full_result = simulate_full_request(&prefill_config, args.output_tokens);

    println!("--- Full Request Summary ---");
    println!("  Prompt tokens:     {}", args.prompt_len.unwrap_or(args.prompt_tokens));
    println!("  Output tokens:     {}", args.output_tokens);
    println!("  TTFT:              {}", format_ns(full_result.prefill.ttft_ns));
    println!("  Decode time:       {}", format_ns(full_result.decode.ns_per_token * args.output_tokens as u64));
    println!("  Total time:        {}", format_ns(full_result.total_ns));
    println!("  Effective tok/s:   {}", format_tps(full_result.effective_tok_per_sec));
    println!();

    // Compare TP=1 vs TP=N for TTFT
    if base_config.tp > 1 {
        let single_gpu_prefill = model::PrefillConfig {
            base: SimConfig { tp: 1, ..base_config.clone() },
            ..prefill_config.clone()
        };
        let single_result = simulate_full_request(&single_gpu_prefill, args.output_tokens);

        let ttft_overhead = 100.0 * (prefill_result.ttft_ns as f64 - single_result.prefill.ttft_ns as f64)
            / single_result.prefill.ttft_ns as f64;
        let total_overhead = 100.0 * (full_result.total_ns as f64 - single_result.total_ns as f64)
            / single_result.total_ns as f64;

        println!("--- TP={} vs TP=1 Comparison ---", base_config.tp);
        println!("  TP=1 TTFT:         {}", format_ns(single_result.prefill.ttft_ns));
        println!("  TP={} TTFT:        {} ({:+.1}%)", base_config.tp, format_ns(prefill_result.ttft_ns), ttft_overhead);
        println!();
        println!("  TP=1 total:        {}", format_ns(single_result.total_ns));
        println!("  TP={} total:       {} ({:+.1}%)", base_config.tp, format_ns(full_result.total_ns), total_overhead);
    }
}

/// Hint to prevent compiler from optimizing away the value.
#[inline(never)]
fn black_box<T>(x: T) -> T {
    let ptr = &x as *const T;
    unsafe { std::ptr::read_volatile(ptr) }
}

fn run_benchmark(args: &Cli) {
    let config = args.to_config();
    let iters = args.bench_iters;

    println!("=== Micro-benchmark ===");
    println!("WARNING: This is a rough estimate. For accurate benchmarks, use criterion.");
    println!("Iterations: {}", iters);
    println!();

    // Warm up
    for _ in 0..10_000 {
        let result = simulate_token(black_box(&config));
        black_box(result);
    }

    // Timed run
    let start = std::time::Instant::now();
    let mut checksum: u64 = 0;
    for _ in 0..iters {
        let result = simulate_token(black_box(&config));
        checksum = checksum.wrapping_add(black_box(result).ns_per_token);
    }
    let elapsed = start.elapsed();

    black_box(checksum);

    let ns_per_call = elapsed.as_nanos() as f64 / iters as f64;
    let calls_per_sec = if ns_per_call > 0.0 { 1e9 / ns_per_call } else { f64::INFINITY };

    println!("Results:");
    println!("  Total time:    {:.2?}", elapsed);
    println!("  Per call:      {:.1} ns", ns_per_call);
    println!("  Throughput:    {:.1}M calls/sec", calls_per_sec / 1e6);
    println!("  Checksum:      {} (anti-opt)", checksum);

    let result = simulate_token(&config);
    println!();
    println!("Sample result: {} ({})",
        format_ns(result.ns_per_token),
        format_tps(result.tok_per_sec)
    );
}
