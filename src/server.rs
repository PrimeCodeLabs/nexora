//! Simple HTTP server for interactive visualization.
//!
//! Serves an HTML dashboard that allows parameter adjustment
//! and displays simulation results with charts.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use crate::cli::Cli;
use crate::model::{simulate_token, HardwarePreset, SimConfig};

/// Run the HTTP server on the specified port.
pub fn run_server(args: &Cli, port: u16) {
    let addr = format!("127.0.0.1:{}", port);
    let listener = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Error binding to {}: {}", addr, e);
            return;
        }
    };

    println!("=== Nexora Interactive Visualization ===");
    println!("Server running at: http://{}", addr);
    println!("Press Ctrl+C to stop\n");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => handle_connection(stream, args),
            Err(e) => eprintln!("Connection error: {}", e),
        }
    }
}

fn handle_connection(mut stream: TcpStream, args: &Cli) {
    let mut buffer = [0; 4096];
    if stream.read(&mut buffer).is_err() {
        return;
    }

    let request = String::from_utf8_lossy(&buffer);
    let first_line = request.lines().next().unwrap_or("");

    let response = if first_line.starts_with("GET / ") || first_line.starts_with("GET /index") {
        serve_html()
    } else if first_line.starts_with("GET /api/simulate") {
        // Parse query parameters from URL
        let query = extract_query(&request);
        serve_simulation(&query, args)
    } else if first_line.starts_with("GET /api/presets") {
        serve_presets()
    } else {
        http_response(404, "text/plain", "Not Found")
    };

    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

fn extract_query(request: &str) -> std::collections::HashMap<String, String> {
    let mut params = std::collections::HashMap::new();

    if let Some(line) = request.lines().next() {
        if let Some(query_start) = line.find('?') {
            if let Some(query_end) = line[query_start..].find(' ') {
                let query = &line[query_start + 1..query_start + query_end];
                for pair in query.split('&') {
                    if let Some(eq) = pair.find('=') {
                        let key = pair[..eq].to_string();
                        let value = pair[eq + 1..].to_string();
                        params.insert(key, value);
                    }
                }
            }
        }
    }

    params
}

fn serve_simulation(query: &std::collections::HashMap<String, String>, args: &Cli) -> String {
    // Parse parameters with defaults
    let tp: u32 = query.get("tp").and_then(|s| s.parse().ok()).unwrap_or(4);
    let preset = query.get("preset").map(|s| s.as_str()).unwrap_or("pcie4_bad");
    let layers: u32 = query.get("layers").and_then(|s| s.parse().ok()).unwrap_or(32);
    let batch_size: u32 = query.get("batch_size").and_then(|s| s.parse().ok()).unwrap_or(1);
    let overlap: f64 = query.get("overlap").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let group_size: u32 = query.get("group_size").and_then(|s| s.parse().ok()).unwrap_or(1);

    // Get hardware preset values
    let hw = HardwarePreset::from_str(preset).unwrap_or(HardwarePreset::Pcie4Bad);

    // Create configs for TP=1, TP=2, and TP=N
    let base_config = SimConfig {
        tp,
        layers,
        batch_size,
        overlap: overlap.clamp(0.0, 0.95),
        group_size,
        bw_gbps: hw.bandwidth_gbps(),
        latency_ns: hw.latency_ns(),
        memory_bw_gbps: hw.memory_bw_gbps(),
        ..args.to_config()
    };

    let tp1_config = SimConfig { tp: 1, ..base_config.clone() };
    let tp2_config = SimConfig { tp: 2, ..base_config.clone() };

    let tp1 = simulate_token(&tp1_config);
    let tp2 = simulate_token(&tp2_config);
    let tpn = simulate_token(&base_config);

    // Calculate metrics
    let tp2_overhead = (tp2.ns_per_token as f64 / tp1.ns_per_token as f64 - 1.0) * 100.0;
    let tpn_overhead = (tpn.ns_per_token as f64 / tp1.ns_per_token as f64 - 1.0) * 100.0;

    let json = format!(r#"{{
  "config": {{
    "tp": {},
    "preset": "{}",
    "preset_name": "{}",
    "layers": {},
    "batch_size": {},
    "overlap": {},
    "group_size": {},
    "bw_gbps": {},
    "latency_ns": {}
  }},
  "results": {{
    "tp1": {{
      "ns_per_token": {},
      "tok_per_sec": {:.1},
      "compute_ns": {},
      "comm_ns": {},
      "kv_ns": {}
    }},
    "tp2": {{
      "ns_per_token": {},
      "tok_per_sec": {:.1},
      "overhead_pct": {:.1}
    }},
    "tpn": {{
      "ns_per_token": {},
      "tok_per_sec": {:.1},
      "compute_ns": {},
      "comm_ns": {},
      "kv_ns": {},
      "overhead_pct": {:.1},
      "bottleneck": "{}"
    }}
  }}
}}"#,
        tp, preset, hw.name(), layers, batch_size, overlap, group_size,
        hw.bandwidth_gbps(), hw.latency_ns(),
        tp1.ns_per_token, tp1.tok_per_sec, tp1.compute_total_ns, tp1.comm_total_ns, tp1.kv_overhead_ns,
        tp2.ns_per_token, tp2.tok_per_sec, tp2_overhead,
        tpn.ns_per_token, tpn.tok_per_sec, tpn.compute_total_ns, tpn.comm_total_ns, tpn.kv_overhead_ns,
        tpn_overhead, tpn.bottleneck.name()
    );

    http_response(200, "application/json", &json)
}

fn serve_presets() -> String {
    let presets = vec![
        ("pcie4_bad", "PCIe4 Bad Topology", 12.0, 45000),
        ("pcie4_good", "PCIe4 Good Topology", 18.0, 25000),
        ("pcie5_bad", "PCIe5 Bad Topology", 18.0, 35000),
        ("pcie5_good", "PCIe5 Good Topology", 28.0, 18000),
        ("nvlink", "NVLink 3.0 (A100)", 150.0, 3000),
        ("nvlink4", "NVLink 4.0 (H100)", 225.0, 2000),
        ("nvlink5", "NVLink 5.0 (B200)", 450.0, 1500),
        ("infinity_fabric", "AMD Infinity Fabric (MI300)", 200.0, 3000),
        ("ib_hdr", "InfiniBand HDR", 25.0, 1500),
        ("ib_ndr", "InfiniBand NDR", 50.0, 1000),
    ];

    let items: Vec<String> = presets.iter().map(|(id, name, bw, lat)| {
        format!(r#"{{"id":"{}","name":"{}","bw_gbps":{},"latency_ns":{}}}"#, id, name, bw, lat)
    }).collect();

    let json = format!("[{}]", items.join(","));
    http_response(200, "application/json", &json)
}

fn http_response(status: u16, content_type: &str, body: &str) -> String {
    let status_text = match status {
        200 => "OK",
        404 => "Not Found",
        _ => "Unknown",
    };

    format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{}",
        status, status_text, content_type, body.len(), body
    )
}

fn serve_html() -> String {
    let html = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nexora - GPU Inference Simulator</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        /* ============================================
           Design System: Colors, Typography, Spacing
           ============================================ */
        :root {
            /* Primary palette - limited for cohesion */
            --bg-primary: #0a0f14;
            --bg-secondary: #111820;
            --bg-tertiary: #1a2332;
            --bg-elevated: #222d3d;

            /* Accent colors */
            --accent-primary: #3b82f6;
            --accent-secondary: #8b5cf6;
            --accent-success: #22c55e;
            --accent-warning: #f59e0b;
            --accent-danger: #ef4444;

            /* Text colors */
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;

            /* Border */
            --border-color: #1e293b;
            --border-focus: var(--accent-primary);

            /* Spacing scale (8px base) */
            --space-1: 4px;
            --space-2: 8px;
            --space-3: 12px;
            --space-4: 16px;
            --space-5: 24px;
            --space-6: 32px;
            --space-7: 48px;
            --space-8: 64px;

            /* Typography */
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', 'Fira Code', monospace;

            /* Radius */
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 16px;

            /* Shadows */
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
            --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
            --shadow-lg: 0 8px 24px rgba(0,0,0,0.5);
        }

        /* ============================================
           Base Styles
           ============================================ */
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: var(--font-sans);
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }

        /* ============================================
           Layout Grid System
           ============================================ */
        .app {
            display: grid;
            grid-template-rows: auto 1fr;
            min-height: 100vh;
        }

        .header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: var(--space-5) var(--space-6);
        }

        .header-content {
            max-width: 1600px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            gap: var(--space-4);
        }

        .logo {
            display: flex;
            align-items: center;
            gap: var(--space-3);
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.25rem;
        }

        .logo-text h1 {
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .logo-text p {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: -2px;
        }

        .main {
            max-width: 1600px;
            margin: 0 auto;
            padding: var(--space-6);
            width: 100%;
        }

        .dashboard {
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: var(--space-6);
            align-items: start;
        }

        /* ============================================
           Card Component
           ============================================ */
        .card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            overflow: hidden;
        }

        .card-header {
            padding: var(--space-4) var(--space-5);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: var(--space-3);
        }

        .card-header-icon {
            width: 32px;
            height: 32px;
            background: var(--bg-tertiary);
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--accent-primary);
        }

        .card-title {
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
        }

        .card-body {
            padding: var(--space-5);
        }

        /* ============================================
           Form Controls
           ============================================ */
        .control-group {
            margin-bottom: var(--space-5);
        }

        .control-group:last-child {
            margin-bottom: 0;
        }

        .control-label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: var(--space-2);
        }

        .control-label span {
            font-size: 0.8125rem;
            color: var(--text-secondary);
            font-weight: 500;
        }

        .control-value {
            font-family: var(--font-mono);
            font-size: 0.875rem;
            color: var(--accent-primary);
            font-weight: 500;
            background: var(--bg-tertiary);
            padding: var(--space-1) var(--space-2);
            border-radius: var(--radius-sm);
        }

        select {
            width: 100%;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: var(--space-3);
            border-radius: var(--radius-sm);
            font-family: var(--font-sans);
            font-size: 0.875rem;
            cursor: pointer;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        select:hover {
            border-color: var(--text-muted);
        }

        select:focus {
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
        }

        /* Custom range slider */
        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            background: var(--bg-tertiary);
            border-radius: 3px;
            cursor: pointer;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: var(--accent-primary);
            border-radius: 50%;
            cursor: pointer;
            box-shadow: var(--shadow-md);
            transition: transform 0.15s, box-shadow 0.15s;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.1);
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.3);
        }

        input[type="range"]::-moz-range-thumb {
            width: 18px;
            height: 18px;
            background: var(--accent-primary);
            border-radius: 50%;
            cursor: pointer;
            border: none;
        }

        /* ============================================
           Status Indicators
           ============================================ */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: var(--space-2);
            padding: var(--space-3) var(--space-4);
            border-radius: var(--radius-md);
            font-size: 0.8125rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .status-badge::before {
            content: '';
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
        }

        .status-badge.interconnect {
            background: rgba(239, 68, 68, 0.15);
            color: var(--accent-danger);
        }

        .status-badge.compute {
            background: rgba(34, 197, 94, 0.15);
            color: var(--accent-success);
        }

        .status-badge.memory {
            background: rgba(245, 158, 11, 0.15);
            color: var(--accent-warning);
        }

        .status-badge.balanced {
            background: rgba(59, 130, 246, 0.15);
            color: var(--accent-primary);
        }

        /* ============================================
           Insight Box
           ============================================ */
        .insight-box {
            background: var(--bg-tertiary);
            border-radius: var(--radius-md);
            padding: var(--space-4);
            margin-top: var(--space-4);
            border-left: 3px solid var(--accent-primary);
        }

        .insight-box strong {
            color: var(--text-primary);
        }

        .insight-box p {
            font-size: 0.8125rem;
            color: var(--text-secondary);
            line-height: 1.5;
        }

        /* ============================================
           Charts Grid
           ============================================ */
        .charts-area {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            grid-template-rows: auto auto;
            gap: var(--space-5);
        }

        .chart-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: var(--space-5);
        }

        .chart-card.wide {
            grid-column: span 2;
        }

        .chart-header {
            display: flex;
            align-items: center;
            gap: var(--space-3);
            margin-bottom: var(--space-5);
        }

        .chart-icon {
            width: 36px;
            height: 36px;
            background: var(--bg-tertiary);
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
        }

        .chart-title {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .chart-subtitle {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .chart-wrapper {
            position: relative;
            height: 220px;
        }

        /* ============================================
           Stats Grid
           ============================================ */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: var(--space-4);
        }

        .stat-card {
            background: var(--bg-tertiary);
            border-radius: var(--radius-md);
            padding: var(--space-4);
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }

        .stat-icon {
            width: 40px;
            height: 40px;
            margin: 0 auto var(--space-3);
            background: var(--bg-elevated);
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
        }

        .stat-value {
            font-family: var(--font-mono);
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1.2;
        }

        .stat-value.small {
            font-size: 1rem;
        }

        .stat-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: var(--space-1);
        }

        /* Colored stat values */
        .stat-card.success .stat-value { color: var(--accent-success); }
        .stat-card.warning .stat-value { color: var(--accent-warning); }
        .stat-card.danger .stat-value { color: var(--accent-danger); }
        .stat-card.primary .stat-value { color: var(--accent-primary); }

        /* ============================================
           Responsive
           ============================================ */
        @media (max-width: 1200px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
            .charts-area {
                grid-template-columns: 1fr;
            }
            .chart-card.wide {
                grid-column: span 1;
            }
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 600px) {
            .main {
                padding: var(--space-4);
            }
            .stats-grid {
                grid-template-columns: 1fr 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="app">
        <header class="header">
            <div class="header-content">
                <div class="logo">
                    <div class="logo-icon">N</div>
                    <div class="logo-text">
                        <h1>Nexora</h1>
                        <p>Multi-GPU LLM Inference Simulator</p>
                    </div>
                </div>
            </div>
        </header>

        <main class="main">
            <div class="dashboard">
                <!-- Controls Panel -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-header-icon">⚙</div>
                        <span class="card-title">Configuration</span>
                    </div>
                    <div class="card-body">
                        <div class="control-group">
                            <div class="control-label">
                                <span>Hardware Preset</span>
                            </div>
                            <select id="preset"></select>
                        </div>

                        <div class="control-group">
                            <div class="control-label">
                                <span>Tensor Parallelism</span>
                                <span class="control-value" id="tp-value">4</span>
                            </div>
                            <input type="range" id="tp" min="1" max="8" value="4">
                        </div>

                        <div class="control-group">
                            <div class="control-label">
                                <span>Layers</span>
                                <span class="control-value" id="layers-value">32</span>
                            </div>
                            <input type="range" id="layers" min="8" max="128" value="32" step="8">
                        </div>

                        <div class="control-group">
                            <div class="control-label">
                                <span>Batch Size</span>
                                <span class="control-value" id="batch_size-value">1</span>
                            </div>
                            <input type="range" id="batch_size" min="1" max="64" value="1">
                        </div>

                        <div class="control-group">
                            <div class="control-label">
                                <span>Overlap</span>
                                <span class="control-value" id="overlap-value">0%</span>
                            </div>
                            <input type="range" id="overlap" min="0" max="90" value="0" step="5">
                        </div>

                        <div class="control-group">
                            <div class="control-label">
                                <span>Layer Group Size</span>
                                <span class="control-value" id="group_size-value">1</span>
                            </div>
                            <input type="range" id="group_size" min="1" max="8" value="1">
                        </div>

                        <div style="text-align: center; margin-top: var(--space-5);">
                            <div id="bottleneck" class="status-badge"></div>
                        </div>

                        <div id="insight" class="insight-box">
                            <p>Adjust parameters to see performance analysis.</p>
                        </div>
                    </div>
                </div>

                <!-- Charts Area -->
                <div class="charts-area">
                    <div class="chart-card">
                        <div class="chart-header">
                            <div class="chart-icon">◐</div>
                            <div>
                                <div class="chart-title">Time Breakdown</div>
                                <div class="chart-subtitle">Compute vs Communication vs Memory</div>
                            </div>
                        </div>
                        <div class="chart-wrapper">
                            <canvas id="breakdownChart"></canvas>
                        </div>
                    </div>

                    <div class="chart-card">
                        <div class="chart-header">
                            <div class="chart-icon">▥</div>
                            <div>
                                <div class="chart-title">TP Scaling</div>
                                <div class="chart-subtitle">Latency comparison across TP degrees</div>
                            </div>
                        </div>
                        <div class="chart-wrapper">
                            <canvas id="tpChart"></canvas>
                        </div>
                    </div>

                    <div class="chart-card wide">
                        <div class="chart-header">
                            <div class="chart-icon">◉</div>
                            <div>
                                <div class="chart-title">Performance Metrics</div>
                                <div class="chart-subtitle">Key simulation results</div>
                            </div>
                        </div>
                        <div class="stats-grid" id="stats-grid"></div>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <script>
        let breakdownChart, tpChart;

        async function loadPresets() {
            const res = await fetch('/api/presets');
            const presets = await res.json();
            const select = document.getElementById('preset');
            presets.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p.id;
                opt.textContent = `${p.name}`;
                select.appendChild(opt);
            });
        }

        function formatNs(ns) {
            if (ns >= 1e9) return (ns / 1e9).toFixed(2) + 's';
            if (ns >= 1e6) return (ns / 1e6).toFixed(2) + 'ms';
            if (ns >= 1e3) return (ns / 1e3).toFixed(1) + 'µs';
            return ns + 'ns';
        }

        async function simulate() {
            const params = new URLSearchParams({
                tp: document.getElementById('tp').value,
                preset: document.getElementById('preset').value,
                layers: document.getElementById('layers').value,
                batch_size: document.getElementById('batch_size').value,
                overlap: (parseInt(document.getElementById('overlap').value) / 100).toString(),
                group_size: document.getElementById('group_size').value
            });

            const res = await fetch('/api/simulate?' + params);
            const data = await res.json();
            updateCharts(data);
            updateStats(data);
        }

        function updateCharts(data) {
            const r = data.results.tpn;
            const total = r.compute_ns + r.comm_ns + r.kv_ns;

            breakdownChart.data.datasets[0].data = [
                r.compute_ns / total * 100,
                r.comm_ns / total * 100,
                r.kv_ns / total * 100
            ];
            breakdownChart.update();

            tpChart.data.datasets[0].data = [
                data.results.tp1.ns_per_token / 1e6,
                data.results.tp2.ns_per_token / 1e6,
                data.results.tpn.ns_per_token / 1e6
            ];
            tpChart.data.labels[2] = `TP=${data.config.tp}`;
            tpChart.update();
        }

        function updateStats(data) {
            const r = data.results.tpn;

            // Bottleneck badge
            const bn = document.getElementById('bottleneck');
            bn.textContent = r.bottleneck.replace('-', ' ');
            bn.className = 'status-badge ' + r.bottleneck.toLowerCase().split('-')[0];

            // Insight
            const insight = document.getElementById('insight');
            const overhead = r.overhead_pct;
            if (overhead > 100) {
                insight.innerHTML = '<p><strong>⚠ High Overhead</strong><br>TP=' + data.config.tp + ' is ' + (overhead/100+1).toFixed(1) + 'x slower than TP=1. Consider NVLink or reducing TP degree.</p>';
            } else if (overhead > 25) {
                insight.innerHTML = '<p><strong>⚡ Moderate Overhead</strong><br>' + overhead.toFixed(0) + '% slower than TP=1. Try increasing overlap or layer grouping.</p>';
            } else {
                insight.innerHTML = '<p><strong>✓ Efficient Scaling</strong><br>Only ' + overhead.toFixed(0) + '% overhead. Hardware is well-matched for this workload.</p>';
            }

            // Stats grid
            const computePct = (r.compute_ns / r.ns_per_token * 100).toFixed(1);
            const commPct = (r.comm_ns / r.ns_per_token * 100).toFixed(1);

            document.getElementById('stats-grid').innerHTML = `
                <div class="stat-card">
                    <div class="stat-icon">⏱</div>
                    <div class="stat-value">${formatNs(r.ns_per_token)}</div>
                    <div class="stat-label">Time per Token</div>
                </div>
                <div class="stat-card success">
                    <div class="stat-icon">⚡</div>
                    <div class="stat-value">${r.tok_per_sec.toFixed(0)}</div>
                    <div class="stat-label">Tokens/sec</div>
                </div>
                <div class="stat-card primary">
                    <div class="stat-icon">⬛</div>
                    <div class="stat-value">${computePct}%</div>
                    <div class="stat-label">Compute</div>
                </div>
                <div class="stat-card ${parseFloat(commPct) > 50 ? 'danger' : 'warning'}">
                    <div class="stat-icon">⇄</div>
                    <div class="stat-value">${commPct}%</div>
                    <div class="stat-label">Communication</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">🖥</div>
                    <div class="stat-value small">${data.config.preset_name}</div>
                    <div class="stat-label">Hardware</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">📶</div>
                    <div class="stat-value">${data.config.bw_gbps}</div>
                    <div class="stat-label">GB/s Bandwidth</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">📍</div>
                    <div class="stat-value">${formatNs(data.config.latency_ns)}</div>
                    <div class="stat-label">Latency</div>
                </div>
                <div class="stat-card ${r.overhead_pct > 50 ? 'danger' : r.overhead_pct > 20 ? 'warning' : 'success'}">
                    <div class="stat-icon">📊</div>
                    <div class="stat-value">+${r.overhead_pct.toFixed(1)}%</div>
                    <div class="stat-label">TP Overhead</div>
                </div>
            `;
        }

        function initCharts() {
            Chart.defaults.color = '#94a3b8';
            Chart.defaults.borderColor = '#1e293b';

            breakdownChart = new Chart(document.getElementById('breakdownChart'), {
                type: 'doughnut',
                data: {
                    labels: ['Compute', 'Communication', 'KV Cache'],
                    datasets: [{
                        data: [0, 0, 0],
                        backgroundColor: ['#22c55e', '#ef4444', '#f59e0b'],
                        borderWidth: 0,
                        hoverOffset: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '65%',
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                padding: 16,
                                usePointStyle: true,
                                pointStyle: 'circle'
                            }
                        }
                    }
                }
            });

            tpChart = new Chart(document.getElementById('tpChart'), {
                type: 'bar',
                data: {
                    labels: ['TP=1', 'TP=2', 'TP=4'],
                    datasets: [{
                        label: 'Latency (ms)',
                        data: [0, 0, 0],
                        backgroundColor: ['#22c55e', '#3b82f6', '#8b5cf6'],
                        borderRadius: 6,
                        borderSkipped: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: {
                            grid: { display: false }
                        },
                        y: {
                            beginAtZero: true,
                            grid: { color: '#1e293b' },
                            ticks: {
                                callback: v => v + 'ms'
                            }
                        }
                    }
                }
            });
        }

        // Event listeners
        document.querySelectorAll('input[type="range"]').forEach(input => {
            input.addEventListener('input', () => {
                const id = input.id;
                const val = input.value;
                const display = document.getElementById(id + '-value');
                display.textContent = id === 'overlap' ? val + '%' : val;
                simulate();
            });
        });

        document.getElementById('preset').addEventListener('change', simulate);

        // Initialize
        loadPresets().then(() => {
            initCharts();
            simulate();
        });
    </script>
</body>
</html>"#;

    http_response(200, "text/html", html)
}
