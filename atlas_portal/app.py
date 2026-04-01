#!/usr/bin/env python3
"""Atlas Research Portal — Web dashboard for the Atlas workstation.

Provides:
- Live system monitoring (CPU/GPU temps, utilization, memory)
- Running experiment status
- Results dashboard with charts
- Platform guide and resource links
- Quick actions (launch/kill experiments)

Usage:
    python app.py  # serves on http://atlas:8080
"""

import json
import os
import subprocess
import time
import glob
from datetime import datetime
from flask import Flask, render_template_string, jsonify

app = Flask(__name__)

# ─── System monitoring ─────────────────────────────────────────────

def get_cpu_temps():
    try:
        out = subprocess.check_output(["sensors"], text=True, timeout=5)
        temps = {}
        for line in out.splitlines():
            if "Package" in line:
                parts = line.split("+")
                if len(parts) > 1:
                    temp = float(parts[1].split("°")[0])
                    pkg = line.split(":")[0].strip()
                    temps[pkg] = temp
        return temps
    except Exception:
        return {}

def get_gpu_info():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits"
        ], text=True, timeout=5)
        gpus = []
        for line in out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "util": int(parts[2]),
                    "mem_used": int(parts[3]),
                    "mem_total": int(parts[4]),
                    "temp": int(parts[5]),
                })
        return gpus
    except Exception:
        return []

def get_memory():
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                info[parts[0].rstrip(":")] = int(parts[1])
        total = info.get("MemTotal", 0) // 1024
        avail = info.get("MemAvailable", 0) // 1024
        return {"total_mb": total, "available_mb": avail, "used_mb": total - avail}
    except Exception:
        return {}

def get_load():
    try:
        load1, load5, load15 = os.getloadavg()
        return {"1min": round(load1, 1), "5min": round(load5, 1), "15min": round(load15, 1)}
    except Exception:
        return {}

def get_disk():
    try:
        st = os.statvfs("/")
        total = st.f_blocks * st.f_frsize // (1024**3)
        free = st.f_bavail * st.f_frsize // (1024**3)
        return {"total_gb": total, "free_gb": free, "used_gb": total - free}
    except Exception:
        return {}

def get_tmux_sessions():
    try:
        out = subprocess.check_output(
            ["tmux", "list-sessions", "-F", "#{session_name}:#{session_created}"],
            text=True, timeout=5
        )
        sessions = []
        for line in out.strip().split("\n"):
            if ":" in line:
                name, created = line.split(":", 1)
                sessions.append({"name": name, "created": created})
        return sessions
    except Exception:
        return []

def get_results():
    results = []
    for jf in sorted(glob.glob("/home/claude/tensor-3body/result_*.json")):
        try:
            with open(jf) as f:
                r = json.load(f)
            bl = r.get("baselines", {})
            results.append({
                "name": r.get("name", "?"),
                "N": r.get("N", 0),
                "d": r.get("d", 0),
                "test_f1": r.get("test_f1", 0),
                "train_f1": r.get("train_f1", 0),
                "formula": r.get("formula", "?")[:40],
                "vs_gb": f"{bl.get('GB', {}).get('sigma', 0):.1f} {bl.get('GB', {}).get('dir', '?')}",
                "vs_rf": f"{bl.get('RF', {}).get('sigma', 0):.1f} {bl.get('RF', {}).get('dir', '?')}",
                "vs_lr": f"{bl.get('LR', {}).get('sigma', 0):.1f} {bl.get('LR', {}).get('dir', '?')}",
                "beats_gb": bl.get("GB", {}).get("dir") == "A*>",
            })
        except Exception:
            pass
    return results


# ─── API endpoints ─────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "cpu_temps": get_cpu_temps(),
        "gpus": get_gpu_info(),
        "memory": get_memory(),
        "load": get_load(),
        "disk": get_disk(),
        "sessions": get_tmux_sessions(),
    })

@app.route("/api/results")
def api_results():
    return jsonify(get_results())


# ─── Main page ─────────────────────────────────────────────────────

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Atlas Research Portal</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; }
  .header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 24px 32px; border-bottom: 1px solid #2d3748; }
  .header h1 { font-size: 28px; font-weight: 300; letter-spacing: 2px; }
  .header h1 span { color: #63b3ed; font-weight: 600; }
  .header .subtitle { color: #718096; font-size: 14px; margin-top: 4px; }
  .container { max-width: 1400px; margin: 0 auto; padding: 24px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; margin-bottom: 24px; }
  .card { background: #1a202c; border-radius: 12px; padding: 20px; border: 1px solid #2d3748; }
  .card h2 { font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: #718096; margin-bottom: 16px; }
  .metric { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #2d3748; }
  .metric:last-child { border-bottom: none; }
  .metric .label { color: #a0aec0; font-size: 14px; }
  .metric .value { font-size: 18px; font-weight: 600; }
  .temp-safe { color: #48bb78; }
  .temp-warn { color: #ecc94b; }
  .temp-crit { color: #fc8181; }
  .gpu-bar { height: 8px; background: #2d3748; border-radius: 4px; margin-top: 4px; }
  .gpu-bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
  .gpu-bar-fill.high { background: linear-gradient(90deg, #48bb78, #38a169); }
  .gpu-bar-fill.med { background: linear-gradient(90deg, #ecc94b, #d69e2e); }
  .gpu-bar-fill.low { background: linear-gradient(90deg, #4299e1, #3182ce); }
  .results-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .results-table th { text-align: left; padding: 10px 8px; color: #718096; border-bottom: 2px solid #2d3748; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
  .results-table td { padding: 8px; border-bottom: 1px solid #2d3748; }
  .results-table tr:hover { background: #2d3748; }
  .badge-win { background: #276749; color: #9ae6b4; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .badge-loss { background: #742a2a; color: #feb2b2; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
  .session { display: inline-block; background: #2d3748; padding: 4px 10px; border-radius: 6px; margin: 2px; font-size: 12px; color: #a0aec0; }
  .session.active { border-left: 3px solid #48bb78; }
  .resources { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
  .resource-link { display: block; padding: 12px 16px; background: #2d3748; border-radius: 8px; color: #63b3ed; text-decoration: none; font-size: 13px; transition: background 0.2s; }
  .resource-link:hover { background: #4a5568; }
  .resource-link .desc { color: #718096; font-size: 11px; margin-top: 4px; }
  .formula { font-family: 'Consolas', monospace; font-size: 12px; color: #a0aec0; }
  .refresh-note { text-align: center; color: #4a5568; font-size: 11px; padding: 8px; }
</style>
</head>
<body>

<div class="header">
  <h1><span>ATLAS</span> Research Portal</h1>
  <div class="subtitle">HP Z840 — 2x Xeon E5-2690 v3 · 2x Quadro GV100 32GB · Theory Radar · ARC-AGI-2</div>
</div>

<div class="container">

  <!-- System Status -->
  <div class="grid" id="system-grid">
    <div class="card">
      <h2>CPU Temperature</h2>
      <div id="cpu-temps">Loading...</div>
    </div>
    <div class="card">
      <h2>GPU Status</h2>
      <div id="gpu-status">Loading...</div>
    </div>
    <div class="card">
      <h2>System</h2>
      <div id="system-info">Loading...</div>
    </div>
  </div>

  <!-- Active Experiments -->
  <div class="card" style="margin-bottom: 24px;">
    <h2>Active Experiments</h2>
    <div id="sessions">Loading...</div>
  </div>

  <!-- Results -->
  <div class="card" style="margin-bottom: 24px;">
    <h2>Theory Radar Results — Formula vs Ensemble</h2>
    <div style="overflow-x: auto;">
      <table class="results-table" id="results-table">
        <thead>
          <tr>
            <th>Dataset</th><th>N</th><th>d</th><th>Test F1</th>
            <th>Formula</th><th>vs GB</th><th>vs RF</th><th>vs LR</th>
          </tr>
        </thead>
        <tbody id="results-body">
          <tr><td colspan="8">Loading...</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Resources -->
  <div class="card">
    <h2>Resources</h2>
    <div class="resources">
      <a class="resource-link" href="https://github.com/ahb-sjsu/theory-radar">
        Theory Radar <div class="desc">Symbolic formula search — GitHub</div>
      </a>
      <a class="resource-link" href="https://pypi.org/project/theory-radar/">
        theory-radar <div class="desc">PyPI package</div>
      </a>
      <a class="resource-link" href="https://pypi.org/project/batch-probe/">
        batch-probe <div class="desc">GPU batch size + thermal management</div>
      </a>
      <a class="resource-link" href="https://github.com/ahb-sjsu/erisml-lib">
        ErisML <div class="desc">Governed agent modeling language</div>
      </a>
      <a class="resource-link" href="https://www.kaggle.com/competitions/kaggle-measuring-agi">
        Measuring AGI <div class="desc">Kaggle competition — 5 tracks</div>
      </a>
      <a class="resource-link" href="https://www.kaggle.com/competitions/arc-prize-2025">
        ARC-AGI-2 <div class="desc">Abstract reasoning challenge</div>
      </a>
    </div>
  </div>

  <!-- Platform Guide -->
  <div class="card" style="margin-top: 24px;">
    <h2>Platform Guide</h2>
    <div style="font-size: 13px; color: #a0aec0; line-height: 1.8;">
      <strong>Hardware:</strong> 2x Xeon E5-2690 v3 (48 threads), 128GB RAM (upgrading to 320-384GB), 2x Quadro GV100 32GB (NVLink pending)<br>
      <strong>Venvs:</strong> <code>/home/claude/env</code> (Theory Radar: cupy, sklearn) · <code>/home/claude/env-infer</code> (LLM inference: torch, transformers, peft)<br>
      <strong>GLM-5:</strong> <code>/home/claude/models/GLM-5-REAP-50-Q3_K_M/</code> (182GB GGUF) · llama.cpp at <code>/home/claude/llama.cpp/</code><br>
      <strong>Qwen v7:</strong> LoRA adapter at <code>/home/claude/arc-agi-2/models/qwen2.5-7b-arc-code-lora/</code><br>
      <strong>Thermal limit:</strong> Package 0 throttles at 82°C, critical at 100°C. Use batch-probe ThermalJobManager.<br>
      <strong>Access:</strong> Tailscale IP 100.68.134.21 · SSH user claude · RDP via xrdp :3389
    </div>
  </div>

  <div class="refresh-note">Auto-refreshes every 10 seconds</div>
</div>

<script>
function tempClass(t) {
  if (t < 75) return 'temp-safe';
  if (t < 85) return 'temp-warn';
  return 'temp-crit';
}

function gpuBarClass(u) {
  if (u > 80) return 'high';
  if (u > 30) return 'med';
  return 'low';
}

async function refresh() {
  try {
    // System status
    const status = await (await fetch('/api/status')).json();

    // CPU temps
    let cpuHtml = '';
    for (const [pkg, temp] of Object.entries(status.cpu_temps)) {
      cpuHtml += `<div class="metric"><span class="label">${pkg}</span><span class="value ${tempClass(temp)}">${temp}°C</span></div>`;
    }
    const load = status.load;
    cpuHtml += `<div class="metric"><span class="label">Load</span><span class="value">${load['1min']} / ${load['5min']} / ${load['15min']}</span></div>`;
    document.getElementById('cpu-temps').innerHTML = cpuHtml;

    // GPUs
    let gpuHtml = '';
    for (const gpu of status.gpus) {
      const memPct = Math.round(100 * gpu.mem_used / gpu.mem_total);
      gpuHtml += `<div style="margin-bottom: 12px;">
        <div class="metric"><span class="label">GPU ${gpu.index}: ${gpu.name}</span><span class="value ${tempClass(gpu.temp)}">${gpu.temp}°C</span></div>
        <div class="metric"><span class="label">Utilization</span><span class="value">${gpu.util}%</span></div>
        <div class="gpu-bar"><div class="gpu-bar-fill ${gpuBarClass(gpu.util)}" style="width:${gpu.util}%"></div></div>
        <div class="metric"><span class="label">Memory</span><span class="value">${gpu.mem_used}/${gpu.mem_total} MB (${memPct}%)</span></div>
      </div>`;
    }
    document.getElementById('gpu-status').innerHTML = gpuHtml;

    // System
    const mem = status.memory;
    const disk = status.disk;
    document.getElementById('system-info').innerHTML = `
      <div class="metric"><span class="label">RAM</span><span class="value">${Math.round(mem.used_mb/1024)}/${Math.round(mem.total_mb/1024)} GB</span></div>
      <div class="metric"><span class="label">Disk</span><span class="value">${disk.used_gb}/${disk.total_gb} GB</span></div>
      <div class="metric"><span class="label">Time</span><span class="value">${new Date(status.timestamp).toLocaleTimeString()}</span></div>
    `;

    // Sessions
    let sessHtml = '';
    for (const s of status.sessions) {
      sessHtml += `<span class="session active">${s.name}</span>`;
    }
    document.getElementById('sessions').innerHTML = sessHtml || '<span style="color:#4a5568">No active experiments</span>';

    // Results
    const results = await (await fetch('/api/results')).json();
    let tbody = '';
    let wins = 0, total = results.length;
    for (const r of results.sort((a,b) => a.name.localeCompare(b.name))) {
      if (r.beats_gb) wins++;
      const badge = r.beats_gb ? '<span class="badge-win">A*&gt;</span>' : '<span class="badge-loss">GB&gt;</span>';
      tbody += `<tr>
        <td><strong>${r.name}</strong></td>
        <td>${r.N}</td><td>${r.d}</td>
        <td>${r.test_f1.toFixed(4)}</td>
        <td class="formula">${r.formula}</td>
        <td>${badge} ${r.vs_gb}</td>
        <td>${r.vs_rf}</td>
        <td>${r.vs_lr}</td>
      </tr>`;
    }
    tbody += `<tr style="font-weight:600; border-top:2px solid #4a5568">
      <td colspan="5">Formula beats GB: ${wins}/${total}</td>
      <td colspan="3"></td></tr>`;
    document.getElementById('results-body').innerHTML = tbody;

  } catch(e) {
    console.error('Refresh failed:', e);
  }
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(TEMPLATE)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
