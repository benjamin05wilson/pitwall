"""Bounded, network-blocked API + synthetic Kalman demonstration; no training."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi.testclient import TestClient
from pitwall import server
from pitwall.live.updater import KalmanTyreModel
from pitwall.models.params import FuelModel


def blocked(*args, **kwargs):
    raise RuntimeError('Network disabled by offline_demo.py')
socket.socket.connect = blocked
socket.create_connection = blocked
out = ROOT / 'evidence' / 'offline'
out.mkdir(parents=True, exist_ok=True)
request = json.loads((ROOT / 'fixtures/offline-request.json').read_text())
start = time.perf_counter()
with TestClient(server.app) as client:
    health = client.get('/api/health').json()
    response = client.post('/api/optimize', json=request)
    response.raise_for_status()
    result = response.json()
    heatmap_status = client.post('/api/heatmap', json=request).status_code
elapsed = time.perf_counter() - start
(out / 'recommendation.json').write_text(json.dumps(result, indent=2) + '\n')
fixture = json.loads((ROOT / 'fixtures/synthetic-stint.json').read_text())
kalman = KalmanTyreModel(24, fuel=FuelModel(k_fuel=0), prior_pace=95, prior_deg=.05)
updates = [kalman.update(row['lap'], row['tyre_age'], row['corrected_lap_s']) for row in fixture['laps']]
rows = [{'lap': u.lap, 'observed': u.observed, 'predicted_before_update': u.predicted,
         'anomaly': bool(u.anomaly), 'deg_rate': u.belief.deg_rate,
         'deg_lo': u.belief.credible_deg()[0], 'deg_hi': u.belief.credible_deg()[1]} for u in updates]
(out / 'kalman.json').write_text(json.dumps(rows, indent=2) + '\n')
fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
laps = [u.lap for u in updates]
axes[0].plot(laps, [u.observed for u in updates], 'o-', label='Synthetic observation', markersize=3)
axes[0].plot(laps, [u.predicted for u in updates], label='One-step prediction')
axes[0].scatter([u.lap for u in updates if u.anomaly], [u.observed for u in updates if u.anomaly], marker='x', color='red', label='Flagged anomaly', zorder=3)
axes[0].set(ylabel='Fuel-corrected lap time (s)', title='Synthetic stint · planted +3 s anomaly on lap 12')
axes[0].legend()
axes[1].plot(laps, [r['deg_rate'] for r in rows], label='Posterior slope')
axes[1].fill_between(laps, [r['deg_lo'] for r in rows], [r['deg_hi'] for r in rows], alpha=.2, label='Model 90% band; coverage unvalidated')
axes[1].axhline(.08, color='black', linestyle='--', label='Planted slope')
axes[1].set(xlabel='Lap', ylabel='Degradation (s/lap)')
axes[1].legend()
fig.savefig(out / 'synthetic-stint.png', dpi=150)
plt.close(fig)
manifest = {
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'command': 'python scripts/offline_demo.py', 'network': 'socket connections blocked after imports',
    'environment': {'python': sys.version, 'platform': platform.platform(), 'packages': {n: importlib.metadata.version(n) for n in ('numpy','scipy','pandas','fastapi','matplotlib')}},
    'inputs': request, 'scenario_seed': 7, 'rival_seed': 1, 'shortlist': 3, 'cars': 20, 'laps': 57,
    'health': health, 'heatmap_http_status': heatmap_status, 'elapsed_seconds': round(elapsed, 4),
    'timing_definition': 'One local TestClient health + optimize + heatmap request, excludes imports, plot and browser. Not a benchmark.',
    'metrics': {'mean_pos': 'Mean simulated finishing position, includes DNFs in classification', 'p_podium': 'Fraction of scenarios classified top three', 'cvar10': 'Mean of worst ceil(10% * scenarios) positions'},
    'fixture_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'fixtures').glob('*.json'))},
}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
transcript = f"{manifest['command']}\nsource commit: {manifest['source_commit']}\nnetwork connections blocked\n{result['backend']}; focal: {result['focal_model']}; rivals: {result['rival_model']}\n20 scenarios; 3 candidates; 57 laps; scenario seed 7; rival seed 1\noptional heatmap HTTP {heatmap_status}\n"
for row in result['ranked']:
    transcript += f"{row['label']}: mean position {row['mean_pos']}, podium {row['p_podium']}, CVaR10 {row['cvar10']}\n"
transcript += f"request elapsed: {elapsed:.4f} seconds (local illustration only)\nsynthetic anomaly flags: {[u.lap for u in updates if u.anomaly]}\n"
(out / 'transcript.txt').write_text(transcript)
print(transcript)
