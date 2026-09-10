"""Verify packaged priors through a real isolated target install, offline."""
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
wheel = max((root / 'dist').glob('pitwall-*.whl'), key=lambda p: p.stat().st_mtime)
(root / '.cache').mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=root / '.cache', prefix='wheel-check-') as target:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                    '--target', target, str(wheel)], check=True)
    code = "import sys; sys.path.insert(0, sys.argv[1]); from pitwall.data.priors import PRIORS_PATH, load_priors; assert str(PRIORS_PATH).startswith(sys.argv[1]); assert 'circuits' in load_priors(); print('Installed priors resolve outside source tree')"
    subprocess.run([sys.executable, '-I', '-c', code, target], check=True)
print(f'{wheel.name}: installed calibration resource verified')
