"""Check the required installed calibration resource is present in the wheel."""
from pathlib import Path
import json
import zipfile
wheel = max(Path('dist').glob('pitwall-*.whl'), key=lambda p: p.stat().st_mtime)
with zipfile.ZipFile(wheel) as archive:
    data = json.loads(archive.read('pitwall/data/calibrated/priors.json'))
    assert 'circuits' in data
print(f'{wheel.name}: calibrated priors packaged')
