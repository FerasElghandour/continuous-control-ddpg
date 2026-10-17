"""Validate source syntax without starting training or external services."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
checked = 0
for path in root.rglob('*'):
    if any(p in {'.git', '.venv', '__pycache__'} for p in path.parts):
        continue
    if path.suffix == '.py':
        ast.parse(path.read_text(), filename=str(path)); checked += 1
    elif path.suffix == '.ipynb':
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                ast.parse(source, filename=f'{path.name}:cell{index}')
                checked += 1
print(f'Validated {checked} source units; no training or provider calls executed.')
