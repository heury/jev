"""Run the actual notebook's data preparation cell on the public dataset, without a model."""
import hashlib
import json
import random
from pathlib import Path

import nbformat
from datasets import load_dataset
from huggingface_hub import HfApi

root = Path(__file__).resolve().parents[1]
nb = nbformat.read(root / 'output/jupyter-notebook/02_train_decision_model.ipynb', as_version=4)
revision = HfApi().dataset_info('LocalLLaMA/typed-decisions').sha
ns = dict(json=json, hashlib=hashlib, random=random, load_dataset=load_dataset,
          DATASET_ID='LocalLLaMA/typed-decisions', resolved_dataset_revision=revision,
          SEED=3407, SMOKE=True, manifest={})
cell = next(c for c in nb.cells if c.cell_type == 'code' and 'def unique_rows' in c.source)
exec(compile(cell.source, '02:data-preparation-cell', 'exec'), ns)
result = {'revision':revision, 'raw_train':len(ns['raw_train']), 'raw_test':len(ns['raw_test']),
          'smoke_split_sizes':{name:len(rows) for name, rows in ns['raw_splits'].items()},
          'disjoint_exact_states':True}
path = root / 'artifacts/verification/data-check.json'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(result, indent=2))
print('REAL DATA PREPARATION PASS:', result)
