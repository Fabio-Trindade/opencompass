import os

from mmengine.config import read_base

with read_base():
    from .ruler_cwe_gen import cwe_datasets as cwe  # CWE
    from .ruler_fwe_gen import fwe_datasets as fwe  # FWE
    from .ruler_niah_gen import niah_datasets as niah  # Niah
    from .ruler_qa_gen import qa_datasets as qa  # QA
    from .ruler_vt_gen import vt_datasets as vt  # VT


import_ds = sum((cwe, fwe, niah, qa, vt), [])

NUM_SAMPLES = 20
  
tokenizer_model = os.environ.get('TOKENIZER_MODEL', 'gpt-4')

max_seq_lens = [ 8000, 64000, 120000]
abbr_suffixs = ['8k', '64k', '120k']

ruler_datasets = []

for max_seq_len, abbr_suffix in zip(max_seq_lens, abbr_suffixs):
    for dataset in import_ds:
        tmp_dataset = dataset.deepcopy()
        tmp_dataset['abbr'] = tmp_dataset['abbr'] + '_' + abbr_suffix
        tmp_dataset['num_samples'] = NUM_SAMPLES
        tmp_dataset['max_seq_length'] = max_seq_len
        tmp_dataset['tokenizer_model'] = tokenizer_model
        ruler_datasets.append(tmp_dataset)
