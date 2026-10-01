"""
Where the CSVs live and where outputs go.

Defaults are the local layout on this PC. Both are overridable by environment
variable so the identical scripts run on Kaggle, where the input folder is
read-only and everything written must land in /kaggle/working:

    set EXGNET_DATA_DIR=/kaggle/input/chap-ning-ptb
    set EXGNET_OUT_DIR=/kaggle/working/exgnet

Nothing else in the codebase constructs a path, so these two variables are the
whole portability surface.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.environ.get(
    'EXGNET_DATA_DIR',
    os.path.join(HERE, 'data'))

OUT_DIR = os.environ.get('EXGNET_OUT_DIR', HERE)

# MIT-BIH ships as WFDB records, not CSV, so it sits outside DATA_DIR.
MITBIH_DIR = os.environ.get(
    'EXGNET_MITBIH_DIR',
    os.path.join(HERE, 'data', 'mit-bih-arrhythmia-database-1.0.0'))

CACHE_DIR = os.path.join(OUT_DIR, 'cache')
CKPT_DIR = os.path.join(OUT_DIR, 'checkpoints')
RESULT_DIR = os.path.join(OUT_DIR, 'results')


def ensure_dirs():
    for d in (CACHE_DIR, CKPT_DIR, RESULT_DIR):
        os.makedirs(d, exist_ok=True)


def describe():
    return (f'data   : {DATA_DIR}\n'
            f'cache  : {CACHE_DIR}\n'
            f'ckpt   : {CKPT_DIR}\n'
            f'results: {RESULT_DIR}')


if __name__ == '__main__':
    print(describe())
    print('\ndata dir exists:', os.path.isdir(DATA_DIR))
    if os.path.isdir(DATA_DIR):
        print('files:', sorted(os.listdir(DATA_DIR))[:8])
