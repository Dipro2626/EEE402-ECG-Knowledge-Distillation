"""
Convert PhysioNet/CinC 2021 WFDB records (CPSC-2018, Georgia, ...) into the
parquet layout that data.py reads for the external databases.

Output layout (one row per record):
    label columns : 'Sinus Rhythm', 'Sinus Bradycardia', 'Sinus Tachycardia',
                    'Sinus Irregularity', 'Atrial Fibrillation', 'Atrial Flutter'
                    (0/1, multi-label as in the source)
    signal columns: signal_1 ... signal_60000 -- 12 leads stored in blocks of
                    5,000 samples (signal_1..5000 = lead I, 5001..10000 = lead II, ...)

Each record is resampled to 500 Hz if needed, then cropped (or zero-padded) to
the first 10 s. No filtering is applied, so the output corresponds to the
"wo_filter" files used in this project; scores can differ slightly from the
processed files we used.

Usage:
    python scripts/convert_cinc2021_to_parquet.py --src <folder with .hea/.mat> --out data/GEORGIA/tweleve_lead_georgia_wo_filters.parquet
    python scripts/convert_cinc2021_to_parquet.py --src <cpsc folder> --out data/CPSC18/cpsc_data_wo_filter.parquet
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

FS, SECONDS, N_LEADS = 500, 10, 12
SNOMED = {                       # SNOMED-CT code -> label column
    '426783006': 'Sinus Rhythm',
    '426177001': 'Sinus Bradycardia',
    '427084000': 'Sinus Tachycardia',
    '427393009': 'Sinus Irregularity',   # sinus arrhythmia
    '164889003': 'Atrial Fibrillation',
    '164890007': 'Atrial Flutter',
}
LABELS = list(dict.fromkeys(SNOMED.values()))


def dx_codes(header):
    for line in header.comments:
        if line.replace(' ', '').lower().startswith('dx:'):
            return [c.strip() for c in line.split(':', 1)[1].split(',')]
    return []


def fixed_length(sig, fs):
    """(n, 12) at fs Hz -> (12, 5000) at 500 Hz, first 10 s, zero-padded."""
    sig = np.nan_to_num(sig.astype(np.float32))
    if fs != FS:
        g = np.gcd(int(FS), int(fs))
        sig = resample_poly(sig, FS // g, int(fs) // g, axis=0).astype(np.float32)
    out = np.zeros((N_LEADS, FS * SECONDS), np.float32)
    n = min(len(sig), FS * SECONDS)
    out[:min(N_LEADS, sig.shape[1]), :n] = sig[:n, :N_LEADS].T
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='folder containing WFDB .hea files (searched recursively)')
    ap.add_argument('--out', required=True, help='output .parquet path')
    a = ap.parse_args()

    heads = sorted(glob.glob(os.path.join(a.src, '**', '*.hea'), recursive=True))
    if not heads:
        raise SystemExit(f'no .hea files under {a.src}')
    labels, signals, skipped = [], [], 0
    for h in heads:
        base = h[:-4]
        try:
            rec = wfdb.rdrecord(base)
        except Exception as e:                       # unreadable record
            print('skip', base, e); skipped += 1
            continue
        codes = dx_codes(rec)
        labels.append([int(any(SNOMED.get(c) == lab for c in codes)) for lab in LABELS])
        signals.append(fixed_length(rec.p_signal, rec.fs).reshape(-1))
    lab = pd.DataFrame(labels, columns=LABELS)
    sig = pd.DataFrame(np.stack(signals), columns=[f'signal_{i}' for i in range(1, N_LEADS * FS * SECONDS + 1)])
    df = pd.concat([lab, sig], axis=1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f'wrote {a.out}: {len(df)} records ({skipped} skipped)')
    print(lab.sum().to_string())


if __name__ == '__main__':
    main()
