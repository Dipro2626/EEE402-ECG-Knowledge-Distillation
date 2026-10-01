"""
Cross-dataset transfer: apply Chapman-trained students to Ningbo, unchanged.

    python evaluate_external.py                 # all checkpoints found
    python evaluate_external.py --modes ce rkd_kd

Design is fixed by PREREGISTER.md Amendment 2, written before any Ningbo result
existed:

  * Sinus Irregularity records are dropped -- 1,232 Ningbo records carry a label
    with no corresponding Chapman output unit, so scoring them would measure a
    label-space mismatch rather than transfer. 15,179 records remain.
  * SVT/AT is excluded from the primary endpoint: Ningbo has 15 such records
    against Chapman's 390. It is reported, but as exploratory.
  * Primary endpoint is AFIB/AFL F1 (1,375 records), the only minority class
    with adequate support on both sides.

Nothing is retrained, fine-tuned or recalibrated, and no Ningbo statistic is
used anywhere: per-record z-scoring uses each record's own mean and standard
deviation. Both datasets are 500 Hz / 5000 samples, so there is no resampling
step to get wrong.
"""

import argparse
import glob
import json
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from data import (load_dataset, load_parquet_dataset, load_mitbih_rhythm,
                  PARQUET)
from student import ECGStudent
from paths import CKPT_DIR, RESULT_DIR

SIG_LEN = 5000
PRIMARY_CLASS = 'AFIB/AFL'
LOW_SUPPORT = 5          # classes below this many records are not measurable
MODE_ORDER = ['ce', 'kd', 'fitnet', 'rkd_d', 'rkd_a', 'rkd', 'rkd_kd']

# PREREGISTER Amendment 5: the confirmatory comparison is CE+KD vs CE.
CONFIRMATORY = ('kd', 'ce')


def build_external(source_names, dataset='ningbo', variant='without_others',
                   lead=1, lead_name='MLII'):
    """
    Returns (signals, labels remapped onto the SOURCE label order, a report of
    what was dropped, kept, total, patient_ids).

    patient_ids is None for the record-per-patient databases and an array for
    MIT-BIH, where one recording contributes many correlated windows and the
    effective sample size is the number of recordings, not the number of rows.
    """
    if dataset == 'mitbih':
        # Rhythm-level MIT-BIH. Two classes survive (SR, AFIB/AFL); the loader
        # prints exactly which were dropped and why. Windows are returned
        # already resampled to 500 Hz, so the only step left is z-scoring, which
        # uses each window's own statistics as everywhere else.
        X, y, pid, drop, _props = load_mitbih_rhythm(source_names,
                                                     lead=lead_name)
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-8
        return (X - mu) / sd, y, drop, len(y), len(y), pid

    if dataset in PARQUET:
        # These are already aligned to the source label space by name inside
        # load_parquet_dataset, including the multi-label filtering, so there is
        # nothing further to remap here.
        X, y = load_parquet_dataset(dataset, source_names, lead=lead)
        sig = X[:, :SIG_LEN]
        mu = sig.mean(axis=1, keepdims=True)
        sd = sig.std(axis=1, keepdims=True) + 1e-8
        return (sig - mu) / sd, y, {}, len(y), len(y), None

    X, y, names = load_dataset(dataset, variant, verbose=False)

    keep_ids = [i for i, n in enumerate(names) if n in source_names]
    dropped = {n: int((y == i).sum())
               for i, n in enumerate(names) if n not in source_names}

    mask = np.isin(y, keep_ids)
    # Relabel into the source model's output order, which is what the network
    # actually predicts. Doing this by name rather than by index is the whole
    # safeguard -- the two datasets do not share a class ordering.
    remap = {i: source_names.index(names[i]) for i in keep_ids}
    y_ext = np.array([remap[v] for v in y[mask]], dtype=np.int64)

    sig = X[mask][:, :5000]
    mu = sig.mean(axis=1, keepdims=True)
    sd = sig.std(axis=1, keepdims=True) + 1e-8
    sig = (sig - mu) / sd

    return sig, y_ext, dropped, int(mask.sum()), len(y), None


def evaluate(ckpt_path, sig, y_ext, class_names, device, batch=256):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck['args']
    model = ECGStudent(num_classes=len(class_names), width=a.get('width', 1.0),
                       embed_dim=a.get('embed_dim', 64),
                       use_gru=not a.get('no_gru', False)).to(device)
    model.load_state_dict(ck['model'])
    model.eval()

    loader = DataLoader(TensorDataset(torch.from_numpy(sig).float()),
                        batch_size=batch, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            logits, _ = model(xb.to(device))
            preds.append(logits.argmax(1).cpu())
    p = torch.cat(preds).numpy()

    labels = list(range(len(class_names)))
    rep = classification_report(y_ext, p, labels=labels,
                                target_names=class_names, output_dict=True,
                                zero_division=0)
    return {
        'mode': a['mode'], 'fold': a['fold'], 'seed': a['seed'],
        'accuracy': float((p == y_ext).mean()),
        'macro_f1': float(f1_score(y_ext, p, average='macro',
                                   labels=labels, zero_division=0)),
        'per_class': rep,
        'confusion': confusion_matrix(y_ext, p, labels=labels).tolist(),
    }, p


def mean_sd(v):
    v = np.asarray(v, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float('nan'))


def paired(runs, a, b, key):
    cells = sorted(set(runs.get(a, {})) & set(runs.get(b, {})))
    if len(cells) < 2:
        return None
    va = np.array([runs[a][c][key] for c in cells])
    vb = np.array([runs[b][c][key] for c in cells])
    try:
        from scipy import stats
        t, p = stats.ttest_rel(va, vb)
    except ImportError:
        t = p = float('nan')
    d = va - vb
    return len(cells), d.mean(), d.std(ddof=1), float(t), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='chapman')
    ap.add_argument('--target', default='ningbo',
                    help='ningbo / ptb (csv), georgia / cpsc (parquet), '
                         'or mitbih (WFDB, rhythm annotations)')
    ap.add_argument('--lead_name', default='MLII',
                    help='which named channel to read from a WFDB target')
    ap.add_argument('--save_preds', action='store_true',
                    help='also write per-record predictions for every run, so a '
                         'failure can be cross-tabulated against measured '
                         'properties of the target rather than guessed at')
    ap.add_argument('--variant', default='without_others')
    ap.add_argument('--min_class', type=int, default=0,
                    help="must match the value the students were trained with; "
                         "it determines the source label space and therefore "
                         "the classifier width")
    ap.add_argument('--lead', type=int, default=1,
                    help='which of the 12 leads to read from a parquet target; '
                         'signal columns are stored in blocks of 5000')
    ap.add_argument('--modes', nargs='*', default=None)
    ap.add_argument('--teacher', default='v2',
                    help='which teacher variant produced the students. Without '
                         'this the v1 and v2 checkpoints would be pooled and '
                         'the comparison would be meaningless.')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    # The source label space must be exactly the one the students were trained
    # on, or the classifier width will not match the checkpoint.
    _, _, source_names = load_dataset(args.source, args.variant, verbose=False,
                                      min_count=args.min_class)

    sig, y_ext, dropped, kept, total, pid = build_external(
        source_names, args.target, args.variant, lead=args.lead,
        lead_name=args.lead_name)

    print(f'{args.source} -> {args.target}')
    print(f'  source classes : {source_names}')
    print(f'  dropped (not in source label space): {dropped}')
    print(f'  evaluated on {kept:,} of {total:,} records\n')

    counts = np.bincount(y_ext, minlength=len(source_names))
    print(f"  {'class':<10}{'target n':>10}{'measurable':>12}")
    for i, n in enumerate(source_names):
        print(f'  {n:<10}{counts[i]:>10}'
              f"{'yes' if counts[i] >= LOW_SUPPORT * 20 else 'NO -- too few':>12}")
    print()

    # Where one recording contributes many windows, the row count overstates the
    # sample size badly. Say so here rather than letting a reader infer n=6,898.
    if pid is not None:
        print('  NOTE: rows are 10-s windows cut from long recordings, so the '
              'effective\n        sample size is the number of RECORDINGS:')
        for i, n in enumerate(source_names):
            if counts[i]:
                print(f'          {n:<10}{counts[i]:>7} windows from '
                      f'{len(set(pid[y_ext == i])):>3} recordings')
        print('        Every figure below should be read against those counts.\n')

    pattern = os.path.join(
        CKPT_DIR, f'student_*_{args.source}_{args.variant}_fold*_seed*.pt')
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(
            f'no student checkpoints in {CKPT_DIR}.\n'
            'The grid must be re-run now that checkpointing is on:\n'
            '    run_grid.bat')

    # CE carries no teacher and is shared between grids; every other mode must
    # match the requested variant or the two grids would be pooled.
    def belongs(path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        a = ck['args']
        return a['mode'] == 'ce' or a.get('teacher', 'v1') == args.teacher

    paths = [p for p in paths if belongs(p)]
    if not paths:
        raise SystemExit(f'no student checkpoints for teacher {args.teacher}')
    print(f'  teacher variant: {args.teacher}  ({len(paths)} checkpoints)\n')

    runs, records, preds = {}, [], []
    for i, path in enumerate(paths, 1):
        r, p = evaluate(path, sig, y_ext, source_names, device)
        if args.modes and r['mode'] not in args.modes:
            continue
        records.append(r)
        if args.save_preds:
            preds.append((r['mode'], r['fold'], r['seed'], p.astype(np.int8)))
        cell = (r['fold'], r['seed'])
        runs.setdefault(r['mode'], {})[cell] = {
            'accuracy': 100 * r['accuracy'],
            'macro_f1': 100 * r['macro_f1'],
            'primary': 100 * r['per_class'][PRIMARY_CLASS]['f1-score'],
            'macro4': float(np.mean([
                100 * r['per_class'][c]['f1-score'] for c in source_names
                if counts[source_names.index(c)] >= LOW_SUPPORT * 20])),
        }
        if i % 20 == 0:
            print(f'  evaluated {i}/{len(paths)}', flush=True)

    print()
    for key, label in [('primary', f'PRIMARY: {PRIMARY_CLASS} F1'),
                       ('macro4', 'SECONDARY: macro-F1 over adequately supported classes'),
                       ('macro_f1', 'EXPLORATORY: 5-class macro-F1 (SVT/AT has 15 records)'),
                       ('accuracy', 'EXPLORATORY: accuracy')]:
        print(f'===== {label} =====')
        for m in MODE_ORDER:
            if m not in runs:
                continue
            mu, sd = mean_sd([v[key] for v in runs[m].values()])
            print(f'  {m:<8} n={len(runs[m]):<3} {mu:6.2f} +- {sd:4.2f}')
        for a, b in [('rkd_kd', 'ce'), ('rkd_kd', 'kd'), ('rkd', 'ce'), ('kd', 'ce')]:
            r = paired(runs, a, b, key)
            if r is None:
                continue
            n, mean, sd, t, p = r
            star = '*' if p < 0.05 else ' '
            print(f'   {a:<7} - {b:<7} n={n:<3} {mean:+6.2f} (sd {sd:4.2f})  '
                  f'p={p:.4f} {star}')
        if key == 'primary':
            print(f'\n  --- verdict, PREREGISTER.md Amendment 5 (confirmatory) ---')
            a, b = CONFIRMATORY
            r = paired(runs, a, b, key)
            if r:
                _, mean, _, _, p = r
                ok = mean >= 1.0 and p < 0.05
                print(f'   CE+KD vs CE on {PRIMARY_CLASS}: {mean:+.2f} pts '
                      f'(need >= 1.0), p={p:.4f} (need < 0.05)  ->  '
                      f'{"SUPPORTED" if ok else "NOT SUPPORTED"}')
                print(f'   Chapman discovery value was +0.96; Ningbo is the '
                      f'held-out confirmation.')
            r2 = paired(runs, 'rkd_kd', 'kd', key)
            if r2:
                _, mean2, _, _, p2 = r2
                print(f'   [secondary] RKD+KD vs CE+KD: {mean2:+.2f} pts, '
                      f'p={p2:.4f} -- does the relational term\'s harm '
                      f'(Chapman: -0.47) replicate?')
        print()

    # Amendment 2 committed to reporting where the dropped records are absorbed.
    # A model that places Sinus Irregularity in the sinus family (SR/SB/ST) is
    # failing more sensibly than one that calls it atrial fibrillation, so this
    # is worth knowing even though the records cannot be scored.
    absorbed = {}
    if dropped and args.target not in PARQUET and args.target != 'mitbih':
        Xd, yd, dnames = load_dataset(args.target, args.variant, verbose=False)
        drop_ids = [i for i, n in enumerate(dnames) if n not in source_names]
        dmask = np.isin(yd, drop_ids)
        dsig = Xd[dmask][:, :5000]
        dsig = (dsig - dsig.mean(1, keepdims=True)) / (dsig.std(1, keepdims=True) + 1e-8)

        # Prefer the deployed configuration, but fall back to whatever exists --
        # a grid without rkd_kd left this list empty and silently reported
        # "0.0% land in the sinus family".
        ck_paths = [p for p in paths if '_kd_' in os.path.basename(p)] or paths
        counts_sum = np.zeros(len(source_names))
        for path in ck_paths:
            ck = torch.load(path, map_location=device, weights_only=False)
            a = ck['args']
            m = ECGStudent(num_classes=len(source_names), width=a.get('width', 1.0),
                           embed_dim=a.get('embed_dim', 64),
                           use_gru=not a.get('no_gru', False)).to(device)
            m.load_state_dict(ck['model'])
            m.eval()
            pr = []
            with torch.no_grad():
                for (xb,) in DataLoader(TensorDataset(torch.from_numpy(dsig).float()),
                                        batch_size=256):
                    pr.append(m(xb.to(device))[0].argmax(1).cpu())
            counts_sum += np.bincount(torch.cat(pr).numpy(),
                                      minlength=len(source_names))
        counts_sum /= max(len(ck_paths), 1)
        absorbed = {n: float(counts_sum[i]) for i, n in enumerate(source_names)}

        print(f'===== dropped records: where they are absorbed (rkd_kd, mean) =====')
        n_drop = int(dmask.sum())
        for n, v in absorbed.items():
            print(f'  {n:<10}{v:>8.0f}  ({100 * v / n_drop:5.1f}%)')
        sinus = sum(absorbed.get(k, 0) for k in ('SR', 'SB', 'ST'))
        print(f'  -> {100 * sinus / n_drop:.1f}% land in the sinus family, '
              f'which is the clinically sensible failure mode\n')

    out = os.path.join(RESULT_DIR, f'external_{args.source}_to_{args.target}.json')
    with open(out, 'w') as fh:
        json.dump({'source': args.source, 'target': args.target,
                   'class_names': source_names, 'dropped': dropped,
                   'evaluated': kept, 'total': total,
                   'n_recordings': (None if pid is None
                                    else len(set(pid.tolist()))),
                   'runs': records}, fh, indent=2)
    print(f'saved -> {out}')

    if args.save_preds and preds:
        pout = os.path.join(RESULT_DIR,
                            f'preds_{args.source}_to_{args.target}.npz')
        np.savez_compressed(
            pout,
            mode=np.array([m for m, _, _, _ in preds]),
            fold=np.array([f for _, f, _, _ in preds]),
            seed=np.array([s for _, _, s, _ in preds]),
            pred=np.stack([p for _, _, _, p in preds]),
            y_true=y_ext.astype(np.int8),
            class_names=np.array(source_names),
            pid=(np.array([]) if pid is None else np.asarray(pid)))
        print(f'saved -> {pout}  ({len(preds)} runs x {len(y_ext):,} records)')


if __name__ == '__main__':
    main()
