"""
Build every per-class table the report needs, from the stored result JSONs, into
one markdown file.

    python make_all_tables.py                     # -> CLASSWISE_TABLES.md
    python make_all_tables.py --out ..\\TABLES.md

Nothing is recomputed: each number comes from the JSON the experiment wrote, so
a table can never drift from the run that produced it. Folds and seeds are
averaged, and the spread across them is reported alongside the mean rather than
hidden -- a class whose F1 moves by two points between seeds is not a class that
should be quoted to two decimal places without that context.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from paths import RESULT_DIR

VARIANT = 'without_others'

# In-domain arms to tabulate: (dataset, min_class the students were trained with)
IN_DOMAIN = [('chapman', 0), ('ningbo', 100),
             ('chapman4', 0), ('ningbo4', 0), ('pooled', 0)]

MODE_LABEL = {'ce': 'CE', 'kd': 'CE+KD', 'rkd': 'CE+RKD', 'rkd_kd': 'CE+RKD+KD'}
MODE_ORDER = ['ce', 'kd', 'rkd', 'rkd_kd']


def mean_sd(v):
    v = np.asarray(v, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def fmt(v):
    m, s = mean_sd(v)
    return f'{m:.2f}' + (f' ± {s:.2f}' if s > 0.005 else '')


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_teacher(dataset, tag='v2'):
    """5-fold teacher runs for one dataset -> list of records."""
    out = []
    for p in sorted(glob.glob(os.path.join(
            RESULT_DIR, f'teacher{tag}_{dataset}_{VARIANT}*_fold*.json'))):
        if 'argmax' in os.path.basename(p):
            continue            # attempt 1, kept for the record, not for tables
        out.append(json.load(open(p)))
    return out


def load_students(dataset, min_class=0, teacher='v2'):
    """-> {mode: [records]} for the default architecture only."""
    out = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(
            RESULT_DIR, f'student_*_{dataset}_{VARIANT}_fold*_seed*.json'))):
        r = json.load(open(p))
        a = r['args']
        if a.get('no_gru') or float(a.get('width', 1.0)) != 1.0 \
                or int(a.get('embed_dim', 64)) != 64 \
                or int(a.get('min_class', 0)) != int(min_class):
            continue
        if a['mode'] != 'ce' and a.get('teacher', 'v1') != teacher:
            continue
        out[a['mode']].append(r)
    return out


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def in_domain_table(dataset, min_class, fh):
    teach = load_teacher(dataset)
    stud = load_students(dataset, min_class)
    if not stud:
        return False

    names = next(iter(stud.values()))[0]['class_names']
    modes = [m for m in MODE_ORDER if m in stud]

    fh.write(f'\n### {dataset} — in domain\n\n')
    n_seed = len({(r["args"]["fold"], r["args"]["seed"]) for r in stud[modes[0]]})
    fh.write(f'Student runs: {n_seed} (fold, seed) cells per mode. '
             f'Teacher: {len(teach)} folds.\n\n')

    cols = (['Teacher v2'] if teach else []) + [MODE_LABEL[m] for m in modes]
    fh.write('| class | support | ' + ' | '.join(cols) + ' |\n')
    fh.write('|---|---:|' + '---:|' * len(cols) + '\n')

    for c in names:
        sup = int(np.mean([r['val']['per_class'][c]['support']
                           for r in stud[modes[0]]]))
        row = []
        if teach:
            row.append(fmt([100 * r['val']['per_class'][c]['f1-score']
                            for r in teach]))
        for m in modes:
            row.append(fmt([100 * r['val']['per_class'][c]['f1-score']
                            for r in stud[m]]))
        fh.write(f'| {c} | {sup:,} | ' + ' | '.join(row) + ' |\n')

    for key, label in [('macro_f1', '**macro-F1**'), ('accuracy', '**accuracy**')]:
        row = []
        if teach:
            row.append(fmt([100 * r['val'][key] for r in teach]))
        for m in modes:
            row.append(fmt([100 * r['val'][key] for r in stud[m]]))
        fh.write(f'| {label} | | ' + ' | '.join(row) + ' |\n')

    par_t = teach[0]['parameters'] if teach else None
    par_s = stud[modes[0]][0]['parameters']
    fh.write(f'| **parameters** | | '
             + (f'{par_t:,} | ' if teach else '')
             + ' | '.join([f'{par_s:,}'] * len(modes)) + ' |\n')

    if teach and par_t:
        fh.write(f'\nCompression: **{par_t / par_s:.1f}×**. '
                 f'Values are F1 (%) unless labelled otherwise; ± is the '
                 f'standard deviation across runs.\n')
    return True


def external_table(path, fh):
    e = json.load(open(path))
    runs = defaultdict(list)
    for r in e['runs']:
        runs[r['mode']].append(r)
    names = e['class_names']
    modes = [m for m in MODE_ORDER if m in runs]
    if not modes:
        return

    sup = {c: int(e['runs'][0]['per_class'][c]['support']) for c in names}
    live = [c for c in names if sup[c] >= 100]

    fh.write(f"\n### {e['source']} → {e['target']}\n\n")
    fh.write(f"Evaluated on {e['evaluated']:,} of {e['total']:,} records, "
             f"unchanged — no retraining, fine-tuning or recalibration.")
    if e.get('n_recordings'):
        fh.write(f" **These are 10-s windows from {e['n_recordings']} "
                 f"recordings, so the effective sample size is the recording "
                 f"count, not the row count.**")
    fh.write('\n\n')

    fh.write('| class | target n | ' + ' | '.join(MODE_LABEL[m] for m in modes)
             + ' | KD − CE |\n')
    fh.write('|---|---:|' + '---:|' * (len(modes) + 1) + '\n')

    for c in names:
        vals = {m: [100 * r['per_class'][c]['f1-score'] for r in runs[m]]
                for m in modes}
        note = '' if sup[c] >= 100 else ' *(too few)*'
        delta = ''
        if 'kd' in vals and 'ce' in vals:
            cells = sorted(set((r['fold'], r['seed']) for r in runs['kd'])
                           & set((r['fold'], r['seed']) for r in runs['ce']))
            if cells:
                idx = {m: {(r['fold'], r['seed']): 100 * r['per_class'][c]['f1-score']
                           for r in runs[m]} for m in ('kd', 'ce')}
                d = np.array([idx['kd'][k] - idx['ce'][k] for k in cells])
                delta = f'{d.mean():+.2f}'
        fh.write(f'| {c}{note} | {sup[c]:,} | '
                 + ' | '.join(fmt(vals[m]) for m in modes)
                 + f' | {delta} |\n')

    for key, label in [('macro_f1', '**macro-F1, all classes**'),
                       ('accuracy', '**accuracy**')]:
        fh.write(f'| {label} | | '
                 + ' | '.join(fmt([100 * r[key] for r in runs[m]])
                              for m in modes) + ' | |\n')
    if live and len(live) != len(names):
        fh.write('| **macro-F1, supported classes only** | | '
                 + ' | '.join(fmt([float(np.mean(
                     [100 * r['per_class'][c]['f1-score'] for c in live]))
                     for r in runs[m]]) for m in modes) + ' | |\n')
    if e.get('dropped'):
        fh.write(f"\nDropped, absent from the model's label space: "
                 f"{e['dropped']}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'docs', 'CLASSWISE_TABLES.md'))
    args = ap.parse_args()

    with open(args.out, 'w', encoding='utf-8') as fh:
        fh.write('# Per-class results\n\n')
        fh.write('Generated by `make_all_tables.py` from the stored result '
                 'JSONs. Every figure is the one the experiment wrote; nothing '
                 'is recomputed here.\n\n')

        fh.write('## In-domain\n')
        any_in = False
        for ds, mc in IN_DOMAIN:
            any_in |= in_domain_table(ds, mc, fh)
        if not any_in:
            fh.write('\n*(no in-domain student results found)*\n')

        fh.write('\n---\n\n## Cross-database\n')
        paths = sorted(glob.glob(os.path.join(RESULT_DIR, 'external_*.json')))
        for p in paths:
            external_table(p, fh)
        if not paths:
            fh.write('\n*(no external results found)*\n')

    print(f'wrote {os.path.abspath(args.out)}')
    print(f'  in-domain arms : '
          f'{[d for d, _ in IN_DOMAIN if load_students(d, _)]}')
    print(f'  external files : {len(paths)}')


if __name__ == '__main__':
    main()
