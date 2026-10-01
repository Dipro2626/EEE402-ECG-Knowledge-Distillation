"""
Paired significance tests across folds and seeds, against the pre-registered
endpoints in PREREGISTER.md (Amendment 1).

    python analyze.py                 # all folds found
    python analyze.py --folds 0       # just the screening fold

Pairing
-------
Every mode is trained on the same (fold, seed) grid, and the seed is set before
model construction, so two modes sharing a (fold, seed) start from identical
weights and see identical batch order. That makes the comparison paired, which
is far more sensitive than comparing marginal means: the fold-to-fold variation
(which is large) cancels out, leaving only the effect of the loss.

Only (fold, seed) cells present for BOTH modes are used, so a partially
finished grid gives a valid if smaller test rather than a wrong one.
"""

import argparse
import glob
import json
import os
import numpy as np

from paths import RESULT_DIR

# Amendment 1 defined the primary endpoint as "the two rarest classes". On
# Chapman those are SVT/AT and AFIB/AFL, which is what was hardcoded. Deriving
# them from the observed support instead gives the identical answer there and
# also works on datasets with a different class list -- Ningbo has no SVT/AT
# (15 records, dropped) but does have Sinus Irregularity. The definition is
# unchanged; only its implementation is now general.
MINORITY_K = 2


def minority_classes(runs):
    """The MINORITY_K rarest classes by validation support."""
    rec = next(iter(next(iter(runs.values())).values()))
    per = rec['val']['per_class']
    names = rec['class_names']
    sup = {c: per[c]['support'] for c in names if c in per}
    return [c for c, _ in sorted(sup.items(), key=lambda kv: kv[1])[:MINORITY_K]]
CONFIRMATORY = [('rkd_kd', 'ce'), ('rkd_kd', 'kd')]
EFFECT_THRESHOLD = 1.0
ALPHA = 0.05

MODE_ORDER = ['ce', 'kd', 'fitnet', 'rkd_d', 'rkd_a', 'rkd', 'rkd_kd']


def load(dataset, variant, folds=None, teacher='v2',
         no_gru=False, width=1.0, embed_dim=64, min_class=0):
    """
    -> {mode: {(fold, seed): record}}

    Filtering on the teacher version matters: the v1 and v2 grids live in the
    same directory and mixing them would compare students distilled from
    different teachers as if they were the same experiment. CE runs use no
    teacher and are shared by both.
    """
    out = {}
    pat = os.path.join(RESULT_DIR, f'student_*_{dataset}_{variant}_fold*_seed*.json')
    for path in glob.glob(pat):
        r = json.load(open(path))
        a = r['args']
        if folds and a['fold'] not in folds:
            continue
        if a['mode'] != 'ce' and a.get('teacher', 'v1') != teacher:
            continue
        # Architecture variants (--no_gru, --width, --embed_dim) are separate
        # experiments and must not be pooled with the main grid.
        if (a.get('no_gru', False) != no_gru
                or float(a.get('width', 1.0)) != float(width)
                or int(a.get('embed_dim', 64)) != int(embed_dim)
                or int(a.get('min_class', 0)) != int(min_class)):
            continue
        out.setdefault(a['mode'], {})[(a['fold'], a['seed'])] = r
    return out


def metric(rec, key, minority=None):
    v = rec['val']
    if key == 'macro_f1':
        return 100 * v['macro_f1']
    if key == 'accuracy':
        return 100 * v['accuracy']
    if key == 'minority':
        return float(np.mean([100 * v['per_class'][c]['f1-score']
                              for c in minority]))
    return 100 * v['per_class'][key]['f1-score']


def paired(runs, a, b, key, minority=None):
    """Returns (n, mean_diff, sd_diff, t, p) over cells present in both modes."""
    cells = sorted(set(runs.get(a, {})) & set(runs.get(b, {})))
    if len(cells) < 2:
        return None
    da = np.array([metric(runs[a][c], key, minority) for c in cells])
    db = np.array([metric(runs[b][c], key, minority) for c in cells])
    d = da - db
    try:
        from scipy import stats
        t, p = stats.ttest_rel(da, db)
    except ImportError:
        # No scipy: report the effect and mark the p-value unavailable rather
        # than hand-rolling an incomplete beta. Study 1 lost a day to exactly
        # that mistake -- a hand-written p-value returned 0.024 for t=0.03.
        t, p = float('nan'), float('nan')
    return len(cells), d.mean(), d.std(ddof=1), float(t), float(p)


def summarise(runs, key, label, minority=None):
    print(f'\n===== {label} =====')
    print(f"{'mode':<9}{'n':>4}{'mean':>9}{'sd':>7}   per-fold means")
    print('-' * 72)
    for m in MODE_ORDER:
        if m not in runs:
            continue
        cells = sorted(runs[m])
        v = np.array([metric(runs[m][c], key, minority) for c in cells])
        folds = sorted({f for f, _ in cells})
        pf = [np.mean([metric(runs[m][c], key, minority)
                       for c in cells if c[0] == f]) for f in folds]
        print(f'{m:<9}{len(v):>4}{v.mean():>9.2f}{v.std(ddof=1):>7.2f}   '
              + ' '.join(f'{x:.2f}' for x in pf))
    return


def tests(runs, key, pairs, tag, minority=None):
    print(f'  --- paired tests ({tag}) ---')
    rows = []
    for a, b in pairs:
        r = paired(runs, a, b, key, minority)
        if r is None:
            continue
        n, mean, sd, t, p = r
        star = '*' if p < ALPHA else ' '
        print(f'   {a:<7} - {b:<7} n={n:<3} {mean:+6.2f} (sd {sd:4.2f})  '
              f't={t:+6.2f}  p={p:.4f} {star}')
        rows.append((a, b, n, mean, sd, p))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='chapman')
    p.add_argument('--variant', default='without_others')
    p.add_argument('--folds', type=int, nargs='*', default=None)
    p.add_argument('--teacher', default='v2',
                   help='which teacher variant the distilled runs came from '
                        '(v1, v2, v2n, ...)')
    p.add_argument('--no_gru', action='store_true',
                   help='analyse the convolution-only students instead')
    p.add_argument('--width', type=float, default=1.0)
    p.add_argument('--embed_dim', type=int, default=64)
    p.add_argument('--min_class', type=int, default=0,
                   help='select runs trained with this class-size cutoff')
    args = p.parse_args()

    runs = load(args.dataset, args.variant, args.folds, args.teacher,
                args.no_gru, args.width, args.embed_dim, args.min_class)
    if not runs:
        raise SystemExit(f'no student results for teacher={args.teacher} '
                         f'no_gru={args.no_gru} width={args.width}')
    arch = ('conv-only' if args.no_gru else 'CNN-GRU') + \
           (f', width {args.width:g}' if args.width != 1.0 else '')
    print(f'teacher: {args.teacher}   student: {arch}')

    cells = sorted({c for m in runs.values() for c in m})
    folds = sorted({f for f, _ in cells})
    seeds = sorted({s for _, s in cells})
    print(f'{args.dataset}/{args.variant}')
    print(f'folds {folds}  seeds {seeds}  modes {sorted(runs)}')
    expected = len(folds) * len(seeds)
    for m in sorted(runs):
        if len(runs[m]) != expected:
            print(f'  NOTE: mode {m} has {len(runs[m])}/{expected} cells')

    # ---- PRIMARY -----------------------------------------------------------
    MINORITY = minority_classes(runs)
    sup = {c: int(next(iter(next(iter(runs.values())).values()))
                  ['val']['per_class'][c]['support']) for c in MINORITY}
    print(f'\nminority classes (the {MINORITY_K} rarest by support): '
          f'{", ".join(f"{c} n={sup[c]}" for c in MINORITY)}')

    summarise(runs, 'minority', 'PRIMARY: minority-class F1 (mean of '
                                + ', '.join(MINORITY) + ')', MINORITY)
    rows = tests(runs, 'minority', CONFIRMATORY, 'confirmatory, pre-registered',
                 MINORITY)

    print('\n  --- verdict against PREREGISTER.md Amendment 1 ---')
    for a, b, n, mean, sd, p in rows:
        ok_e = mean >= EFFECT_THRESHOLD
        ok_p = p < ALPHA
        verdict = ('SUPPORTED' if ok_e and ok_p else
                   'NOT SUPPORTED — effect below 1.0 point' if not ok_e and ok_p else
                   'NOT SUPPORTED — not significant' if ok_e else
                   'NOT SUPPORTED')
        print(f'   {a} vs {b}: {mean:+.2f} pts (need >= {EFFECT_THRESHOLD}), '
              f'p={p:.4f} (need < {ALPHA})  ->  {verdict}')

    # ---- SECONDARY ---------------------------------------------------------
    summarise(runs, 'macro_f1', 'SECONDARY: macro-F1')
    tests(runs, 'macro_f1', CONFIRMATORY, 'secondary')

    # ---- EXPLORATORY -------------------------------------------------------
    names = json.loads(json.dumps(next(iter(runs.values()))[cells[0]]['class_names']))
    extra = [('rkd', 'ce'), ('kd', 'ce'), ('rkd_kd', 'rkd')]

    # The minority endpoint is also reported for the CE+KD contrast, which was
    # not pre-registered and is therefore exploratory -- but it is the quantity
    # the primary endpoint is defined on, so omitting it would be perverse.
    tests(runs, 'minority', [('kd', 'ce')],
          'exploratory, minority-class endpoint', MINORITY)

    summarise(runs, 'accuracy', 'EXPLORATORY: accuracy')
    tests(runs, 'accuracy', CONFIRMATORY + extra, 'exploratory')

    for c in names:
        summarise(runs, c, f'EXPLORATORY: {c} F1')
        tests(runs, c, CONFIRMATORY + extra, 'exploratory')

    print('\nEverything under EXPLORATORY is uncorrected for multiple '
          'comparisons and must be reported as hypothesis-generating only.')


if __name__ == '__main__':
    main()
