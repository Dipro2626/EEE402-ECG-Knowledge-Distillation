"""
Is the MIT-BIH failure the model's, or the label's?

    python evaluate_external.py --source ningbo --target mitbih \
           --teacher v2 --min_class 100 --save_preds
    python analyze_mitbih.py

MIT-BIH annotates rhythm with a single '(N' for normal sinus rhythm. Chapman and
Ningbo split that same territory four ways -- Sinus Bradycardia below 60 bpm,
Sinus Rhythm between 60 and 100, Sinus Tachycardia above 100, Sinus Irregularity
when the RR interval is unstable. A model trained on the finer label space will
therefore be scored wrong on MIT-BIH every time it makes the finer distinction
correctly.

That is an excuse unless it is measured. MIT-BIH ships expert beat annotations,
so each window's true rate and RR variability are known exactly. This script
cross-tabulates what the model predicted against what the rate actually was.

The claim it can support: "of the windows MIT-BIH calls normal sinus rhythm and
the model calls bradycardia, the median measured rate is X bpm." If X is below
60, the label is coarse and the model is right. If X is 75, the model is wrong
and the excuse fails.
"""

import argparse
import os
import numpy as np

from data import load_mitbih_rhythm
from paths import RESULT_DIR

# The thresholds Chapman and Ningbo use, and standard clinical practice.
BRADY, TACHY = 60.0, 100.0
IRREGULAR_CV = 0.10          # RR coefficient of variation above this is unstable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='ningbo')
    ap.add_argument('--mode', default='ce',
                    help='which student mode to cross-tabulate')
    args = ap.parse_args()

    path = os.path.join(RESULT_DIR, f'preds_{args.source}_to_mitbih.npz')
    if not os.path.exists(path):
        raise SystemExit(
            f'{path} not found.\nRe-run the evaluation with --save_preds:\n'
            f'  python evaluate_external.py --source {args.source} '
            f'--target mitbih --teacher v2 --min_class 100 --save_preds')

    z = np.load(path, allow_pickle=True)
    names = [str(c) for c in z['class_names']]
    sel = z['mode'] == args.mode
    if not sel.any():
        raise SystemExit(f'no runs with mode {args.mode}; '
                         f'available: {sorted(set(z["mode"].tolist()))}')
    P = z['pred'][sel]                       # (runs, windows)
    y = z['y_true']

    # Majority vote across the runs, so each window gets one prediction rather
    # than 25. Ties go to the lowest class index, which is arbitrary but rare.
    vote = np.stack([(P == k).sum(0) for k in range(len(names))])
    pred = vote.argmax(0)
    agree = vote.max(0) / P.shape[0]

    _, y2, pid, _, props = load_mitbih_rhythm(names, verbose=False)
    assert np.array_equal(y, y2.astype(np.int8)), \
        'prediction file and loader disagree on window order; re-run the eval'
    bpm, cv, ect = props[:, 0], props[:, 1], props[:, 2]

    sr = names.index('SR')
    is_n = y == sr                            # MIT-BIH calls these '(N'

    print(f'MIT-BIH windows labelled sinus rhythm by MIT-BIH: {is_n.sum():,}')
    print(f'Predictions: majority vote over {P.shape[0]} {args.mode} runs '
          f'(median agreement {100*np.median(agree[is_n]):.0f}%)\n')

    print('What the model predicted, against the rate MEASURED from MIT-BIH\'s '
          'own beat annotations:\n')
    print(f"  {'model says':<12}{'windows':>9}{'median bpm':>12}"
          f"{'<60 bpm':>10}{'>100 bpm':>10}{'RR cv':>9}{'ectopic':>9}")
    print('  ' + '-' * 71)
    for k, n in enumerate(names):
        m = is_n & (pred == k)
        if not m.any():
            continue
        print(f'  {n:<12}{m.sum():>9}{np.nanmedian(bpm[m]):>12.0f}'
              f'{100*np.nanmean(bpm[m] < BRADY):>9.0f}%'
              f'{100*np.nanmean(bpm[m] > TACHY):>9.0f}%'
              f'{np.nanmedian(cv[m]):>9.3f}{100*np.nanmedian(ect[m]):>8.0f}%')

    print('\n--- the test ---')
    verdicts = []
    for cls, lo, hi, rule in [('SB', -np.inf, BRADY, f'median rate < {BRADY:.0f}'),
                              ('ST', TACHY, np.inf, f'median rate > {TACHY:.0f}')]:
        if cls not in names:
            continue
        m = is_n & (pred == names.index(cls))
        if m.sum() < 20:
            print(f'  {cls}: only {m.sum()} windows, not enough to judge')
            continue
        med = float(np.nanmedian(bpm[m]))
        ok = lo < med <= hi if cls == 'SB' else lo <= med < hi
        verdicts.append(ok)
        print(f'  Windows MIT-BIH calls sinus rhythm and the model calls {cls}: '
              f'median measured rate {med:.0f} bpm  ({rule})  -> '
              f'{"MODEL RIGHT, LABEL COARSE" if ok else "MODEL WRONG"}')

    if 'SI' in names:
        m = is_n & (pred == names.index('SI'))
        if m.sum() >= 20:
            med = float(np.nanmedian(cv[m]))
            base = float(np.nanmedian(cv[is_n & (pred == sr)]))
            print(f'  Windows the model calls SI: RR cv {med:.3f} against '
                  f'{base:.3f} for those it calls SR  -> '
                  f'{"MODEL RIGHT" if med > base else "no separation"}')

    afib = names.index('AFIB/AFL')
    m = is_n & (pred == afib)
    if m.any():
        med_cv = float(np.nanmedian(cv[m]))
        med_ect = float(np.nanmedian(ect[m]))
        base_cv = float(np.nanmedian(cv[is_n & (pred == sr)]))
        print(f'  Windows the model calls AFIB: RR cv {med_cv:.3f} '
              f'(vs {base_cv:.3f} when it says SR), '
              f'{100*med_ect:.0f}% ectopic beats.')
        print('    Frequent ectopy produces an irregular RR sequence, which is '
              'the same cue\n    atrial fibrillation presents. This is the '
              'shared-cue failure seen on Georgia,\n    not a separate one.')

    # How much of the apparent error is label granularity rather than error?
    recoverable = is_n & np.isin(pred, [names.index(c) for c in ('SB', 'ST')
                                        if c in names])
    correct_by_rate = recoverable & (
        ((pred == names.index('SB')) & (bpm < BRADY)) |
        ((pred == names.index('ST')) & (bpm > TACHY)))
    print(f'\n  Of {is_n.sum():,} windows MIT-BIH calls sinus rhythm, the model '
          f'calls {recoverable.sum():,}\n  bradycardia or tachycardia, and '
          f'{correct_by_rate.sum():,} of those are confirmed by the measured '
          f'rate.')
    print(f'  That is {100*correct_by_rate.sum()/is_n.sum():.1f}% of the '
          f'apparent sinus-rhythm error that is not error at all.')

    strict = (pred == y).mean()
    lenient = ((pred == y) | correct_by_rate).mean()
    print(f'\n  Accuracy as scored              : {100*strict:.2f}%')
    print(f'  Accuracy crediting measured rate: {100*lenient:.2f}%')
    print('\n  The second number is NOT a result to report as accuracy -- it '
          'is scored\n  against a relabelling this project performed. It is '
          'reported to show how\n  much of the gap is label granularity.')


if __name__ == '__main__':
    main()
