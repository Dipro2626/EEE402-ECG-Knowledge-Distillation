"""
Data loading: Chapman / Ningbo CSVs, external parquet databases (Georgia,
CPSC), MIT-BIH at the rhythm level, and the pooled four-class training set.

CSV layout (verified on chapman_data_without_others.csv, 5087 x 5023):
    cols [0    : 5000]  raw single-lead signal, 500 Hz, 10 s
    cols [5000 : 5017]  17 HRV / quantitative features (bpm, mean_nn, ... hti)
    cols [5017 : 5023]  6 one-hot class columns

The notebook feeds columns [:5017] to the network -- signal AND features
concatenated along the time axis -- and splits them apart inside joint_model.
We keep that convention so the teacher reproduces exactly.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from paths import DATA_DIR

SIG_LEN = 5000
N_FEAT = 17

# Set by load_dataset('pooled'): which source database each row came from, so a
# per-source breakdown is still possible after the two are concatenated.
LAST_ORIGIN = None

# Three arms sharing ONE label space (the four classes Chapman and Ningbo have in
# common), so that the only variable between them is how many training records
# there are. 'ningbo' and 'chapman' on their own carry different label spaces and
# are therefore not comparable to each other or to the pool.
POOLED_SETS = {
    'chapman4': ('chapman',),            #  4,697 records
    'ningbo4': ('ningbo',),              # 15,164 records
    'pooled': ('chapman', 'ningbo'),     # 19,861 records
}

# Order matters: determine_major_class() returns the FIRST match, so a record
# flagged both 'Sinus Rhythm' and 'Sinus Irregularity' is labelled Sinus Rhythm.
# (In practice every row in the *_without_others files has exactly one flag.)
MAJOR_CLASSES = [
    'Sinus Rhythm', 'Sinus Bradycardia', 'Sinus Tachycardia', 'Sinus Irregularity',
    'Atrial Fibrillation_Atrial Flutter',
    'Supraventricular Tachycardia_Atrial Tachycardia',
    'Others',
]

SHORT_NAME = {
    'Sinus Rhythm': 'SR',
    'Sinus Bradycardia': 'SB',
    'Sinus Tachycardia': 'ST',
    'Sinus Irregularity': 'SI',
    'Atrial Fibrillation_Atrial Flutter': 'AFIB/AFL',
    'Supraventricular Tachycardia_Atrial Tachycardia': 'SVT/AT',
    'Others': 'OTHER',
}

# The number of classes is NOT a constant across these files:
#
#   chapman  5 classes -- the 'Sinus Irregularity' column is entirely zero
#   ningbo   6 classes -- SI is real (1232 records), and SVT/AT has only 15
#   ptb      2 classes -- 6426 SR + 45 AFIB, which is why it is unusable here
#
# and the *_with_others_ST_excluded files carry SEVEN class columns, not six.
# The notebook hardcodes a 5-entry mapping and slices columns[-6:], which works
# for chapman/without_others and silently misbehaves elsewhere. We derive both
# from the file instead.


def csv_path(dataset='chapman', variant='without_others'):
    return os.path.join(DATA_DIR, f'{dataset}_data_{variant}.csv')


def load_dataset(dataset='chapman', variant='without_others', verbose=True,
                 drop_empty=True, min_count=0):
    """
    Returns
    -------
    X       : (N, 5017) float32 -- signal then the 17 features, as the notebook does
    y       : (N,)      int64
    names   : list[str] -- short class names, index == label id

    Set drop_empty=False to keep all-zero columns as real (empty) classes, e.g.
    to force chapman onto the same 6-class head as ningbo.

    dataset='pooled' dispatches to load_pooled (Chapman + Ningbo on their four
    shared classes). Dispatching here rather than in each caller means
    teacher_v2.py, train_student.py, export_teacher_v2.py, analyze.py and
    make_table.py all accept --dataset pooled without modification.
    """
    if dataset in POOLED_SETS:
        global LAST_ORIGIN
        X, y, names, LAST_ORIGIN = load_pooled(POOLED_SETS[dataset],
                                               variant=variant, verbose=verbose)
        if min_count:
            print(f'  NOTE: --min_class {min_count} ignored for {dataset}; '
                  f'the shared four classes all exceed 500 records')
        return X, y, names

    path = csv_path(dataset, variant)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found.\n'
            'Download from https://www.kaggle.com/datasets/tushartalukder11/chap-ning-ptb '
            'and place the CSVs in the data/ folder (see DATA.md).')

    df = pd.read_csv(path)

    # Class columns are whatever follows the 5000 signal + 17 feature columns.
    # This is 6 for the *_without_others files and 7 for *_with_others_*.
    class_cols = [c for c in df.columns[SIG_LEN + N_FEAT:]]
    unexpected = [c for c in class_cols if c not in MAJOR_CLASSES]
    if unexpected:
        raise ValueError(f'unrecognised class columns: {unexpected}')

    multi = int((df[class_cols].sum(axis=1) > 1).sum())
    none_set = int((df[class_cols].sum(axis=1) == 0).sum())

    order = [c for c in MAJOR_CLASSES if c in class_cols]
    if drop_empty:
        order = [c for c in order if df[c].sum() > 0]

    # A class with a handful of records is not a class the model can learn or
    # that macro-F1 can meaningfully average over -- Ningbo's SVT/AT has 15
    # records against 16,411, so a five-fold split leaves about three per
    # validation fold. Including it drags the unweighted mean down by roughly
    # the same amount whatever the model does. Records of a dropped class are
    # removed rather than relabelled, so no record is silently reassigned.
    dropped_small = {}
    if min_count > 0:
        keep_order = []
        for c in order:
            n = int(df[c].sum())
            if n >= min_count:
                keep_order.append(c)
            else:
                dropped_small[c] = n
        order = keep_order
    if none_set and 'Others' not in order:
        raise ValueError(
            f'{none_set} rows have no class flag set and there is no Others '
            f'column. Columns present: {class_cols}')

    mapping = {c: i for i, c in enumerate(order)}

    def major_class(row):
        for m in order:
            if row[m] == 1:
                return m
        return 'Others'

    labels = df.apply(major_class, axis=1)

    if dropped_small:
        keep_rows = labels.isin(order).to_numpy()
        df = df[keep_rows]
        labels = labels[keep_rows]

    unknown = sorted(set(labels) - set(mapping))
    if unknown:
        raise ValueError(f'unmapped labels present: {unknown}')

    y = np.array([mapping[l] for l in labels], dtype=np.int64)
    X = df.iloc[:, :SIG_LEN + N_FEAT].to_numpy(dtype=np.float32)
    names = [SHORT_NAME[c] for c in order]

    if verbose:
        print(f'{dataset}/{variant}: X {X.shape}  y {y.shape}  '
              f'{len(names)} classes')
        counts = np.bincount(y, minlength=len(names))
        for i, name in enumerate(names):
            print(f'  {i} {name:<10} {counts[i]:>6}  '
                  f'({100 * counts[i] / len(y):5.2f}%)')
        nz = counts[counts > 0]
        print(f'  imbalance ratio (max/min): {nz.max() / nz.min():.1f}:1')
        if multi:
            print(f'  NOTE: {multi} rows carry more than one class flag; the '
                  f'first match in MAJOR_CLASSES order wins')
        if dropped_small:
            print(f'  dropped classes below {min_count} records: '
                  f'{dropped_small}  ({len(labels)} records retained)')
        empty = [c for c in class_cols
                 if c not in order and c not in dropped_small]
        if empty:
            print(f'  dropped empty columns: {empty}')

    return X, y, names


# --------------------------------------------------------------------------- #
# External parquet databases (PhysioNet/CinC 2021 collection)
# --------------------------------------------------------------------------- #
PARQUET = {
    'georgia': os.path.join('GEORGIA', 'tweleve_lead_georgia_wo_filters.parquet'),
    'cpsc': os.path.join('CPSC18', 'cpsc_data.parquet'),
    'cpsc_nofilter': os.path.join('CPSC18', 'cpsc_data_wo_filter.parquet'),
}

# Their label columns, mapped onto the Chapman names the student predicts.
# Georgia has no Atrial Flutter column, so its 'Atrial Fibrillation' is mapped
# to Chapman's merged AFIB/AFL class -- a slight mismatch (theirs excludes
# flutter, ours includes it) that is recorded rather than hidden.
PARQUET_LABEL_MAP = {
    'Sinus Rhythm': 'Sinus Rhythm',
    'Sinus Bradycardia': 'Sinus Bradycardia',
    'Sinus Tachycardia': 'Sinus Tachycardia',
    'Sinus Irregularity': 'Sinus Irregularity',
    'Atrial Fibrillation': 'Atrial Fibrillation_Atrial Flutter',
}
# Georgia carries Sinus Irregularity (352 records), which Chapman does not, so a
# Chapman-trained model has no output unit for it and it is dropped there. A
# Ningbo-trained model does have one, so the same target yields five comparable
# classes instead of four. The mapping is shared; which entries apply is decided
# per source-target pair by the intersection of label spaces.


def load_parquet_dataset(dataset, source_names, lead=1, verbose=True):
    """
    Load an external 12-lead parquet database and align it to the student's
    label space.

    The signal is stored in blocks: signal_1..5000 is lead 1, signal_5001..10000
    is lead 2, and so on (verified by lag-1 autocorrelation, 0.993 for the block
    reading against 0.88 for an interleaved reading). Only the requested lead is
    read, so a 2 GB file costs about 160 MB of memory.

    Two filters are applied, both forced by the data:

    1. These databases are MULTI-LABEL -- 44% of Georgia rows carry more than one
       diagnosis, whereas Chapman is single-label. A record is kept only if
       exactly one of the student's classes applies to it. Records carrying two
       of them are contradictory under our label space and are dropped rather
       than resolved by an arbitrary priority rule.
    2. Records with none of the student's classes are dropped: 47% of Georgia
       carries only morphology or conduction findings (T wave change, LVH, ...),
       which the student has no output unit for.

    Amplitude differs hugely from Chapman (0-1 here, 0-4096 there), but the
    student z-scores each record, and z-scoring cancels any per-record linear
    rescaling exactly, so no correction is applied.
    """
    if dataset not in PARQUET:
        raise ValueError(f'unknown parquet dataset {dataset}')
    path = os.path.join(DATA_DIR, PARQUET[dataset])
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    import pyarrow.parquet as pq
    available = set(pq.ParquetFile(path).schema_arrow.names)

    # load_dataset returns SHORT names ('SR', 'SB', ...), while PARQUET_LABEL_MAP
    # is written in the full Chapman column names, so convert before matching.
    usable = {}
    for src, long_dst in PARQUET_LABEL_MAP.items():
        short = SHORT_NAME.get(long_dst, long_dst)
        if src in available and short in source_names:
            usable[src] = short
    if not usable:
        raise ValueError(
            f'{dataset} shares no classes with {source_names}.\n'
            f'  parquet label columns available: '
            f'{sorted(c for c in PARQUET_LABEL_MAP if c in available)}\n'
            f'  mapped to: '
            f'{[SHORT_NAME.get(v, v) for v in PARQUET_LABEL_MAP.values()]}')

    lab = pd.read_parquet(path, columns=sorted(usable))
    flags = lab[sorted(usable)].to_numpy(dtype=np.int64)
    n_flags = flags.sum(axis=1)

    keep = n_flags == 1
    idx = flags[keep].argmax(axis=1)
    cols = sorted(usable)
    y = np.array([source_names.index(usable[cols[i]]) for i in idx],
                 dtype=np.int64)

    start = (lead - 1) * SIG_LEN + 1
    sig_cols = [f'signal_{i}' for i in range(start, start + SIG_LEN)]
    X = pd.read_parquet(path, columns=sig_cols).to_numpy(dtype=np.float32)[keep]

    if verbose:
        print(f'{dataset}: {len(lab)} records, lead {lead}')
        print(f'  classes shared with the student: '
              f'{[usable[c] for c in cols]}')
        missing = [c for c in source_names
                   if c not in [usable[k] for k in cols]]
        if missing:
            print(f'  absent from {dataset}: {missing}')
        print(f'  dropped, no student class    : {int((n_flags == 0).sum())}')
        print(f'  dropped, two student classes : {int((n_flags > 1).sum())}')
        print(f'  kept                         : {int(keep.sum())}')
        counts = np.bincount(y, minlength=len(source_names))
        for i, n in enumerate(source_names):
            print(f'    {n:<10}{counts[i]:>6}')

    return X, y


# --------------------------------------------------------------------------- #
# MIT-BIH Arrhythmia, read at the RHYTHM level
# --------------------------------------------------------------------------- #
# MIT-BIH is normally used as a BEAT database (N/S/V/F/Q). That is a different
# task from ours and its label space does not intersect ours at all. But the
# .atr files also carry rhythm-change annotations in aux_note, and a few of those
# do map onto our classes. Counting whole 10-second windows inside each rhythm
# episode over all 48 records gives:
#
#     (N     6106 windows  42 patients  -> Sinus Rhythm
#     (AFIB   752 windows   8 patients  -> AFIB/AFL
#     (AFL     60 windows   3 patients  -> AFIB/AFL   (subset of the above 8)
#     (SBR    180 windows   1 patient   -> Sinus Bradycardia
#     (SVTA    12 windows   3 patients  -> SVT/AT
#     -- no sinus tachycardia or sinus irregularity annotation exists at all --
#
# So MIT-BIH can only ever be a TWO-class external check for this model, and the
# effective sample size for atrial fibrillation is eight patients, not 812
# windows: consecutive windows from one 30-minute recording are near-duplicates.
# That is why it is an evaluation target here and never a training set.
MITBIH_FS = 360
MITBIH_WIN = 10 * MITBIH_FS          # 10 s, matching our 5000 samples at 500 Hz

MITBIH_RHYTHM_MAP = {
    '(N': 'Sinus Rhythm',
    '(SBR': 'Sinus Bradycardia',
    '(AFIB': 'Atrial Fibrillation_Atrial Flutter',
    '(AFL': 'Atrial Fibrillation_Atrial Flutter',
    '(SVTA': 'Supraventricular Tachycardia_Atrial Tachycardia',
}

# A class carried by one or two patients cannot be evaluated: its F1 measures one
# recording, not a class. Declared here rather than discovered from the results.
MITBIH_MIN_PATIENTS = 3
MITBIH_MIN_WINDOWS = 100

# Annotation symbols that mark a heartbeat (as opposed to a rhythm change, a
# signal-quality note or a comment). 'N' alone is a sinus-conducted normal beat;
# the rest are ectopic, aberrated, paced or unclassifiable.
MITBIH_BEAT_SYMBOLS = set('NLRejAaJSVEFP/fQ')


def load_mitbih_rhythm(source_names, lead='MLII', verbose=True, cache=True,
                       min_patients=MITBIH_MIN_PATIENTS,
                       min_windows=MITBIH_MIN_WINDOWS):
    """
    Returns (X, y, patient_ids, dropped, props) with X as (N, 5000) float32 at
    500 Hz, y indexed into source_names, patient_ids so that scores can be
    aggregated per recording rather than per window, and props as
    (N, 3) = measured heart rate in bpm, RR coefficient of variation, and the
    fraction of beats that are not sinus-conducted. props comes from MIT-BIH's
    own expert beat annotations, so it is measured, not detected.

    Choices forced by the data, all of which are mismatches worth stating:

    * 46 of the 48 records carry MLII; records 102 and 104 do not and are
      excluded rather than silently substituted with a different lead. MLII is
      approximately lead II, whereas the training data is read at lead index 1,
      so a lead mismatch is present and cannot be separated from domain shift.
    * 360 Hz is resampled to 500 Hz with a polyphase filter at exactly 25/18,
      so 3600 samples map to 5000 with no interpolation error.
    * Windows never straddle a rhythm change: each is cut entirely inside one
      annotated episode.
    """
    import glob as _glob
    from paths import MITBIH_DIR, CACHE_DIR

    key = os.path.join(CACHE_DIR, f'mitbih_rhythm_{lead}_v2.npz')
    if cache and os.path.exists(key):
        z = np.load(key, allow_pickle=True)
        sig, lab, pid, props = z['sig'], z['lab'], z['pid'], z['props']
    else:
        import wfdb
        from scipy.signal import resample_poly

        recs = sorted({os.path.splitext(os.path.basename(p))[0]
                       for p in _glob.glob(os.path.join(MITBIH_DIR, '*.hea'))})
        if not recs:
            raise FileNotFoundError(
                f'no WFDB headers in {MITBIH_DIR}. Set EXGNET_MITBIH_DIR.')

        sig_l, lab_l, pid_l, prop_l, no_lead = [], [], [], [], []
        for r in recs:
            base = os.path.join(MITBIH_DIR, r)
            hdr = wfdb.rdheader(base)
            if lead not in hdr.sig_name:
                no_lead.append(r)
                continue
            ch = hdr.sig_name.index(lead)
            rec = wfdb.rdrecord(base, channels=[ch])
            x = rec.p_signal[:, 0].astype(np.float32)

            ann = wfdb.rdann(base, 'atr')
            marks = [(int(s), (a or '').strip('\x00').strip())
                     for s, a in zip(ann.sample, ann.aux_note)
                     if (a or '').startswith('(')]
            # MIT-BIH's expert BEAT annotations, used to MEASURE each window's
            # rate and RR irregularity rather than detect them. This is what
            # makes it possible to ask whether a prediction the label calls
            # wrong is in fact right.
            bs = np.array([int(s) for s, sym in zip(ann.sample, ann.symbol)
                           if sym in MITBIH_BEAT_SYMBOLS])
            bsym = np.array([sym for sym in ann.symbol
                             if sym in MITBIH_BEAT_SYMBOLS])
            for i, (s, tag) in enumerate(marks):
                if tag not in MITBIH_RHYTHM_MAP:
                    continue
                e = marks[i + 1][0] if i + 1 < len(marks) else len(x)
                for w in range(s, e - MITBIH_WIN + 1, MITBIH_WIN):
                    seg = x[w:w + MITBIH_WIN]
                    if not np.isfinite(seg).all():
                        continue
                    m = (bs >= w) & (bs < w + MITBIH_WIN)
                    if m.sum() >= 4:
                        rr = np.diff(bs[m]) / MITBIH_FS
                        prop = (60.0 * m.sum() / 10.0,          # bpm
                                float(rr.std() / rr.mean()),    # RR variability
                                float((bsym[m] != 'N').mean()))  # ectopic share
                    else:
                        prop = (np.nan, np.nan, np.nan)
                    # 360 -> 500 Hz is exactly 25/18; 3600 * 25 / 18 == 5000.
                    sig_l.append(resample_poly(seg, 25, 18).astype(np.float32))
                    lab_l.append(MITBIH_RHYTHM_MAP[tag])
                    pid_l.append(r)
                    prop_l.append(prop)

        sig = np.asarray(sig_l, dtype=np.float32)
        lab = np.asarray(lab_l)
        pid = np.asarray(pid_l)
        props = np.asarray(prop_l, dtype=np.float32)
        if verbose and no_lead:
            print(f'  excluded, no {lead} channel: {no_lead}')
        if cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.savez_compressed(key, sig=sig, lab=lab, pid=pid, props=props)

    assert sig.shape[1] == SIG_LEN, sig.shape

    # Restrict to classes the student can actually predict, then to classes with
    # enough patients and windows to be measurable.
    short = np.array([SHORT_NAME[l] for l in lab])
    stats = {}
    for c in sorted(set(short)):
        m = short == c
        stats[c] = (int(m.sum()), len(set(pid[m])))

    keep_cls = [c for c in source_names
                if c in stats
                and stats[c][0] >= min_windows
                and stats[c][1] >= min_patients]
    dropped = {c: stats[c] for c in stats if c not in keep_cls}

    mask = np.isin(short, keep_cls)
    y = np.array([source_names.index(c) for c in short[mask]], dtype=np.int64)
    X, P, W = sig[mask], pid[mask], props[mask]

    if verbose:
        print(f'mitbih (rhythm level, lead {lead}): '
              f'{len(sig):,} 10-s windows before filtering')
        print(f"  {'class':<10}{'windows':>9}{'patients':>10}   status")
        for c, (n, np_) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
            why = ('kept' if c in keep_cls else
                   'DROPPED, not a student class' if c not in source_names else
                   f'DROPPED, {np_} patient(s) < {min_patients}'
                   if np_ < min_patients else
                   f'DROPPED, {n} windows < {min_windows}')
            print(f'  {c:<10}{n:>9}{np_:>10}   {why}')
        print(f'  evaluated on {len(y):,} windows from '
              f'{len(set(P))} recordings, {len(keep_cls)} classes')
        absent = [c for c in source_names if c not in stats]
        if absent:
            print(f'  absent from MIT-BIH entirely: {absent}')

    return X, y, P, dropped, W


# --------------------------------------------------------------------------- #
# Chapman + Ningbo pooled, on the classes they share
# --------------------------------------------------------------------------- #
# The two databases do not share a label space: Chapman has SVT/AT and no Sinus
# Irregularity, Ningbo the reverse (and only 15 SVT/AT records). Pooling on the
# union would give a 6-class model comparable to neither single-database model,
# which is the wrong experiment. Pooling on the four SHARED classes keeps the
# label space identical across all three arms, so the only thing that changes is
# how many training records there are -- which is the claim under test.
SHARED_CLASSES = ['SR', 'SB', 'ST', 'AFIB/AFL']


def load_pooled(datasets=('chapman', 'ningbo'), variant='without_others',
                classes=SHARED_CLASSES, verbose=True):
    """
    Returns (X, y, names, origin) where origin records which database each row
    came from, so per-source breakdowns stay possible after pooling.
    """
    Xs, ys, os_ = [], [], []
    for ds in datasets:
        Xd, yd, nd = load_dataset(ds, variant, verbose=False)
        keep = [i for i, n in enumerate(nd) if n in classes]
        missing = [c for c in classes if c not in nd]
        if missing:
            raise ValueError(f'{ds} lacks shared classes {missing}')
        m = np.isin(yd, keep)
        remap = {i: classes.index(nd[i]) for i in keep}
        Xs.append(Xd[m])
        ys.append(np.array([remap[v] for v in yd[m]], dtype=np.int64))
        os_.append(np.full(int(m.sum()), ds))
        if verbose:
            c = np.bincount(ys[-1], minlength=len(classes))
            print(f'  {ds:<9}{int(m.sum()):>7} records  '
                  + '  '.join(f'{n}={c[i]}' for i, n in enumerate(classes))
                  + f'   (dropped {int((~m).sum())} not in the shared set)')

    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys)
    origin = np.concatenate(os_)

    if verbose:
        counts = np.bincount(y, minlength=len(classes))
        print(f'pooled {"+".join(datasets)}: X {X.shape}  '
              f'{len(classes)} classes')
        for i, n in enumerate(classes):
            print(f'  {i} {n:<10}{counts[i]:>7}  '
                  f'({100 * counts[i] / len(y):5.2f}%)')
        print(f'  imbalance ratio (max/min): '
              f'{counts.max() / counts.min():.1f}:1')

    return X, y, list(classes), origin


def get_folds(X, y, n_splits=5, seed=42):
    """
    Reproduces the notebook exactly: KFold(n_splits=5, shuffle=True, random_state=42)
    over ROWS.

    In Chapman and Ningbo each row is one 10-second recording from one distinct
    patient, so a row-level split is already patient-disjoint -- unlike MIT-BIH,
    where one patient contributes ~2000 beats. verify_split.py checks this claim
    rather than assuming it.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(X))


def split_signal_features(X):
    """(N, 5017) -> (N, 5000) signal, (N, 17) features."""
    return X[:, :SIG_LEN], X[:, SIG_LEN:]


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='chapman',
                   choices=['chapman', 'ningbo', 'ptb', 'chapman4', 'ningbo4',
                            'pooled', 'all'])
    p.add_argument('--variant', default='without_others')
    a = p.parse_args()

    datasets = ['chapman', 'ningbo', 'ptb'] if a.dataset == 'all' else [a.dataset]
    for ds in datasets:
        X, y, names = load_dataset(ds, a.variant)
        for k, (tr, va) in enumerate(get_folds(X, y)):
            tr_c = np.bincount(y[tr], minlength=len(names))
            va_c = np.bincount(y[va], minlength=len(names))
            empty = [names[i] for i in range(len(names)) if tr_c[i] == 0 or va_c[i] == 0]
            print(f'  fold {k}: train {len(tr)}  val {len(va)}  '
                  f'overlap {len(set(tr) & set(va))}'
                  + (f'  EMPTY IN A SPLIT: {empty}' if empty else ''))
        print()
