"""
Train the compact student on Chapman/Ningbo, optionally distilling from the
cached, frozen Teacher v2.

    python train_student.py --mode ce
    python train_student.py --mode kd

Teacher outputs come from cache/teacherv2_<tag>_fold<k>.npz (see
export_teacher_v2.py), so the teacher is never re-run during student training.
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from data import load_dataset, get_folds
from student import ECGStudent, count_parameters
from distill import RkdDistance, RKdAngle, LogitKD, FitNet, MODES
from paths import CACHE_DIR, CKPT_DIR, OUT_DIR


def set_seed(seed, deterministic=True):
    """
    Seeding the RNGs is not enough on its own.

    cuDNN picks convolution and RNN kernels by autotuned benchmark, and several
    of them accumulate in a non-deterministic order. Two runs of an identical
    configuration at the same seed then differ -- we measured 0.41 macro-F1
    points between two `--mode kd --seed 0` runs, which is larger than the
    RKD+KD vs CE effect we are trying to resolve (0.15). Under that much
    same-configuration noise, a mode comparison at one seed measures nothing,
    and even paired multi-seed testing is compromised because the pairing
    assumes the only difference between two runs is the loss.

    CUBLAS_WORKSPACE_CONFIG must be set before the first CUDA call, so this
    runs early. use_deterministic_algorithms is best-effort: if a layer has no
    deterministic kernel it raises, and we fall back with a warning rather than
    crash the grid.
    """
    import random
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if not deterministic:
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f'WARNING: could not enable deterministic algorithms ({e}); '
              f'run-to-run variation will remain.')


def normalise(sig, ref):
    """Per-record z-score. Statistics come from the record itself, so there is
    no train/val statistic leakage."""
    mu = ref.mean(axis=1, keepdims=True)
    sd = ref.std(axis=1, keepdims=True) + 1e-8
    return (sig - mu) / sd


LOSS_NAMES = ['ce', 'kd', 'rkd_d', 'rkd_a', 'fitnet']


def auto_balance(model, loader, device, mode, cls_w, kd, rkd_d, rkd_a, fitnet,
                 needs_teacher, n_batches=10, ce_target=0.5):
    """
    Choose loss weights so cross-entropy is a fixed share of the total.

    Why this is needed: the hand-set weights (25 for distance, 50 for angle,
    from the RKD paper) assume loss terms of a particular magnitude. Those
    magnitudes depend on the teacher's embedding width and on the dataset, so
    weights carried over from Study 1 (MIT-BIH, 400-d teacher) put
    cross-entropy at 5-13% of the total here -- the labels were being drowned,
    and every distilled run tripped the warning below.

    A comparison run under weights the code itself flags as unbalanced cannot
    support either a positive or a null claim. So instead of guessing new
    constants, measure each term on a few untrained batches and scale it to a
    declared target share. Every mode then gets the same 50% cross-entropy,
    which is what makes the modes comparable to each other.

    The relative weighting *among* the distillation terms is preserved from
    MODES, so the RKD paper's 1:2 distance:angle ratio still holds.
    """
    base = MODES[mode]
    sums, n = np.zeros(5), 0

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            x, yb = batch[0].to(device), batch[1].to(device)
            logits, emb = model(x)
            vals = [float(F.cross_entropy(logits, yb, weight=cls_w)), 0., 0., 0., 0.]
            if needs_teacher:
                tl, te = batch[2].to(device), batch[3].to(device)
                vals[1] = float(kd(logits, tl))
                vals[2] = float(rkd_d(emb, te))
                vals[3] = float(rkd_a(emb, te))
                if fitnet is not None:
                    vals[4] = float(fitnet(emb, te))
            sums += vals
            n += 1
    model.train()

    mean = sums / max(n, 1)
    distill = [i for i in range(1, 5) if base[i] > 0]
    if not distill:
        return base, mean

    # Budget the non-CE half by FAMILY, not by the raw MODES numbers.
    #
    # MODES holds (1, 25, 50) for kd, rkd_d, rkd_a. Those are magnitude
    # corrections from the RKD paper, not statements about relative importance.
    # Splitting the budget in proportion to them would hand KD 1/76 of it --
    # which is what happened on the first attempt, leaving kd at 1% of the loss
    # and making 'rkd_kd' indistinguishable from 'rkd'. The stated comparison
    # (RKD+KD vs logit-KD) would then not have been tested at all.
    #
    # So: each active family gets an equal share of the non-CE budget, and only
    # the RKD paper's internal 1:2 distance:angle ratio is preserved.
    FAMILY = {1: 'kd', 2: 'rkd', 3: 'rkd', 4: 'fitnet'}
    families = sorted({FAMILY[i] for i in distill})
    per_family = (1.0 - ce_target) / len(families)

    target = np.zeros(5)
    target[0] = ce_target
    for fam in families:
        members = [i for i in distill if FAMILY[i] == fam]
        tot = sum(base[i] for i in members)
        for i in members:
            target[i] = per_family * base[i] / tot

    w = np.zeros(5)
    for i in [0] + distill:
        w[i] = target[i] / mean[i] if mean[i] > 1e-12 else 0.0
    if w[0] > 0:
        w = w / w[0]                      # report weights relative to CE = 1

    return tuple(float(v) for v in w), mean


def evaluate(model, loader, device, class_names):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            logits, _ = model(x)
            preds.append(logits.argmax(1).cpu())
            trues.append(y)
    p = torch.cat(preds).numpy()
    t = torch.cat(trues).numpy()
    labels = list(range(len(class_names)))
    cm = confusion_matrix(t, p, labels=labels)

    # Per-class accuracy is recall here (correct / true count). Reporting it as
    # "accuracy" per class is common but ambiguous, so both are stored.
    support = cm.sum(axis=1)
    recall = np.divide(np.diag(cm), support,
                       out=np.zeros(len(cm)), where=support > 0)

    return {
        'accuracy': float((p == t).mean()),
        'macro_f1': float(f1_score(t, p, average='macro')),
        'per_class': classification_report(
            t, p, labels=labels, target_names=class_names,
            output_dict=True, zero_division=0),
        'per_class_f1': [float(v) for v in
                         f1_score(t, p, average=None, labels=labels,
                                  zero_division=0)],
        'per_class_recall': [float(v) for v in recall],
        'support': [int(v) for v in support],
        'confusion': cm.tolist(),
    }, p, t


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='ce', choices=list(MODES))
    p.add_argument('--dataset', default='chapman')
    p.add_argument('--variant', default='without_others')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-3)
    p.add_argument('--width', type=float, default=1.0)
    p.add_argument('--embed_dim', type=int, default=64)
    p.add_argument('--no_gru', action='store_true')
    p.add_argument('--T', type=float, default=4.0)
    p.add_argument('--no_balance', dest='balance', action='store_false',
                   help='use the raw MODES weights instead of auto-balancing')
    p.add_argument('--ce_target', type=float, default=0.5,
                   help='share of the total loss that cross-entropy should carry')
    p.add_argument('--min_class', type=int, default=0,
                   help='drop classes below this many records; must match the '
                        'value used when the teacher was trained')
    p.add_argument('--nondeterministic', action='store_true',
                   help='allow fast non-deterministic cuDNN kernels (faster, '
                        'but identical configs stop reproducing)')
    p.add_argument('--teacher_emb', default='joint', choices=['joint', 'signal'])
    p.add_argument('--teacher', default='v2',
                   help="teacher variant tag. 'v2' = our 256-d "
                        "teacher; 'v2n' = same trunk projected to 32-d "
                        "(Amendment 4 control arm). Any tag works as long as "
                        "the matching cache exists.")
    p.add_argument('--out_dir', default='results')
    p.add_argument('--save_model', default=None,
                   help='explicit checkpoint path; otherwise auto-named under '
                        'checkpoints/ so cross-dataset evaluation can reuse the '
                        'trained student without retraining')
    p.add_argument('--no_save', action='store_true',
                   help='skip checkpointing entirely')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args(argv)

    device = torch.device(args.device)
    tag = f'{args.dataset}_{args.variant}'
    out_dir = os.path.join(OUT_DIR, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Runs distilled from different teachers must not collide. Without this the
    # teacher-v2 grid would silently overwrite the completed teacher-v1 results
    # and the v1-vs-v2 comparison -- the entire point of Amendment 3 -- would be
    # destroyed with no error raised. CE needs no teacher, so it is shared.
    vtag = '' if (args.teacher == 'v1' or args.mode == 'ce') else f'_{args.teacher}'

    # Architecture variants must not collide either. Omitting this overwrote a
    # completed grid once: the --no_gru runs landed on the GRU runs' filenames
    # and silently replaced them, leaving one fold of a five-fold analysis using
    # a different architecture from the other four.
    atag = ''
    if args.no_gru:
        atag += '_nogru'
    if args.width != 1.0:
        atag += f'_w{args.width:g}'
    if args.embed_dim != 64:
        atag += f'_e{args.embed_dim}'

    run_name = (f'student_{args.mode}{vtag}{atag}_{tag}'
                f'_fold{args.fold}_seed{args.seed}')

    # Checkpoint by default. A student is ~220 kB, so the whole 100-run grid is
    # about 22 MB -- cheap insurance against having to retrain for two hours
    # every time a new evaluation set comes along.
    if args.save_model is None and not args.no_save:
        os.makedirs(CKPT_DIR, exist_ok=True)
        args.save_model = os.path.join(CKPT_DIR, run_name + '.pt')

    # ---- data -------------------------------------------------------------
    X, y, class_names = load_dataset(args.dataset, args.variant, verbose=False,
                                     min_count=args.min_class)
    n_classes = len(class_names)
    if args.min_class:
        tag += f'_min{args.min_class}'
    sig = X[:, :5000]
    sig = normalise(sig, sig)
    tr, va = get_folds(X, y)[args.fold]

    needs_teacher = args.mode != 'ce'
    t_logits = t_emb = None
    if needs_teacher:
        prefix = 'teacherv2'
        cache = os.path.join(CACHE_DIR, f'{prefix}_{tag}_fold{args.fold}.npz')
        if not os.path.exists(cache):
            script = 'export_teacher_v2.py'
            raise SystemExit(f'{cache} not found -- run {script} '
                             f'--fold {args.fold} first.')
        d = np.load(cache)
        if not np.array_equal(d['idx_train'], tr):
            raise SystemExit('cached fold indices do not match the current split; '
                             're-export the teacher for this fold.')
        t_logits = d[f'logits_{args.teacher_emb}']
        t_emb = d[f'emb_{args.teacher_emb}']
        print(f'teacher {args.teacher}: {args.teacher_emb} branch, '
              f'{t_emb.shape[1]}-d embedding, student {args.embed_dim}-d')
        if t_emb.shape[1] < args.embed_dim:
            print(f'  NOTE: the teacher embedding is NARROWER than the '
                  f'student\'s. Relational distillation has less structure to '
                  f'transfer than the student can represent — this is the '
                  f'bottleneck Amendment 3 tests.')

    def make_loader(idx, shuffle):
        tensors = [torch.from_numpy(sig[idx]).float(), torch.from_numpy(y[idx])]
        if needs_teacher:
            tensors += [torch.from_numpy(t_logits[idx]).float(),
                        torch.from_numpy(t_emb[idx]).float()]
        return DataLoader(TensorDataset(*tensors), batch_size=args.batch,
                          shuffle=shuffle, drop_last=shuffle)

    # Seed AFTER data prep but BEFORE model construction, so that runs sharing a
    # seed start from identical weights and the comparison between modes is paired.
    set_seed(args.seed, deterministic=not args.nondeterministic)
    train_loader = make_loader(tr, True)
    val_loader = make_loader(va, False)

    # ---- model ------------------------------------------------------------
    model = ECGStudent(num_classes=n_classes, width=args.width,
                       embed_dim=args.embed_dim,
                       use_gru=not args.no_gru).to(device)
    n_par = count_parameters(model)

    rkd_d, rkd_a = RkdDistance(), RKdAngle()
    kd = LogitKD(T=args.T)
    fitnet = (FitNet(args.embed_dim, t_emb.shape[1]).to(device)
              if MODES[args.mode][4] else None)

    counts = np.bincount(y[tr], minlength=n_classes)
    cls_w = torch.tensor(np.log(counts.sum() / (counts + 1e-8)),
                         dtype=torch.float32, device=device)
    cls_w = (cls_w / cls_w.mean()).clamp(min=0.2)

    params = list(model.parameters()) + (list(fitnet.parameters()) if fitnet else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f'\nmode {args.mode}  seed {args.seed}  fold {args.fold}  '
          f'student params {n_par:,}')
    print(f'class weights: {[round(float(v), 2) for v in cls_w]}')

    weights = MODES[args.mode]
    if args.balance and needs_teacher:
        weights, measured = auto_balance(
            model, train_loader, device, args.mode, cls_w,
            kd, rkd_d, rkd_a, fitnet, needs_teacher, ce_target=args.ce_target)
        print(f'auto-balanced weights (CE target {args.ce_target:.0%}):')
        for n, w0, w1, m in zip(LOSS_NAMES, MODES[args.mode], weights, measured):
            if w0:
                print(f'    {n:<7} raw {m:>9.4f}   weight {w0:>5.1f} -> {w1:>8.3f}')
    w_ce, w_kd, w_rd, w_ra, w_fn = weights
    print()

    best = {'macro_f1': -1.0}
    best_epoch = -1
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        sums = np.zeros(5)
        for batch in train_loader:
            x, yb = batch[0].to(device), batch[1].to(device)
            logits, emb = model(x)

            l_ce = F.cross_entropy(logits, yb, weight=cls_w)
            l_kd = l_rd = l_ra = l_fn = torch.zeros((), device=device)

            if needs_teacher:
                tl, te = batch[2].to(device), batch[3].to(device)
                if w_kd:
                    l_kd = kd(logits, tl)
                if w_rd:
                    l_rd = rkd_d(emb, te)
                if w_ra:
                    l_ra = rkd_a(emb, te)
                if w_fn:
                    l_fn = fitnet(emb, te)

            loss = (w_ce * l_ce + w_kd * l_kd + w_rd * l_rd
                    + w_ra * l_ra + w_fn * l_fn)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

            sums += [float(w_ce * l_ce), float(w_kd * l_kd), float(w_rd * l_rd),
                     float(w_ra * l_ra), float(w_fn * l_fn)]

        sched.step()

        if ep == 0:
            total = sums.sum() or 1.0
            share = 100 * sums / total
            print('epoch-1 loss split  ce %.0f%%  kd %.0f%%  rkd_d %.0f%%  '
                  'rkd_a %.0f%%  fitnet %.0f%%' % tuple(share))
            if share[0] < 20 and args.mode != 'ce':
                print('WARNING: cross-entropy is under 20% of the loss. The '
                      'distillation terms are drowning the labels -- lower the '
                      'RKD weights before trusting this run.')

        m, _, _ = evaluate(model, val_loader, device, class_names)
        if m['macro_f1'] > best['macro_f1']:
            best, best_epoch = m, ep + 1
            if args.save_model:
                torch.save({'model': model.state_dict(), 'args': vars(args)},
                           args.save_model)

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"ep {ep + 1:>3}  acc {m['accuracy']:.4f}  "
                  f"macro-F1 {m['macro_f1']:.4f}", flush=True)

    dt = time.time() - t0
    print(f"\nbest epoch {best_epoch}: acc {best['accuracy']:.4f}  "
          f"macro-F1 {best['macro_f1']:.4f}   ({dt / 60:.1f} min)")

    print(f"\n{'class':<10}{'prec':>8}{'recall':>8}{'F1':>8}{'support':>9}")
    print('-' * 43)
    for name in class_names:
        r = best['per_class'][name]
        print(f"{name:<10}{r['precision']:>8.4f}{r['recall']:>8.4f}"
              f"{r['f1-score']:>8.4f}{int(r['support']):>9}")
    print('-' * 43)
    ma = best['per_class']['macro avg']
    print(f"{'macro avg':<10}{ma['precision']:>8.4f}{ma['recall']:>8.4f}"
          f"{ma['f1-score']:>8.4f}{int(ma['support']):>9}")

    rec = {'args': vars(args), 'parameters': int(n_par),
           'class_names': class_names,
           'loss_weights': dict(zip(LOSS_NAMES, [float(v) for v in weights])),
           'teacher_embed_dim': int(t_emb.shape[1]) if needs_teacher else None,
           'best_epoch': best_epoch, 'val': best, 'minutes': round(dt / 60, 2)}
    out = os.path.join(out_dir, run_name + '.json')
    with open(out, 'w') as fh:
        json.dump(rec, fh, indent=2)
    print(f'saved -> {out}')
    return rec


if __name__ == '__main__':
    main()
