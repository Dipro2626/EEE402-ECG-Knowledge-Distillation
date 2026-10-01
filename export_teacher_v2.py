"""
Freeze Teacher v2 and cache its logits and embeddings for the distillation grid.

    python export_teacher_v2.py --fold 0

Writes cache/teacherv2_<tag>_fold<k>.npz with the same key layout as the v1
cache, so train_student.py needs only a filename switch (`--teacher v2`).

Two decisions worth stating explicitly:

1. **The teacher uses the privileged HRV features when producing targets.**
   That is the point of privileged-information distillation: the teacher sees
   more than the student ever will, and the extra knowledge reaches the student
   only through the soft targets. The student still receives waveform alone.

2. **The exported embedding is the 256-d trunk output, which excludes the
   feature branch.** RKD therefore transfers only waveform-derived relational
   structure — the student is never asked to reproduce geometry it has no way of
   computing. This is also what makes the v1-vs-v2 comparison clean: both
   embeddings are trunk-only, and the only difference is width (32 vs 256).

Unlike v1 these are true pre-softmax logits, not log-probabilities, so the
temperature in logit-KD behaves exactly as intended. That is a small extra
difference from v1 and is noted in the report rather than hidden.
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from data import load_dataset, get_folds
from teacher_v2 import TeacherV2, SIG_LEN
from paths import CKPT_DIR, CACHE_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='chapman')
    ap.add_argument('--variant', default='without_others')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--no_privileged', action='store_true',
                    help='produce targets without the HRV branch (ablation)')
    ap.add_argument('--tag', default='v2',
                    help='which teacher variant to export; must match the '
                         '--tag used when training')
    ap.add_argument('--min_class', type=int, default=0,
                    help='must match the value used when the teacher was '
                         'trained; it is part of the checkpoint filename')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    device = torch.device(args.device)
    tag = f'{args.dataset}_{args.variant}'
    if args.min_class:
        tag += f'_min{args.min_class}'
    ck_path = os.path.join(CKPT_DIR, f'teacher{args.tag}_{tag}_fold{args.fold}.pt')
    if not os.path.exists(ck_path):
        raise SystemExit(f'{ck_path} not found — run teacher_v2.py --fold '
                         f'{args.fold} --tag {args.tag} first.')

    ck = torch.load(ck_path, map_location=device, weights_only=False)
    a = ck['args']
    class_names = ck['class_names']

    X, y, names = load_dataset(args.dataset, args.variant, verbose=False,
                               min_count=args.min_class)
    if names != class_names:
        raise SystemExit(f'class list changed since training: {names} vs {class_names}')
    tr, va = get_folds(X, y)[args.fold]

    sig = X[:, :SIG_LEN]
    sig = (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True) + 1e-8)
    feats = (X[:, SIG_LEN:] - ck['feat_mu']) / ck['feat_sd']
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    use_feat = a.get('use_features', not a.get('no_features', False))
    model = TeacherV2(num_classes=len(class_names),
                      widths=tuple(a.get('widths', [32, 64, 128, 256])),
                      use_features=use_feat,
                      embed_proj=a.get('embed_proj') or None).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    loader = DataLoader(TensorDataset(torch.from_numpy(sig).float(),
                                      torch.from_numpy(feats).float()),
                        batch_size=args.batch, shuffle=False)

    give_feats = use_feat and not args.no_privileged
    L, E = [], []
    with torch.no_grad():
        for xb, fb in loader:
            logits, emb = model(xb.to(device),
                                fb.to(device) if give_feats else None)
            L.append(logits.cpu())
            E.append(emb.cpu())
    logits = torch.cat(L).numpy().astype(np.float32)
    emb = torch.cat(E).numpy().astype(np.float32)

    out = os.path.join(CACHE_DIR, f'teacher{args.tag}_{tag}_fold{args.fold}.npz')
    np.savez_compressed(
        out,
        logits_signal=logits, logits_joint=logits,
        emb_signal=emb, emb_joint=emb,
        y=y.astype(np.int64),
        class_names=np.array(class_names),
        idx_train=tr.astype(np.int64), idx_val=va.astype(np.int64))

    pred = logits[va].argmax(1)
    acc = float((pred == y[va]).mean())
    mf = float(f1_score(y[va], pred, average='macro',
                        labels=list(range(len(class_names))), zero_division=0))

    print(f'teacher v2, fold {args.fold}')
    print(f'  embedding      : {emb.shape[1]}-d  (v1 was 32-d signal / 160-d joint)')
    print(f'  privileged HRV : {"used" if give_feats else "withheld"}')
    print(f'  val accuracy   : {100 * acc:.2f}%')
    print(f'  val macro-F1   : {100 * mf:.2f}%')
    print(f'\nsaved -> {out}  ({os.path.getsize(out) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
