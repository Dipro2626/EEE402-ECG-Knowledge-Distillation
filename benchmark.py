"""
Deployment cost of the student against its teacher: size, latency, memory, and
what INT8 quantization and magnitude pruning actually buy.

    python benchmark.py                      # student vs teacher, CPU + GPU
    python benchmark.py --no_gpu

A parameter count is not a deployment claim. Reviewers of compression papers ask
for latency and memory on real hardware, and for a comparison against the
cheaper alternatives -- quantization and pruning -- because those require no
teacher, no second training run, and no distillation loss at all. This script
produces those numbers so the compression claim can stand or fall on evidence.

Honest scope, stated up front:

* **Batch 1 is the number that matters.** A wearable classifies one 10-second
  strip at a time; throughput at batch 64 is a different, easier question. Both
  are reported, and they differ by more than an order of magnitude per record.
* **Latency is reported as median and IQR, not mean.** The distribution is
  right-skewed by scheduling noise, so a mean flatters whichever model happened
  to get a quiet moment.
* **Dynamic INT8 quantization covers Linear and GRU layers only.** Conv1d needs
  static quantization with calibration, which is attempted separately and may
  be unavailable depending on the backend. The size reduction reported is
  therefore a floor, not the best achievable.
* **Unstructured pruning removes weights, not work.** Without sparse kernels a
  90%-sparse model runs at exactly the same speed and occupies the same memory
  as the dense one. Its accuracy curve answers "how redundant is this network",
  not "how fast can it go". Structured channel pruning is the version that
  would deliver speed, and is not attempted here.
"""

import argparse
import copy
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from data import load_dataset, get_folds
from student import ECGStudent, count_parameters
from teacher_v2 import TeacherV2, SIG_LEN
from paths import CKPT_DIR, RESULT_DIR


# --------------------------------------------------------------------------- #
def model_size_mb(model):
    """Serialized size, which is what actually ships to a device."""
    tmp = os.path.join(RESULT_DIR, '_size_probe.pt')
    torch.save(model.state_dict(), tmp)
    mb = os.path.getsize(tmp) / 1e6
    os.remove(tmp)
    return mb


@torch.no_grad()
def latency(model, device, batch=1, n=200, warmup=30, feats=False):
    """Median and IQR milliseconds per forward pass."""
    model.eval().to(device)
    x = torch.randn(batch, SIG_LEN, device=device)
    f = torch.randn(batch, 17, device=device) if feats else None

    for _ in range(warmup):
        model(x, f) if feats else model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(x, f) if feats else model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    t = np.array(times)
    return {
        'median_ms': float(np.median(t)),
        'iqr_ms': float(np.percentile(t, 75) - np.percentile(t, 25)),
        'per_record_ms': float(np.median(t) / batch),
    }


@torch.no_grad()
def peak_gpu_mb(model, device, batch=1, feats=False):
    if device.type != 'cuda':
        return None
    model.eval().to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch, SIG_LEN, device=device)
    f = torch.randn(batch, 17, device=device) if feats else None
    model(x, f) if feats else model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


@torch.no_grad()
def accuracy(model, loader, device, n_classes, feats=False):
    model.eval().to(device)
    p, t = [], []
    for batch in loader:
        x = batch[0].to(device)
        f = batch[1].to(device) if feats else None
        logits, _ = model(x, f) if feats else model(x)
        p.append(logits.argmax(1).cpu())
        t.append(batch[-1])
    p, t = torch.cat(p).numpy(), torch.cat(t).numpy()
    return {
        'accuracy': float((p == t).mean()),
        'macro_f1': float(f1_score(t, p, average='macro',
                                   labels=list(range(n_classes)),
                                   zero_division=0)),
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='chapman')
    ap.add_argument('--variant', default='without_others')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mode', default='kd', help='which student checkpoint')
    ap.add_argument('--teacher', default='v2')
    ap.add_argument('--no_gru', action='store_true',
                    help='benchmark the convolution-only student')
    ap.add_argument('--width', type=float, default=1.0)
    ap.add_argument('--embed_dim', type=int, default=64)
    ap.add_argument('--ckpt', default=None,
                    help='explicit student checkpoint, overriding the name built '
                         'from the flags above')
    ap.add_argument('--batches', type=int, nargs='+', default=[1, 64])
    ap.add_argument('--no_gpu', action='store_true')
    args = ap.parse_args()

    tag = f'{args.dataset}_{args.variant}'
    cpu = torch.device('cpu')
    gpu = (torch.device('cuda')
           if torch.cuda.is_available() and not args.no_gpu else None)

    X, y, names = load_dataset(args.dataset, args.variant, verbose=False)
    n_classes = len(names)
    _, va = get_folds(X, y)[args.fold]

    sig = X[:, :SIG_LEN]
    sig = (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True) + 1e-8)

    # ---- student ----------------------------------------------------------
    if args.ckpt:
        s_path = args.ckpt
    else:
        # Must mirror the naming in train_student.py exactly, or the wrong
        # architecture gets benchmarked under the right label.
        vtag = ('' if (args.teacher == 'v1' or args.mode == 'ce')
                else f'_{args.teacher}')
        atag = ''
        if args.no_gru:
            atag += '_nogru'
        if args.width != 1.0:
            atag += f'_w{args.width:g}'
        if args.embed_dim != 64:
            atag += f'_e{args.embed_dim}'
        s_path = os.path.join(
            CKPT_DIR, f'student_{args.mode}{vtag}{atag}_{tag}'
                      f'_fold{args.fold}_seed{args.seed}.pt')
    if not os.path.exists(s_path):
        raise SystemExit(f'{s_path} not found -- run the student grid first.')
    print(f'student checkpoint: {os.path.basename(s_path)}\n')
    sck = torch.load(s_path, map_location=cpu, weights_only=False)
    sa = sck['args']
    student = ECGStudent(num_classes=n_classes, width=sa.get('width', 1.0),
                         embed_dim=sa.get('embed_dim', 64),
                         use_gru=not sa.get('no_gru', False))
    student.load_state_dict(sck['model'])

    s_loader = DataLoader(
        TensorDataset(torch.from_numpy(sig[va]).float(),
                      torch.from_numpy(y[va])), batch_size=256)

    # ---- teacher ----------------------------------------------------------
    t_path = os.path.join(CKPT_DIR, f'teacher{args.teacher}_{tag}_fold{args.fold}.pt')
    teacher = t_loader = None
    if os.path.exists(t_path):
        tck = torch.load(t_path, map_location=cpu, weights_only=False)
        ta = tck['args']
        teacher = TeacherV2(num_classes=n_classes,
                            widths=tuple(ta.get('widths', [32, 64, 128, 256])),
                            use_features=ta.get('use_features',
                                                not ta.get('no_features', False)),
                            embed_proj=ta.get('embed_proj') or None)
        teacher.load_state_dict(tck['model'])
        feats = (X[:, SIG_LEN:] - tck['feat_mu']) / tck['feat_sd']
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        t_loader = DataLoader(
            TensorDataset(torch.from_numpy(sig[va]).float(),
                          torch.from_numpy(feats[va]).float(),
                          torch.from_numpy(y[va])), batch_size=256)
    else:
        print(f'NOTE: {t_path} not found; teacher comparison skipped.\n')

    report = {'args': vars(args)}

    # ---- size and accuracy ------------------------------------------------
    print('=' * 72)
    print('MODEL SIZE AND ACCURACY (Chapman fold %d validation)' % args.fold)
    print('=' * 72)
    rows = []
    s_acc = accuracy(student, s_loader, cpu, n_classes)
    rows.append(('Student (CE+KD)', count_parameters(student),
                 model_size_mb(student), s_acc))
    if teacher is not None:
        t_acc = accuracy(teacher, t_loader, cpu, n_classes, feats=True)
        rows.append(('Teacher v2', count_parameters(teacher),
                     model_size_mb(teacher), t_acc))

    print(f"{'model':<20}{'params':>12}{'size MB':>10}{'accuracy':>11}{'macro-F1':>11}")
    print('-' * 64)
    for n, p, mb, a in rows:
        print(f"{n:<20}{p:>12,}{mb:>10.2f}{100 * a['accuracy']:>11.2f}"
              f"{100 * a['macro_f1']:>11.2f}")
    if teacher is not None:
        print(f"\ncompression: {rows[1][1] / rows[0][1]:.1f}x parameters, "
              f"{rows[1][2] / rows[0][2]:.1f}x on disk, "
              f"{100 * (rows[0][3]['accuracy'] - rows[1][3]['accuracy']):+.2f} "
              f"accuracy points")
    report['size_accuracy'] = [
        {'model': n, 'params': p, 'size_mb': mb, **a} for n, p, mb, a in rows]

    # ---- latency ----------------------------------------------------------
    print('\n' + '=' * 72)
    print('LATENCY  (median [IQR] ms per forward pass)')
    print('=' * 72)
    lat = {}
    for dev_name, dev in [('CPU', cpu)] + ([('GPU', gpu)] if gpu else []):
        for b in args.batches:
            s = latency(student, dev, batch=b)
            lat[f'student_{dev_name}_b{b}'] = s
            line = (f'{dev_name:<4} batch {b:<4} student '
                    f"{s['median_ms']:8.3f} [{s['iqr_ms']:.3f}]  "
                    f"= {s['per_record_ms']:7.4f} ms/record")
            if teacher is not None:
                t = latency(teacher, dev, batch=b, feats=True)
                lat[f'teacher_{dev_name}_b{b}'] = t
                line += (f"   | teacher {t['median_ms']:8.3f}  "
                         f"-> {t['median_ms'] / s['median_ms']:5.1f}x slower")
            print(line)
        student.to(cpu)
        if teacher is not None:
            teacher.to(cpu)
    report['latency'] = lat

    if gpu:
        print('\nPeak GPU memory, batch 1:')
        print(f'  student {peak_gpu_mb(student, gpu):.2f} MB')
        if teacher is not None:
            print(f'  teacher {peak_gpu_mb(teacher, gpu, feats=True):.2f} MB')
        student.to(cpu)
        if teacher is not None:
            teacher.to(cpu)

    # ---- INT8 dynamic quantization ---------------------------------------
    print('\n' + '=' * 72)
    print('INT8 DYNAMIC QUANTIZATION  (Linear and GRU only; Conv1d unchanged)')
    print('=' * 72)
    try:
        q = torch.ao.quantization.quantize_dynamic(
            copy.deepcopy(student).cpu(), {nn.Linear, nn.GRU}, dtype=torch.qint8)
        q_acc = accuracy(q, s_loader, cpu, n_classes)
        q_lat = latency(q, cpu, batch=1)
        q_mb = model_size_mb(q)
        print(f"  size      {model_size_mb(student):.2f} MB -> {q_mb:.2f} MB  "
              f"({100 * (1 - q_mb / model_size_mb(student)):.1f}% smaller)")
        print(f"  accuracy  {100 * s_acc['accuracy']:.2f}% -> "
              f"{100 * q_acc['accuracy']:.2f}%  "
              f"({100 * (q_acc['accuracy'] - s_acc['accuracy']):+.2f})")
        print(f"  CPU b1    {lat['student_CPU_b1']['median_ms']:.3f} ms -> "
              f"{q_lat['median_ms']:.3f} ms")
        report['int8_dynamic'] = {'size_mb': q_mb, **q_acc, **q_lat}
    except Exception as e:
        print(f'  unavailable on this build: {e!r}')
        report['int8_dynamic'] = None

    # ---- magnitude pruning ------------------------------------------------
    print('\n' + '=' * 72)
    print('GLOBAL MAGNITUDE PRUNING  (accuracy vs sparsity)')
    print('=' * 72)
    print('  Unstructured pruning removes weights, not computation: without')
    print('  sparse kernels these models run at the same speed and occupy the')
    print('  same memory. This curve measures redundancy, not deployment cost.')
    print()
    import torch.nn.utils.prune as prune
    targets = [(m, 'weight') for m in student.modules()
               if isinstance(m, (nn.Conv1d, nn.Linear))]
    prune_rows = []
    for sparsity in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        m = copy.deepcopy(student)
        if sparsity > 0:
            tg = [(mm, 'weight') for mm in m.modules()
                  if isinstance(mm, (nn.Conv1d, nn.Linear))]
            prune.global_unstructured(tg, pruning_method=prune.L1Unstructured,
                                      amount=sparsity)
            for mm, nme in tg:
                prune.remove(mm, nme)
        a = accuracy(m, s_loader, cpu, n_classes)
        nz = sum(int((p != 0).sum()) for p in m.parameters())
        print(f"  sparsity {sparsity:4.0%}  nonzero {nz:>8,}  "
              f"accuracy {100 * a['accuracy']:6.2f}%  "
              f"macro-F1 {100 * a['macro_f1']:6.2f}%")
        prune_rows.append({'sparsity': sparsity, 'nonzero': nz, **a})
    report['pruning'] = prune_rows

    # Architecture must appear in the filename, or benchmarking a second width
    # silently overwrites the first one's results.
    btag = f'_{args.mode}' + ('_nogru' if args.no_gru else '')
    if args.width != 1.0:
        btag += f'_w{args.width:g}'
    out = os.path.join(RESULT_DIR, f'benchmark{btag}_{tag}_fold{args.fold}.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
