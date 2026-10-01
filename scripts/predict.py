"""
Classify one 10-second single-lead ECG with a pretrained student.

    python scripts/predict.py --ecg my_ecg.csv
    python scripts/predict.py --ecg my_ecg.npy --ckpt checkpoints/student_kd_v2_chapman_without_others_fold4_seed4.pt

The ECG file must hold one lead sampled at 500 Hz; the first 5,000 samples
(10 s) are used and a shorter signal is zero-padded. Amplitude units do not
matter because every record is z-scored.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from student import ECGStudent  # noqa: E402

CLASSES = {
    'ningbo': ['SR', 'SB', 'ST', 'SI', 'AFIB/AFL'],
    'chapman': ['SR', 'SB', 'ST', 'AFIB/AFL', 'SVT/AT'],
}
NAMES = {'SR': 'Sinus rhythm', 'SB': 'Sinus bradycardia', 'ST': 'Sinus tachycardia', 'SI': 'Sinus irregularity',
         'AFIB/AFL': 'Atrial fibrillation / flutter', 'SVT/AT': 'Supraventricular / atrial tachycardia'}


def load_ecg(path):
    x = np.load(path) if path.endswith('.npy') else np.loadtxt(path, delimiter=',' if path.endswith('.csv') else None)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    out = np.zeros(5000, np.float32)
    out[:min(5000, len(x))] = x[:5000]
    return (out - out.mean()) / (out.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ecg', required=True, help='.npy, .csv or .txt with one lead at 500 Hz')
    ap.add_argument('--ckpt', default=os.path.join(ROOT, 'checkpoints', 'student_kd_v2_ningbo_without_others_fold4_seed4.pt'))
    a = ap.parse_args()

    try:
        ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    except TypeError:                                   # older PyTorch
        ck = torch.load(a.ckpt, map_location='cpu')
    ds = 'ningbo' if 'ningbo' in os.path.basename(a.ckpt) else 'chapman'
    classes = CLASSES[ds]
    model = ECGStudent(num_classes=len(classes), width=float(ck['args'].get('width', 1.0)),
                       embed_dim=int(ck['args'].get('embed_dim', 64)))
    model.load_state_dict(ck['model'])
    model.eval()

    x = torch.from_numpy(load_ecg(a.ecg)).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(x)[0], dim=1)[0].numpy()
    k = int(p.argmax())
    print(f'Predicted rhythm: {classes[k]} ({NAMES[classes[k]]}), probability {p[k]:.3f}')
    for c, v in sorted(zip(classes, p), key=lambda t: -t[1]):
        print(f'  {c:<9} {v:.3f}')
    print('Screening output only - not a medical diagnosis.')


if __name__ == '__main__':
    main()
