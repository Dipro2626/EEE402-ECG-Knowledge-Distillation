# Pretrained weights

| File | Model | Trained on | Classes (output order) |
|---|---|---|---|
| `student_kd_v2_chapman_without_others_fold4_seed4.pt` | student, 54,709 params | Chapman (fold 4, seed 4) | SR, SB, ST, AFIB/AFL, SVT/AT |
| `student_kd_v2_ningbo_without_others_fold4_seed4.pt` | student, 54,709 params | Ningbo (fold 4, seed 4) | SR, SB, ST, SI, AFIB/AFL |
| `teacherv2_chapman_without_others_fold4.pt` | teacher, 1,817,373 params | Chapman (fold 4) | SR, SB, ST, AFIB/AFL, SVT/AT |
| `teacherv2_ningbo_without_others_min100_fold4.pt` | teacher, 1,817,373 params | Ningbo (fold 4) | SR, SB, ST, SI, AFIB/AFL |

Only the final fold's models are shipped; `run_all.bat` retrains all 5 folds × 5 seeds.
Each student was distilled from the teacher of the same fold.

Student files hold `{'model': state_dict, 'args': training arguments}`; load them with
`student.ECGStudent(num_classes=5)` (see `scripts/predict.py`). Teacher files additionally hold the
class names and the HRV-feature mean/std of their training fold.

Inputs: one ECG lead, 10 s at 500 Hz (5,000 samples), z-scored per record.
Fold-4 models never saw fold 4's validation records (KFold(5, shuffle=True, random_state=42)).
