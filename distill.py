"""
Distillation losses. RKD follows Park et al. (2019), 'Relational Knowledge
Distillation', matching the official repository.

The reason RKD is the right tool here: the teacher embedding is 32-d (signal
branch) or 160-d (joint) and the student's is 64-d. Distance- and angle-wise
losses compare *relations between* embeddings, which are scale- and
dimension-free, so no projection layer is needed. FitNet, included as a
baseline, does need one -- and that projection is extra parameters that exist
only during training, which muddies a compression claim.

Note that the joint teacher embedding (160-d) is WIDER than the signal one
(32-d) mostly because of the 128-d feature branch, which encodes the 17 HRV
features. The student never sees those features, so relational distillation
from the joint embedding is genuine privileged-information transfer -- that is
the scientifically interesting target, and the default.

The weights in MODES below are the RKD paper's and are NOT used as-is:
train_student.py rescales them so cross-entropy carries a declared share of the
total loss. See auto_balance() there for why.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pdist(e, squared=False, eps=1e-12):
    e_sq = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_sq.unsqueeze(1) + e_sq.unsqueeze(0) - 2 * prod).clamp(min=eps)
    if not squared:
        res = res.sqrt()
    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RkdDistance(nn.Module):
    """Match pairwise distances, each normalised by its own batch mean."""

    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            t_d = t_d / (t_d[t_d > 0].mean() + 1e-12)
        d = pdist(student, squared=False)
        d = d / (d[d > 0].mean() + 1e-12)
        return F.smooth_l1_loss(d, t_d, reduction='mean')


class RKdAngle(nn.Module):
    """Match the cosine of every angle formed by triplets in the batch."""

    def forward(self, student, teacher):
        with torch.no_grad():
            td = teacher.unsqueeze(0) - teacher.unsqueeze(1)
            td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(td, td.transpose(1, 2)).view(-1)
        sd = student.unsqueeze(0) - student.unsqueeze(1)
        sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(sd, sd.transpose(1, 2)).view(-1)
        return F.smooth_l1_loss(s_angle, t_angle, reduction='mean')


class LogitKD(nn.Module):
    """
    Hinton et al. response distillation.

    teacher_logits are the true pre-softmax logits of the frozen Teacher v2,
    cached by export_teacher_v2.py.
    """

    def __init__(self, T=4.0):
        super().__init__()
        self.T = T

    def forward(self, student_logits, teacher_logits):
        s = F.log_softmax(student_logits / self.T, dim=1)
        t = F.softmax(teacher_logits / self.T, dim=1)
        return F.kl_div(s, t, reduction='batchmean') * (self.T ** 2)


class FitNet(nn.Module):
    """Point-wise feature matching. Needs a projection, unlike RKD."""

    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_emb, teacher_emb):
        return F.mse_loss(self.proj(student_emb), teacher_emb)


MODES = {
    # mode      -> (ce, kd, rkd_distance, rkd_angle, fitnet)
    'ce':       (1.0, 0.0, 0.0,  0.0, 0.0),
    'kd':       (1.0, 1.0, 0.0,  0.0, 0.0),
    'fitnet':   (1.0, 0.0, 0.0,  0.0, 1.0),
    'rkd_d':    (1.0, 0.0, 25.0, 0.0, 0.0),
    'rkd_a':    (1.0, 0.0, 0.0,  50.0, 0.0),
    'rkd':      (1.0, 0.0, 25.0, 50.0, 0.0),
    'rkd_kd':   (1.0, 1.0, 25.0, 50.0, 0.0),
}
# The 25 / 50 weighting is the ratio recommended in the RKD paper. It looks
# large only because the distance and angle terms are small in magnitude;
# train_student.py prints the realised loss split at epoch 1 so the balance can
# be checked rather than assumed.
