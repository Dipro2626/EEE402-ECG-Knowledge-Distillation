"""
Compact PyTorch student for 10-second single-lead ECG rhythm classification.

Design constraints
------------------
* Signal only. The 17 HRV features are teacher-side privileged information --
  the published reference drops them at inference too, and the whole point of the compression
  claim is that the student needs nothing but the raw waveform.
* Aggressive striding. 5000 samples is long; four stride-2 convs plus two
  pools bring it to ~39 positions before global pooling.
* Small enough that the compression ratio is the headline. At width=1 this is
  roughly 35k parameters against the teacher's several million.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k, stride=1, pool=1, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, k, stride=stride,
                              padding=k // 2, bias=False)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.pool(self.act(self.bn(self.conv(x)))))


class ECGStudent(nn.Module):
    """
    Returns (logits, embedding). The embedding is what RKD operates on.
    """

    def __init__(self, num_classes=5, width=1.0, embed_dim=64,
                 dropout=0.1, use_gru=True):
        super().__init__()
        w = lambda c: max(8, int(round(c * width)))

        self.features = nn.Sequential(
            ConvBlock(1,     w(16), k=15, stride=2, pool=2, dropout=dropout),
            ConvBlock(w(16), w(32), k=9,  stride=2, pool=2, dropout=dropout),
            ConvBlock(w(32), w(48), k=7,  stride=2, pool=2, dropout=dropout),
            ConvBlock(w(48), w(64), k=5,  stride=2, pool=1, dropout=dropout),
        )

        self.use_gru = use_gru
        if use_gru:
            # bidirectional, so hidden size is halved to keep the output at w(64)
            self.gru = nn.GRU(w(64), w(64) // 2, batch_first=True,
                              bidirectional=True)
            feat_dim = (w(64) // 2) * 2
        else:
            feat_dim = w(64)

        self.embed = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)                 # (B, 5000) -> (B, 1, 5000)
        h = self.features(x)                   # (B, C, L)

        if self.use_gru:
            h, _ = self.gru(h.transpose(1, 2))  # (B, L, C)
            h = h.mean(dim=1)
        else:
            h = h.mean(dim=2)

        emb = self.embed(h)
        return self.classifier(emb), emb


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    for width in (0.5, 1.0, 2.0):
        for gru in (True, False):
            m = ECGStudent(width=width, use_gru=gru)
            out, emb = m(torch.randn(4, 5000))
            print(f'width {width:<4} gru {str(gru):<5} '
                  f'params {count_parameters(m):>8,}  '
                  f'logits {tuple(out.shape)}  emb {tuple(emb.shape)}')
