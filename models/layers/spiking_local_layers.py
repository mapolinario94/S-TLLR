# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
__all__ = ["DropoutLIF"]


class DropoutLIF(nn.Module):
    """
    Dropout layer for SNN
    """
    def __init__(self,
                 p):
        super(DropoutLIF, self).__init__()
        self.mask = None
        self.ones = torch.ones(1)
        self.dropout = nn.Dropout(p)

    def reset_state(self):
        self.mask = self.dropout(self.ones).trunc()

    def _init_states(self, x):
        if self.mask.dim() == 1:
            self.ones = torch.ones_like(x)
            self.mask = self.dropout(self.ones).trunc()
        elif self.mask.shape[0] != x.shape[0]:
            self.ones = torch.ones_like(x)
            self.mask = self.dropout(self.ones).trunc()

    def forward(self, x: torch.tensor):
        self._init_states(x)
        return self.mask*x

    def extra_repr(self) -> str:
        return f'{self.dropout}'

