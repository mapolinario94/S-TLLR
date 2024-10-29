# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import copy
from .surrogate_gradients import *


class BPTTLIF(nn.Module):
    """
    Leaky Integrate and Fire layer with learnable parameters
    """
    def __init__(self,
                 threshold: float = 0.6,
                 leak: float = 2.0,
                 grad_tl: bool = False,
                 activation=None,
                 reset_mechanism: str = "soft",
                 accumulate: bool = False):
        super(BPTTLIF, self).__init__()
        self.u = None
        self.state_reset = None
        self.trace_spike = None
        self.leak = nn.Parameter(torch.tensor(leak), requires_grad=grad_tl)
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=grad_tl)
        self.reset_mechanism = reset_mechanism
        self.accumulate = accumulate
        if activation is None:
            self.activation = LinearSpike.apply
        else:
            self.activation = activation.apply

    def reset_state(self):
        self.u = copy.deepcopy(self.state_reset)

    def _init_states(self, x):
        if self.u is None or self.u.shape[0] != x.shape[0]:
            self.u = torch.zeros_like(x).to(x.device)
            self.state_reset = torch.zeros_like(x).to(x.device)

    def forward(self, x: torch.tensor):
        self._init_states(x)

        if not self.accumulate:
            self.u = torch.sigmoid(self.leak) * self.u + x
            u_thr = self.u - self.threshold.clamp(min=0.5)
            out = self.activation(u_thr)
            rst = out.detach()
            if self.reset_mechanism == "hard":
                self.u = self.u * (1 - rst)
            else:
                self.u = self.u - self.threshold.clamp(min=0.5) * rst
        else:
            self.u = torch.sigmoid(self.leak)*self.u + x
            out = self.u

        return out

    def extra_repr(self) -> str:
        return 'threshold={0:.2f}, leak={1:.2f}, reset={2}, grad_leak={3}, grad_threshold={4}'.format(
            self.threshold, torch.sigmoid(self.leak), self.reset_mechanism, self.leak.requires_grad,
            self.threshold.requires_grad
        )

