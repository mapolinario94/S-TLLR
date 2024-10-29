import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import copy
import math
__all__ = ["LinearSTLLR", "Conv2dSTLLR", "LinearRecSTLLR"]


class LinearSTLLR(nn.Linear):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 threshold: float = 0.6,
                 leak: float = 2.0,
                 grad_tl: bool = False,
                 activation=None,
                 reset_mechanism: str = "soft",
                 accumulate: bool = False,
                 device=None,
                 dtype=None,
                 factors=None,
                 **kwargs
                 ):
        super(LinearSTLLR, self).__init__(in_features, out_features, bias, device, dtype)
        self.u = None
        self.state_reset = None
        self.trace_in = None
        self.trace_out = None
        self.leak = nn.Parameter(torch.tensor(leak), requires_grad=grad_tl)
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=grad_tl)
        self.reset_mechanism = reset_mechanism
        self.accumulate = accumulate
        self.gain = nn.Parameter(torch.ones(self.out_features, 1))
        if activation is None:
            if not accumulate:
                self.activation = STLLRLinearGrad.apply
            else:
                self.activation = STLLRAccumulationGrad.apply
        else:
            self.activation = activation.apply
        self.eps = 1e-4

        if factors is None:
            # factors are the STDP parameters discussed in the paper
            # $[\lambda_{post}, \lambda_{pre}, \alpha_{post}, \alpha_{pre}]$
            self.register_buffer("factors", torch.tensor([0.5, 0.8, -0.2, 1]))
        else:
            self.register_buffer("factors", torch.tensor(factors))

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1], keepdims=True)
        var = torch.var(self.weight, axis=[1], keepdims=True)
        weight = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def reset_state(self):
        if self.u is not None:
            with torch.no_grad():
                self.u = self.u.mul(0).detach()
                self.trace_in = self.trace_in.mul(0).detach()
                self.trace_out = self.trace_out.mul(0).detach()

    def _init_states(self, x):
        batch_size = x.size(0)
        if self.u is None or self.u.shape[0] != batch_size:
            with torch.no_grad():
                a = F.linear(x, self.weight, None)
                self.u = torch.zeros_like(a).to(x.device)
        # if self.training:
        if self.trace_in is None or self.trace_in.shape[0] != batch_size:
            self.trace_in = torch.zeros([batch_size, self.in_features]).to(x.device)
            self.trace_out = torch.zeros([batch_size, self.out_features]).to(x.device)

    def trace_input(self, spikes):
        with torch.no_grad():
            self.trace_in = self.factors[1] * self.trace_in + spikes.clone().detach()

    def forward(self, input: torch.Tensor):
        self._init_states(input)

        if not self.accumulate:
            out, mem, trace_in, trace_out = self.activation(input, self.get_weight(), self.bias, self.trace_in,
                                                            self.trace_out, self.u, self.leak, self.threshold,
                                                            self.factors)
            self.u = mem.detach()
            self.trace_out = trace_out.detach()
            self.trace_in = trace_in.detach()
            rst = out.detach()
            if self.reset_mechanism == "hard":
                self.u = self.u * (1 - rst)
            else:
                self.u = self.u - self.threshold.clamp(min=0.5) * rst

        else:
            self.trace_input(input)
            x = self.activation(input, self.weight, self.bias, self.trace_in)
            self.u = torch.sigmoid(self.leak)*self.u.detach() + x
            out = self.u

        return out

    def extra_repr(self) -> str:
        return 'in_features={0}, out_features={1}, bias={2}, threshold={3:.2f}, leak={4:.2f}, reset={5}, grad_leak={6}, grad_threshold={7}, factors={8}'.format(
            self.in_features, self.out_features, self.bias is not None,
            self.threshold.data, torch.sigmoid(self.leak), self.reset_mechanism, self.leak.requires_grad,
            self.threshold.requires_grad, self.factors
        )


class LinearRecSTLLR(nn.Linear):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 threshold: float = 0.6,
                 leak: float = 2.0,
                 grad_tl: bool = False,
                 activation=None,
                 reset_mechanism: str = "soft",
                 device=None,
                 dtype=None,
                 factors=None,
                 **kwargs
                 ):
        super(LinearRecSTLLR, self).__init__(in_features, out_features, bias, device, dtype)
        self.u = None
        self.state_reset = None
        self.trace_in = None
        self.trace_in_rec = None
        self.trace_out = None
        self.output_spikes = None
        self.leak = nn.Parameter(torch.tensor(leak), requires_grad=grad_tl)
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=grad_tl)
        self.reset_mechanism = reset_mechanism
        self.gain = nn.Parameter(torch.ones(self.out_features, 1))
        self.gain_rec = nn.Parameter(torch.ones(self.out_features, 1))

        self.weight_rec = Parameter(torch.empty((out_features, out_features)))
        torch.nn.init.kaiming_uniform_(self.weight_rec, a=math.sqrt(5))
        if activation is None:
            self.activation = STLLRLinearRecGrad.apply
        else:
            self.activation = activation.apply
        self.eps = 1e-4

        if factors is None:
            # factors are the STDP parameters discussed in the paper
            # $[\lambda_{post}, \lambda_{pre}, \alpha_{post}, \alpha_{pre}]$
            self.register_buffer("factors", torch.tensor([0.5, 0.8, -0.2, 1]))
        else:
            self.register_buffer("factors", torch.tensor(factors))

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1], keepdims=True)
        var = torch.var(self.weight, axis=[1], keepdims=True)
        weight = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
        if self.gain is not None:
            weight = weight * self.gain

        fan_in = np.prod(self.weight_rec.shape[1:])
        mean = torch.mean(self.weight_rec, axis=[1], keepdims=True)
        var = torch.var(self.weight_rec, axis=[1], keepdims=True)
        weight_rec = (self.weight_rec - mean) / ((var * fan_in + self.eps) ** 0.5)
        if self.gain_rec is not None:
            weight_rec = weight_rec * self.gain_rec
        return weight, weight_rec

    def reset_state(self):
        if self.u is not None:
            with torch.no_grad():
                self.u = self.u.mul(0).detach()
                self.trace_in = self.trace_in.mul(0).detach()
                self.trace_in_rec = self.trace_in_rec.mul(0).detach()
                self.trace_out = self.trace_out.mul(0).detach()
                self.output_spikes = self.output_spikes.mul(0).detach()

    def _init_states(self, x):
        batch_size = x.size(0)
        if self.u is None or self.u.shape[0] != batch_size:
            with torch.no_grad():
                a = F.linear(x, self.weight, None)
                self.u = torch.zeros_like(a).to(x.device)
        # if self.training:
        if self.trace_in is None or self.trace_in.shape[0] != batch_size:
            self.trace_in = torch.zeros([batch_size, self.in_features]).to(x.device)
            self.trace_in_rec = torch.zeros([batch_size, self.out_features]).to(x.device)
            self.trace_out = torch.zeros([batch_size, self.out_features]).to(x.device)
            self.output_spikes = torch.zeros([batch_size, self.out_features]).to(x.device)

    def forward(self, input: torch.Tensor):
        self._init_states(input)

        w, w_rec = self.get_weight()
        out, mem, trace_in, trace_in_rec, trace_out = self.activation(input, self.output_spikes, w, w_rec, self.bias,
                                                                      self.trace_in, self.trace_in_rec, self.trace_out,
                                                                      self.u, self.leak, self.threshold, self.factors)
        self.u = mem.detach()
        rst = out.detach()
        self.output_spikes = out.detach()
        self.trace_in = trace_in.detach()
        self.trace_in_rec = trace_in_rec.detach()
        self.trace_out = trace_out.detach()
        if self.reset_mechanism == "hard":
            self.u = self.u * (1 - rst)
        else:
            self.u = self.u - self.threshold.clamp(min=0.5) * rst

        return out

    def extra_repr(self) -> str:
        return 'in_features={0}, out_features={1}, bias={2}, threshold={3:.2f}, leak={4:.2f}, reset={5}, grad_leak={6}, grad_threshold={7}, factors={8}'.format(
            self.in_features, self.out_features, self.bias is not None,
            self.threshold.data, torch.sigmoid(self.leak), self.reset_mechanism, self.leak.requires_grad,
            self.threshold.requires_grad, self.factors
        )


class Conv2dSTLLR(nn.Conv2d):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 threshold: float = 0.6,
                 leak: float = 2.0,
                 grad_tl: bool = False,
                 activation=None,
                 reset_mechanism: str = "soft",
                 accumulate: bool = False,
                 device=None,
                 dtype=None,
                 factors=None,
                 normalization=True,
                 **kwargs
                 ):
        super(Conv2dSTLLR, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                          padding_mode, device, dtype)
        self.u = None
        self.state_reset = None
        self.trace_in = None
        self.trace_out = None
        self.leak = nn.Parameter(torch.tensor(leak), requires_grad=grad_tl)
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=grad_tl)
        self.reset_mechanism = reset_mechanism
        self.accumulate = accumulate
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.normalization = normalization
        if activation is None:
            self.activation = STLLRConv2dGrad.apply
        else:
            self.activation = activation.apply
        self.eps = 1e-4
        if factors is None:
            # factors are the STDP parameters discussed in the paper
            # $[\lambda_{post}, \lambda_{pre}, \alpha_{post}, \alpha_{pre}]$
            self.register_buffer("factors", torch.tensor([0.5, 0.8, -0.2, 1]))
        else:
            self.register_buffer("factors", torch.tensor(factors))

    def get_weight(self):
        if self.normalization:
            fan_in = np.prod(self.weight.shape[1:])
            mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
            var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
            weight = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
            if self.gain is not None:
                weight = weight * self.gain
            return weight
        else:
            return self.weight

    def reset_state(self):
        if self.u is not None:
            with torch.no_grad():
                self.u = self.u.mul(0).detach()
                self.trace_in = self.trace_in.mul(0).detach()
                self.trace_out = self.trace_out.mul(0).detach()

    def _init_states(self, x):
        batch_size = x.size(0)
        if self.u is None or self.u.shape[0] != batch_size:
            with torch.no_grad():
                a = F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
                self.u = torch.zeros_like(a).to(x.device)
                self.trace_out = torch.zeros_like(a).to(x.device)
        # if self.training:
        if self.trace_in is None or self.trace_in.shape[0] != batch_size:
            self.trace_in = torch.zeros_like(x).to(x.device)

    def forward(self, input: torch.Tensor):
        self._init_states(input)
        out, u, trace_in, trace_out = self.activation(input, self.get_weight(), self.bias, self.stride, self.padding,
                                                                  self.dilation, self.groups, self.trace_in,
                                                                  self.trace_out, self.u, self.leak, self.threshold,
                                                                  self.factors)
        self.u = u.detach()
        self.trace_out = trace_out.detach()
        self.trace_in = trace_in.detach()
        rst = out.detach()
        if self.reset_mechanism == "hard":
            self.u = self.u * (1 - rst)
        else:
            self.u = self.u - self.threshold.clamp(min=0.5) * rst

        return out

    def extra_repr(self) -> str:
        s = super(nn.Conv2d, self).extra_repr()
        return s + ', threshold={0:.2f}, leak={1:.2f}, reset={2}, grad_leak={3}, grad_threshold={4}, factors={5}'.format(
            self.threshold.data, torch.sigmoid(self.leak), self.reset_mechanism, self.leak.requires_grad,
            self.threshold.requires_grad, self.factors
        )


class STLLRAccumulationGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, trace):
        ctx.save_for_backward(input, weight, bias, trace)
        with torch.no_grad():
            output = F.linear(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, trace = ctx.saved_tensors

        grad_input = torch.mm(grad_output, weight)

        grad_weight = torch.mm(grad_output.T, trace)
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias, None, None, None


class STLLRLinearGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, trace_in, trace_out, mem, leak, threshold, factors):

        with torch.no_grad():
            # Trace of the pre-synaptic activity $\mathrm{tr}x_j[t]$, first term of the RHS in equation (10)
            trace_in = factors[1] * trace_in + input
            # LIF computations
            output = F.linear(input, weight, bias)
            mem = torch.sigmoid(leak) * mem + output
            u_thr = mem - threshold.clamp(min=0.5)
            output = (u_thr > 0).float()
            # Trace of the post-synaptic activity $\mathrm{tr}\Psi(y_i[t])$, second term of the RHS in equation (10)
            psi = 1 / torch.pow(100 * torch.abs(u_thr) + 1, 2)
            trace_out_next = factors[0] * trace_out + psi
        ctx.save_for_backward(input, weight, bias, trace_in, trace_out, u_thr, threshold, factors)
        return output, mem, trace_in, trace_out_next

    @staticmethod
    def backward(ctx, grad_output, grad_mem, gr, gg):
        input, weight, bias, trace_in, trace_out, u_thr, threshold, factors = ctx.saved_tensors
        psi = 1/torch.pow(100*torch.abs(u_thr)+1, 2)
        grad = psi*grad_output

        grad_input = torch.mm(grad, weight)
        grad_weight = factors[2] * torch.matmul(grad_output.T * trace_out.T, input) + factors[3] * torch.matmul(
            grad_output.T * psi.T, trace_in)
        # delta_w_pre = factors[2]*trace_out.unsqueeze(2) * input.unsqueeze(1)
        # delta_w_post = factors[3]*psi.unsqueeze(2) * trace_in.unsqueeze(1)
        #
        # grad_weight = (grad_output.unsqueeze(2)*(delta_w_post + delta_w_pre)).sum(0)
        grad_bias = None
        if bias is not None:
            grad_bias = grad.sum(dim=0)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class STLLRLinearRecGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, input_rec, weight, weight_rec, bias, trace_in, trace_in_rec, trace_out, mem, leak, threshold, factors):

        with torch.no_grad():
            # Trace of the pre-synaptic activity $\mathrm{tr}x_j[t]$, first term of the RHS in equation (10) for the
            # feedforward connections
            trace_in = factors[1] * trace_in + input
            # LIF computations
            output = F.linear(input, weight, bias)
            mem = torch.sigmoid(leak) * mem + output + F.linear(input_rec, weight_rec, None)
            u_thr = mem - threshold.clamp(min=0.5)
            output = (u_thr > 0).float()
            # Trace of the pre-synaptic activity $\mathrm{tr}x_j[t]$, first term of the RHS in equation (10) for the
            # recurrent connections
            trace_in_rec_next = factors[1] * trace_in_rec + output
            # Trace of the post-synaptic activity $\mathrm{tr}\Psi(y_i[t])$, second term of the RHS in equation (10)
            psi = 1 / torch.pow(100 * torch.abs(u_thr) + 1, 2)
            trace_out_next = factors[0] * trace_out + psi
        ctx.save_for_backward(input, input_rec, weight, bias, trace_in, trace_in_rec, trace_out, u_thr, threshold, factors)
        return output, mem, trace_in, trace_in_rec_next, trace_out_next

    @staticmethod
    def backward(ctx, grad_output, grad_mem, grad_trace_in, grad_trace_rec, grad_trace_out):
        input, input_rec, weight, bias, trace_in, trace_in_rec, trace_out, u_thr, threshold, factors = ctx.saved_tensors

        psi = 1 / torch.pow(100 * torch.abs(u_thr) + 1, 2)
        grad = psi*grad_output

        grad_input = torch.mm(grad, weight)

        # weight update for the feedforward connections
        delta_w_pre = factors[2]*trace_out.unsqueeze(2) * input.unsqueeze(1)
        delta_w_post = factors[3]*psi.unsqueeze(2) * trace_in.unsqueeze(1)
        grad_weight = (grad_output.unsqueeze(2)*(delta_w_post + delta_w_pre)).sum(0)

        # weight update for the recurrent connections

        delta_w_pre_rec = factors[2] * trace_out.unsqueeze(2) * input_rec.unsqueeze(1)
        delta_w_post_rec = factors[3] * psi.unsqueeze(2) * trace_in_rec.unsqueeze(1)
        grad_weight_rec = (grad_output.unsqueeze(2) * (delta_w_post_rec + delta_w_pre_rec)).sum(0)
        grad_bias = None
        if bias is not None:
            grad_bias = grad.sum(dim=0)
        return grad_input, None, grad_weight, grad_weight_rec, grad_bias, None, None, None, None, None, None, None


class STLLRConv2dGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, trace_in, trace_out, mem, leak, threshold, factors):
        with torch.no_grad():
            # Trace of the pre-synaptic activity $\mathrm{tr}x_j[t]$, first term of the RHS in equation (10)
            trace_in = factors[1] * trace_in + input
            # LIF computations
            output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
            mem = torch.sigmoid(leak) * mem + output
            u_thr = mem - threshold.clamp(min=0.5)
            out = (u_thr > 0).float()
            # Trace of the post-synaptic activity $\mathrm{tr}\Psi(y_i[t])$, second term of the RHS in equation (10)
            psi = 0.3 * F.threshold(1.0 - torch.abs(u_thr), 0, 0)
            trace_out_next = factors[0] * trace_out + psi
        ctx.save_for_backward(input, weight, bias, trace_in, trace_out, psi, threshold, factors)
        ctx.in1 = [stride, padding, dilation, groups]
        return out, mem, trace_in, trace_out_next

    @staticmethod
    def backward(ctx, grad_output, grad_mem, grad_trace_in, grad_trace_out):
        input, weight, bias, trace_in, trace_out, psi, threshold, factors = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.in1

        grad = psi * grad_output
        # print("Grad Begin")
        # print(input.size())
        # print(grad.size())
        # print(weight.size())
        # print("Grad End")
        grad_input = torch.nn.grad.conv2d_input(input.size(), weight, grad, stride, padding, dilation, groups)

        delta_weight_pre = torch.nn.grad.conv2d_weight(trace_in, weight.size(),  grad_output * psi, stride, padding,
                                                       dilation, groups)
        delta_weight_post = torch.nn.grad.conv2d_weight(input, weight.size(),
                                                      grad_output * trace_out, stride, padding,
                                                      dilation, groups)

        # The following line implements the weights updates
        # for the current layer (i.e. equations (10) and (11) on the paper)):
        # $\Delta w = \alpha_{post} \times (non-causal term) + \alpha_{pre} \times (causal term)$
        grad_weight = factors[2] * delta_weight_post + factors[3] * delta_weight_pre

        grad_bias = None
        if bias is not None:
            grad_bias = grad.sum(dim=(0, 2, 3))
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None