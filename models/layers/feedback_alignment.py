"""
This code was adapted from DFA implementation on:
C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback: Fixed Random Learning Signals Allow for
Feedforward Training of Deep Neural Networks," Frontiers in Neuroscience, vol. 15, no. 629892, 2021.
doi: 10.3389/fnins.2021.629892
"""

import torch
import torch.nn as nn
from numpy import prod


class HookFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, labels, y, fixed_fb_weights, feedback_mode):
        if feedback_mode in ["DFA", "sDFA", "DRTP", "LocalLoss"]:
            ctx.save_for_backward(input, labels, y, fixed_fb_weights)
        ctx.in1 = feedback_mode
        return input

    @staticmethod
    def backward(ctx, grad_output):
        feedback_mode = ctx.in1
        if feedback_mode == "BP":
            return grad_output, None, None, None, None
        elif feedback_mode == "shallow":
            grad_output.data.zero_()
            return grad_output, None, None, None, None

        input, labels, y, fixed_fb_weights = ctx.saved_variables
        grad_fb = None
        if feedback_mode == "DFA":
            grad_output_est = (y-labels).mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape)
        elif feedback_mode == "sDFA":
            grad_output_est = torch.sign(y-labels).mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape)
        elif feedback_mode == "DRTP":
            grad_output_est = labels.mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape)
        elif feedback_mode == "LocalLoss":
            batch_size = input.size(0)
            logits = torch.matmul(input.view(batch_size, -1), fixed_fb_weights.view(-1, prod(fixed_fb_weights.shape[1:])).T)
            error_fb = torch.softmax(logits, dim=1) - labels
            grad_output_est = (error_fb).mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape)
            grad_fb = (error_fb.T).mm(input.view(batch_size, -1)).view(fixed_fb_weights.shape)
        else:
            raise NameError("=== ERROR: training mode " + str(feedback_mode) + " not supported")
        return grad_output_est, None, None, grad_fb, None


class FA_wrapper(nn.Module):
    def __init__(self, module, layer_type, dim, stride=None, padding=None):
        super(FA_wrapper, self).__init__()
        self.module = module
        self.layer_type = layer_type
        self.stride = stride
        self.padding = padding
        self.output_grad = None
        self.x_shape = None

        # FA feedback weights definition
        self.fixed_fb_weights = nn.Parameter(torch.Tensor(torch.Size(dim)))
        self.reset_weights()

    def forward(self, x):
        if x.requires_grad:
            x.register_hook(self.FA_hook_pre)
            self.x_shape = x.shape
            x = self.module(x)
            x.register_hook(self.FA_hook_post)
            return x
        else:
            return self.module(x)

    def reset_weights(self):
        torch.nn.init.kaiming_uniform_(self.fixed_fb_weights)
        self.fixed_fb_weights.requires_grad = False

    def FA_hook_pre(self, grad):
        if self.output_grad is not None:
            if (self.layer_type == "fc"):
                return self.output_grad.mm(self.fixed_fb_weights)
            elif (self.layer_type == "conv"):
                return torch.nn.grad.conv2d_input(self.x_shape, self.fixed_fb_weights, self.output_grad, self.stride, self.padding)
            else:
                raise NameError("=== ERROR: layer type " + str(self.layer_type) + " is not supported in FA wrapper")
        else:
            return grad

    def FA_hook_post(self, grad):
        self.output_grad = grad
        return grad


class TrainingHook(nn.Module):
    def __init__(self, label_features, dim_hook, feedback_mode):
        super(TrainingHook, self).__init__()
        self.feedback_mode = feedback_mode
        assert feedback_mode in ["BP", "FA", "DFA", "DRTP", "sDFA", "shallow", "LocalLoss"], "=== ERROR: Unsupported hook training mode " + feedback_mode + "."

        # Feedback weights definition (FA feedback weights are handled in the FA_wrapper class)
        if self.feedback_mode in ["DFA", "DRTP", "sDFA", "LocalLoss"]:
            self.fixed_fb_weights = nn.Parameter(torch.Tensor(torch.Size(dim_hook)))
            self.reset_weights()
        else:
            self.fixed_fb_weights = None

        self.training_hook = HookFunction.apply

    def reset_weights(self):
        torch.nn.init.kaiming_uniform_(self.fixed_fb_weights)
        if self.feedback_mode == "LocalLoss":
            self.fixed_fb_weights.requires_grad = True
        else:
            self.fixed_fb_weights.requires_grad = False

    def forward(self, input, labels, y):
        if type(input) is list:
            x1 = self.training_hook(input[0], labels, y, self.fixed_fb_weights, self.feedback_mode if (self.feedback_mode != "FA") else "BP")
            x2 = self.training_hook(input[1], labels, y, self.fixed_fb_weights, self.feedback_mode if (self.feedback_mode != "FA") else "BP") #FA is handled in FA_wrapper, not in TrainingHook
            return (x1, x2)
        else:
            return self.training_hook(input, labels, y, self.fixed_fb_weights, self.feedback_mode if (
                        self.feedback_mode != "FA") else "BP")  # FA is handled in FA_wrapper, not in TrainingHook

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.feedback_mode + ')'