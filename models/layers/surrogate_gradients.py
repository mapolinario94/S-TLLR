import torch
import torch.nn.functional as F
__all__ = ["Surrogate", "LinearSpike", "STLLRConv2dGradNormOut",
           "STLLRConv2dGradExp", "STLLRConv2dSigmoid", "GradSigmoid", "STLLRLinearRecGradNorm", "STLLRLinearRecGradSigmoid"]




class STLLRLinearRecGradSigmoid(torch.autograd.Function):

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
            sgax = (u_thr * 4).sigmoid_()
            psi = (1. - sgax) * sgax * 4
            trace_out_next = factors[0] * trace_out + psi
        ctx.save_for_backward(input, input_rec, weight, bias, trace_in, trace_in_rec, trace_out, u_thr, threshold, factors)
        return output, mem, trace_in, trace_in_rec_next, trace_out_next

    @staticmethod
    def backward(ctx, grad_output, grad_mem, grad_trace_in, grad_trace_rec, grad_trace_out):
        input, input_rec, weight, bias, trace_in, trace_in_rec, trace_out, u_thr, threshold, factors = ctx.saved_tensors

        sgax = (u_thr * 4).sigmoid_()
        psi = (1. - sgax) * sgax * 4
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


class STLLRLinearRecGradNorm(torch.autograd.Function):

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
            # sgax = (u_thr * 4).sigmoid_()
            psi = 0.3 * F.threshold(1.0 - torch.abs(u_thr), 0, 0)
            trace_out_next = factors[0] * trace_out + psi
        ctx.save_for_backward(input, input_rec, weight, bias, trace_in, trace_in_rec, trace_out, u_thr, threshold, factors)
        return output, mem, trace_in, trace_in_rec_next, trace_out_next

    @staticmethod
    def backward(ctx, grad_output, grad_mem, grad_trace_in, grad_trace_rec, grad_trace_out):
        input, input_rec, weight, bias, trace_in, trace_in_rec, trace_out, u_thr, threshold, factors = ctx.saved_tensors

        # sgax = (u_thr * 4).sigmoid_()
        psi = 0.3 * F.threshold(1.0 - torch.abs(u_thr), 0, 0)
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






class GradSigmoid(torch.autograd.Function):
    """
    Surrogate gradient based on arctan, used in Feng et al. (2021)
    """
    gamma = 0.3
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        output = (x > 0).float()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        u_thr = ctx.saved_tensors[0]
        sgax = (u_thr * 4).sigmoid_()
        surrogate = (1. - sgax) * sgax * 4
        grad_x = surrogate * grad_output

        return grad_x, None


class Surrogate(torch.autograd.Function):
    """
    Surrogate gradient based on arctan, used in Feng et al. (2021)
    """
    gamma = 0.3
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        output = (x > 0.6).float()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        vmem = ctx.saved_tensors[0]
        thr = 0.6
        grad_x = Surrogate.gamma * torch.max(torch.zeros_like(vmem), 1 - torch.abs((vmem - thr) / thr)) * grad_output

        return grad_x, None


class SurrogateAudio(torch.autograd.Function):
    """
    Surrogate gradient based on arctan, used in Feng et al. (2021)
    """
    gamma = 0.3
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        output = (x > 0).float()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        u_thr = ctx.saved_tensors[0]
        surrogate = 1 / torch.pow(100 * torch.abs(u_thr) + 1, 2)
        # thr = 0.6
        grad_x = surrogate * grad_output

        return grad_x, None


class LinearSpike(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3  # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = (input_ > 0).float()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = LinearSpike.gamma * F.threshold(1.0 - torch.abs(input_), 0, 0)
        return grad * grad_input, None


# Additional implementations for STLLR convolutional layers
class STLLRConv2dGradNormOut(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, trace_in, trace_out, mem, leak, threshold, factors):
        with torch.no_grad():
            # Trace of the pre-synaptic activity $\mathrm{tr}x_j[t]$, first term of the RHS in equation (10)
            trace_in = factors[1] * trace_in + input
            # LIF computations
            output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
            mem = torch.sigmoid(leak) * mem + output
            u_thr = mem - threshold.clamp(min=0.5)
            output = (u_thr > 0).float()
            # Surrogate gradient
            surrogate = 0.3 * F.threshold(1.0 - torch.abs(u_thr), 0, 0)
            # Trace of the post-synaptic activity $\mathrm{tr}\Psi(y_i[t])$, second term of the RHS in equation (10)
            trace_out_next = factors[0] * trace_out + output
        ctx.save_for_backward(input, weight, bias, trace_in, trace_out, surrogate, threshold, factors, output)
        ctx.in1 = [stride, padding, dilation, groups]
        return output, mem, trace_in, trace_out_next

    @staticmethod
    def backward(ctx, grad_output, grad_mem, grad_trace_in, grad_trace_out):
        input, weight, bias, trace_in, trace_out, surrogate, threshold, factors, output= ctx.saved_tensors
        stride, padding, dilation, groups = ctx.in1

        grad = surrogate * grad_output

        grad_input = torch.nn.grad.conv2d_input(input.size(), weight, grad, stride, padding, dilation, groups)

        delta_weight_pre = torch.nn.grad.conv2d_weight(trace_in, weight.size(),  grad_output * output, stride, padding,
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


class STLLRConv2dGradExp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, trace_in, trace_out, mem, leak, threshold, factors):
        with torch.no_grad():
            # Trace of the pre-synaptic activity $\mathrm{tr}x_j[t]$, first term of the RHS in equation (10)
            trace_in = factors[1] * trace_in + input
            # LIF computations
            output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
            mem = torch.sigmoid(leak) * mem + output
            u_thr = mem - threshold.clamp(min=0.5)
            output = (u_thr > 0).float()
            # Trace of the post-synaptic activity $\mathrm{tr}\Psi(y_i[t])$, second term of the RHS in equation (10)
            # surrogate = 0.3 * F.threshold(1.0 - torch.abs(u_thr), 0, 0)
            psi = 1 / torch.pow(100 * torch.abs(u_thr) + 1, 2)
            trace_out_next = factors[0] * trace_out + psi
        ctx.save_for_backward(input, weight, bias, trace_in, trace_out, psi, threshold, factors, output)
        ctx.in1 = [stride, padding, dilation, groups]
        return output, mem, trace_in, trace_out_next

    @staticmethod
    def backward(ctx, grad_output, grad_mem, grad_trace_in, grad_trace_out):
        input, weight, bias, trace_in, trace_out, psi, threshold, factors, output= ctx.saved_tensors
        stride, padding, dilation, groups = ctx.in1

        grad = psi * grad_output

        grad_input = torch.nn.grad.conv2d_input(input.size(), weight, grad, stride, padding, dilation, groups)

        delta_weight_pre = torch.nn.grad.conv2d_weight(trace_in, weight.size(), grad_output * psi, stride, padding,
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


class STLLRConv2dSigmoid(torch.autograd.Function):
    alpha = 4
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, trace_in, trace_out, mem, leak, threshold, factors):
        with torch.no_grad():
            # Trace of the pre-synaptic activity $\mathrm{tr}x_j[t]$, first term of the RHS in equation (10)
            trace_in = factors[1] * trace_in + input
            # LIF computations
            output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
            mem = torch.sigmoid(leak) * mem + output
            u_thr = mem - threshold.clamp(min=0.5)
            output = (u_thr > 0).float()
            # Trace of the post-synaptic activity $\mathrm{tr}\Psi(y_i[t])$, second term of the RHS in equation (10)
            sgax = (u_thr * STLLRConv2dSigmoid.alpha).sigmoid_()
            psi = (1. - sgax) * sgax * STLLRConv2dSigmoid.alpha
            trace_out_next = factors[0] * trace_out + psi
        ctx.save_for_backward(input, weight, bias, trace_in, trace_out, psi, threshold, factors, output)
        ctx.in1 = [stride, padding, dilation, groups]
        return output, mem, trace_in, trace_out_next

    @staticmethod
    def backward(ctx, grad_output, grad_mem, grad_trace_in, grad_trace_out):
        input, weight, bias, trace_in, trace_out, psi, threshold, factors, output= ctx.saved_tensors
        stride, padding, dilation, groups = ctx.in1

        grad = psi * grad_output

        grad_input = torch.nn.grad.conv2d_input(input.size(), weight, grad, stride, padding, dilation, groups)

        delta_weight_pre = torch.nn.grad.conv2d_weight(trace_in, weight.size(), grad_output * psi, stride, padding,
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