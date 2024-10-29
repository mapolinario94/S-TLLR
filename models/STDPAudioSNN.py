import torch
import torch.nn as nn
from models.layers.custom_layers import ScaledWSLinear, ScaledWSConv2d
from models.layers.feedback_alignment import TrainingHook
from models.layers import LinearSTLLR, Conv2dSTLLR, LinearRecSTLLR, DropoutLIF
import models.layers.surrogate_gradients as gradients
import logging
__all__ = ["stllr_shd_net", "stllr_ssc_net"]


class AudioNet(nn.Module):
    def __init__(self, n_inputs: int = 700, labels: int = 20, activation=None, acc_activation=None,
                 feedback_mode: str = 'BP',  device=None, factors=None):
        super(AudioNet, self).__init__()
        self.labels = labels
        self.device = device

        # Feedback mode BP, DFA, sDFA
        self.feedback_mode = feedback_mode
        if (feedback_mode == "DFA") or (feedback_mode == "sDFA"):
            self.y = torch.zeros(1, labels)
            self.y.requires_grad = False
        else:
            self.y = None
        # Learn Leak and Threshold
        grad_lr = False
        net_size = 450
        # Layers

        self.linear1 = LinearRecSTLLR(n_inputs, net_size, bias=True, activation=activation, factors=factors, leak=5)
        self.linear1_hook = TrainingHook(labels, dim_hook=[labels, net_size], feedback_mode=feedback_mode)

        self.last_layer = LinearSTLLR(net_size, labels, bias=True, activation=acc_activation, accumulate=True, leak=5,
                                      factors=[0, 0.99, 0, 1])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, ScaledWSConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def reset_states(self):
        self.linear1.reset_state()
        # self.linear2.reset_state()
        self.last_layer.reset_state()
        # self.dropout.reset_state()

    def update_batch_size(self, x:torch.Tensor):
        if (self.feedback_mode == "DFA") or (self.feedback_mode == "sDFA"):
            self.y = torch.zeros(x.shape[0], self.labels, device=x.device)
            self.y.requires_grad = False
        else:
            self.y = None

    def forward(self, x, target=None):

        self.update_batch_size(x)

        training = self.training
        batch_size = x.shape[0]
        x = self.linear1(x)
        x = self.linear1_hook(x, target, self.y)

        logits = self.last_layer(x)

        if logits.requires_grad and (self.y is not None):
            classes = torch.softmax(logits, dim=1)
            self.y.data.copy_(classes.data)

        return logits


def stllr_shd_net(args, device):
    # activation = gradients.__dict__[args.activation]
    if args.activation != "LinearSpike":
        activation = gradients.__dict__[args.activation]
        logging.info("Activation used: " + args.activation)
    else:
        activation = None
        logging.info("Activation used: None")
    acc_act = None
    factors = args.factors_stdp
    model = AudioNet(n_inputs=700, labels=20, activation=activation, feedback_mode=args.feedback_mode, factors=factors)

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'], strict=False)
    return model


def stllr_ssc_net(args, device):
    # activation = gradients.__dict__[args.activation]
    act = None
    acc_act = None
    factors = args.factors_stdp

    model = AudioNet(n_inputs=700, labels=30, activation=act, feedback_mode=args.feedback_mode, device=device, factors=factors)

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'], strict=False)
    return model