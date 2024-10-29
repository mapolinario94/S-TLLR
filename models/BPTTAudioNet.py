import torch
import torch.nn as nn
from models.layers.spiking_layers import BPTTLIF
from models.layers.surrogate_gradients import SurrogateAudio
from models.layers.custom_layers import ScaledWSLinear, ScaledWSConv2d
import logging
__all__ = ["bptt_shd_net", "bptt_ssc_net"]


class BPTTAudioNet(nn.Module):
    def __init__(self, n_inputs: int = 700, labels: int = 20, activation=None, acc_activation=None, feedback_mode: str = 'BP', device=None, factors=None):
        super(BPTTAudioNet, self).__init__()
        self.labels = labels
        self.device = device

        # Learn Leak and Threshold
        grad_lr = False
        net_size = 450
        # Layers

        self.linear1 = ScaledWSLinear(n_inputs, net_size, bias=True)
        self.linear1_rec = ScaledWSLinear(net_size, net_size, bias=False)
        self.lif1 = BPTTLIF(leak=15, activation=SurrogateAudio)
        #self.dropout = DropoutLIF(0.5)
        self.last_layer = nn.Linear(net_size, labels, bias=True)  # best 0.5
        self.lif2 = BPTTLIF(accumulate=True, leak=5.0)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, ScaledWSConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear) or isinstance(m, ScaledWSLinear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def reset_states(self):
        self.lif1.reset_state()
        self.lif2.reset_state()
        self.first_step = True
        self.prev_state = 0

    def forward(self, x, target=None):

        if not self.first_step:
            x1 = self.linear1_rec(self.prev_state)
        else:
            x1 = 0
            self.first_step = False
        x = self.linear1(x)
        x = self.lif1(x + x1)
        self.prev_state = x.clone()
        x = self.last_layer(x)
        logits = self.lif2(x)

        return logits


def bptt_shd_net(args, device):
    # activation = gradients.__dict__[args.activation]
    act = None
    acc_act = None
    factors = args.factors_stdp
    model = BPTTAudioNet(n_inputs=700, labels=20, activation=act, feedback_mode=args.feedback_mode, factors=factors)

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'], strict=False)
    return model


def bptt_ssc_net(args, device):
    # activation = gradients.__dict__[args.activation]
    act = None
    acc_act = None
    factors = args.factors_stdp

    model = BPTTAudioNet(n_inputs=700, labels=30, activation=act, feedback_mode=args.feedback_mode, device=device, factors=factors)

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'], strict=False)
    return model