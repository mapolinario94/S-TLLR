import torch
import torch.nn as nn
from models.layers import LinearSTLLR, Conv2dSTLLR, DropoutLIF
from models.layers.custom_layers import ScaledWSLinear, ScaledWSConv2d
from models.layers.spiking_layers import BPTTLIF
import models.layers.surrogate_gradients as gradients
from models.layers.feedback_alignment import TrainingHook
from collections import OrderedDict
import logging
__all__ = ["dvscifar10_resnet18_bptt", "dvs_resnet18_bptt", "ncaltech101_resnet18_bptt"]


class BasicBlock(nn.Module):
    """
        Quantized ResNet block with residual path.
    """
    def __init__(self, in_channels, out_channels, stride, g_function="ADD", activation=None, factors=None, dropout=0.1):
        super(BasicBlock, self).__init__()
        self.g_function = g_function
        # self.norm = nn.BatchNorm2d
        if in_channels != out_channels or stride>1:
            self.resize_identity = True
        else:
            self.resize_identity = False

        conv1 = nn.Sequential(OrderedDict([
            ("conv", ScaledWSConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                 bias=True)),

             ]))

        lif1 = BPTTLIF(activation=activation, leak=0.0)

        conv2 = nn.Sequential(OrderedDict([
            ("conv", ScaledWSConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                 bias=True)),
        ]))

        lif2 = BPTTLIF(activation=activation, leak=0.0)
        # dropout_layer = DropoutLIF(dropout)

        self.body = nn.Sequential(OrderedDict([
            ("conv1", conv1),
            ("lif1", lif1),
            # ("dropout", dropout_layer),
            ("conv2", conv2),
            ("lif2", lif2),
        ]))

        if self.resize_identity:
            self.identity_conv = nn.Sequential(OrderedDict([
                ("conv", ScaledWSConv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                     padding=0, bias=True)),
                ("lif", BPTTLIF(activation=activation, leak=0.0))
            ]))

    def init_layer(self):
        # self.activ.reset_state()
        self.body.lif1.reset_state()
        self.body.lif2.reset_state()
        # self.body.dropout.reset_state()
        if self.resize_identity:
            self.identity_conv.lif.reset_state()

    def forward(self, x):
        # forward using the quantized modules
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x

        x = self.body(x)
        # print(x.size())
        # if self.g_function == "AND":
        #     x = x*identity
        # elif self.g_function == "IAND":
        #     x = (1-x)*identity
        # else:
        x = x + identity

        return x


class ResNet18(nn.Module):
    """
        Quantized ResNet50 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    """

    def __init__(self,  in_channels=3, g_function="ADD", labels=100, factors=None, activation=None):
        super().__init__()
        # features = getattr(model, 'features')
        # init_block = getattr(features, 'init_block')
        # self.activation = None
        self.factors = factors
        self.in_channels = in_channels
        # For SNN parameters only
        self.g_function = g_function
        # ------------
        # self.norm = nn.BatchNorm2d
        self.activation = activation
        self.channel = [2, 2, 2, 2]
        self.channels_n = [64, 128, 256, 512]
        # self.channels_n = [32*2, 48*2, 96*2, 128*2]
        self.stride_layer = [1, 2, 2, 2]
        self.features = self._make_layers()

        self.population = 1

        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.final_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        # linear = nn.Linear(1024, 10, bias=False)
        self.output = nn.Linear(512, labels*self.population, bias=True)

        # self.dropout = nn.Dropout(p=0.1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                pass
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # if zero_init_residual:
        #     zero_init_blocks(self, g_function)

    def _make_layers(self):
        features_dict = OrderedDict()

        sew_init_block = nn.Sequential(OrderedDict([
            ("conv", ScaledWSConv2d(self.in_channels, self.channels_n[0], kernel_size=3, stride=1, padding=1, bias=True)),
            ("lif", BPTTLIF(activation=self.activation, leak=0.0)),
            # ("bn", nn.BatchNorm2d(self.channels_n[0])),
            # ("activ", LIF(learnable_tl=self.learnable_tl, activation=self.activation)),
            ("init_pool", nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        features_dict["init_block"] = sew_init_block

        channels = self.channels_n[0]

        for stage_num in range(0, len(self.channel)):
            # stage = getattr(features, "stage{}".format(stage_num + 1))
            unit_dict = OrderedDict()
            identity_flag = False
            for unit_num in range(0, self.channel[stage_num]):
                stride = self.stride_layer[stage_num]
                # unit = getattr(stage, "unit{}".format(unit_num + 1))
                sew_unit = BasicBlock(channels,
                                      self.channels_n[stage_num],
                                      stride=stride if not identity_flag else 1,
                                      g_function=self.g_function,
                                      activation=self.activation,
                                      factors=self.factors)
                channels = self.channels_n[stage_num]
                unit_dict["unit{}".format(unit_num + 1)] = sew_unit
                identity_flag = True
            unit_seq = nn.Sequential(unit_dict)
            features_dict["stage{}".format(stage_num + 1)] = unit_seq

        return nn.Sequential(features_dict)

    def reset_states(self):
        self.features.init_block.lif.reset_state()
        # self.lif_output.reset_state()
        for stage_num in range(0, len(self.channel)):
            stage = getattr(self.features, f"stage{stage_num + 1}")
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(stage, f"unit{unit_num + 1}")
                tmp_func.init_layer()

    def forward(self, input, targets=None):
        # self._init_layers()
        out = 0
        # for t in range(self.time_steps):
        x = self.features(input)
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)

        x = self.output(x)
        return x


def dvs_resnet18_bptt(args, device):
    if args.activation != "LinearSpike":
        activation = gradients.__dict__[args.activation]
        logging.info("Activation used: "+args.activation)
    else:
        activation = None
        logging.info("Activation used: None")

    acc_act = None

    factors = args.factors_stdp

    model = ResNet18(in_channels=2, labels=11, activation=activation, factors=factors)

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'], strict=False)
    return model


def dvscifar10_resnet18_bptt(args, device):
    if args.activation != "LinearSpike":
        activation = gradients.__dict__[args.activation]
        logging.info("Activation used: "+args.activation)
    else:
        activation = None
        logging.info("Activation used: None")

    acc_act = None

    factors = args.factors_stdp

    model = ResNet18(in_channels=2, labels=10, activation=activation, factors=factors)

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'], strict=False)
    return model


def ncaltech101_resnet18_bptt(args, device):
    if args.activation != "LinearSpike":
        activation = gradients.__dict__[args.activation]
        logging.info("Activation used: "+args.activation)
    else:
        activation = None
        logging.info("Activation used: None")

    acc_act = None

    factors = args.factors_stdp

    model = ResNet18(in_channels=2, labels=101, activation=activation, factors=factors)

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'], strict=False)
    return model