import torch
import torch.nn as nn
from models.layers.spiking_layers import BPTTLIF
import models.layers.surrogate_gradients as gradients
from models.layers.custom_layers import ScaledWSLinear, ScaledWSConv2d
from models.layers import DropoutLIF
import logging
__all__ = ["dvs_vgg_bptt", "dvscifar10_vgg_bptt", "ncaltech101_vgg_bptt"]


class VGG(nn.Module):
    def __init__(self, n_inputs: int = 3, labels: int = 10, activation=None, feedback_mode: str = 'BP', dropout=0.0):
        super(VGG, self).__init__()
        self.labels = labels

        conv_layer_init = ScaledWSConv2d
        conv_layer = ScaledWSConv2d
        linear_layer = ScaledWSLinear

        # Layers
        self.conv1 = conv_layer(n_inputs, 64, 3, 1, 1, bias=True)
        self.lif1 = BPTTLIF(activation=activation, leak=0.0)
        self.dropout1 = DropoutLIF(dropout)

        self.conv2 = conv_layer(64, 128, 3, 1, 1, bias=True)
        self.lif2 = BPTTLIF(activation=activation, leak=0.0)  # 16x16x128
        self.pool1 = nn.AvgPool2d(2, 2)

        self.dropout2 = DropoutLIF(dropout)

        self.conv3 = conv_layer(128, 256, 3, 1, 1, bias=True)
        self.lif3 = BPTTLIF(activation=activation, leak=0.0)
        self.dropout3 = DropoutLIF(dropout)

        self.conv4 = conv_layer(256, 256, 3, 1, 1, bias=True)

        self.lif4 = BPTTLIF(activation=activation, leak=0.0) # 8x8x256
        self.pool2 = nn.AvgPool2d(2, 2)
        self.dropout4 = DropoutLIF(dropout)

        self.conv8 = conv_layer(256, 512, 3, 1, 1, bias=True)
        self.lif8 = BPTTLIF(activation=activation, leak=0.0)
        self.dropout5 = DropoutLIF(dropout)

        self.conv9 = conv_layer(512, 512, 3, 1, 1, bias=True)
        self.pool4 = nn.AvgPool2d(2, 2)
        self.lif9 = BPTTLIF(activation=activation, leak=0.0) # 4x4x512
        self.dropout6 = DropoutLIF(dropout)

        self.conv11 = conv_layer(512, 512, 3, 1, 1, bias=True)
        self.lif11 = BPTTLIF(activation=activation, leak=0.0)

        self.dropout7 = DropoutLIF(dropout)
        self.conv12 = conv_layer(512, 512, 3, 1, 1, bias=True)
        self.lif12 = BPTTLIF(activation=activation, leak=0.0) # 4x4x512

        self.dropout8 = DropoutLIF(dropout)
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear16 = nn.Linear(512, labels, bias=True)

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
                # nn.init.constant_(m.bias, 0)

    def reset_states(self):
        #self.trace_input.reset_state()
        self.lif1.reset_state()
        self.lif2.reset_state()
        self.lif3.reset_state()
        self.lif4.reset_state()

        self.lif8.reset_state()
        self.lif9.reset_state()
        self.lif11.reset_state()
        self.lif12.reset_state()
        self.dropout1.reset_state()
        self.dropout2.reset_state()
        self.dropout3.reset_state()
        self.dropout4.reset_state()
        self.dropout5.reset_state()
        self.dropout6.reset_state()
        self.dropout7.reset_state()
        self.dropout8.reset_state()

    def update_batch_size(self, x:torch.Tensor):
        if (self.feedback_mode == "DFA") or (self.feedback_mode == "sDFA"):
            self.y = torch.zeros(x.shape[0], self.labels, device=x.device)
            self.y.requires_grad = False
        else:
            self.y = None

    def forward(self, x, target=None):

        #self.update_batch_size(x)

        training = self.training
        batch_size = x.shape[0]
        x1 = self.conv1(x)
        x1 = self.lif1(x1)
        x1 = self.dropout1(x1)


        x1 = self.conv2(x1)
        x1 = self.lif2(x1)
        x1 = self.dropout2(x1)
        x1 = self.pool1(x1)


        x1 = self.conv3(x1)
        x1 = self.lif3(x1)
        x1 = self.dropout3(x1)
        x1 = self.conv4(x1)
        x1 = self.lif4(x1)
        x1 = self.dropout4(x1)
        x1 = self.pool2(x1)


        x1 = self.conv8(x1)
        x1 = self.lif8(x1)
        x1 = self.dropout5(x1)
        x1 = self.conv9(x1)
        x1 = self.lif9(x1)
        x1 = self.dropout6(x1)
        x1 = self.pool4(x1)


        x1 = self.conv11(x1)
        x1 = self.lif11(x1)
        x1 = self.dropout7(x1)
        x1 = self.conv12(x1)
        x1 = self.lif12(x1)
        x1 = self.dropout8(x1)

        x1 = self.globalpool(x1).view(batch_size, -1)
        x1 = self.linear16(x1)

        return x1


def ncaltech101_vgg_bptt(args, device):
    activation = gradients.__dict__[args.activation]

    model = VGG(n_inputs=2, labels=101, activation=activation, feedback_mode=args.feedback_mode, dropout=0.0)

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'], strict=False)
    return model


def dvs_vgg_bptt(args, device):
    activation = gradients.__dict__[args.activation]

    model = VGG(n_inputs=2, labels=11, activation=activation, feedback_mode=args.feedback_mode, dropout=0.0)

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'], strict=False)
    return model


def dvscifar10_vgg_bptt(args, device):
    activation = gradients.__dict__[args.activation]


    model = VGG(n_inputs=2, labels=10, activation=activation, feedback_mode=args.feedback_mode, dropout=0.1)

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'], strict=False)
    return model