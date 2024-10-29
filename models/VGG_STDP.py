import torch
import torch.nn as nn
from models.layers import LinearSTLLR, Conv2dSTLLR, DropoutLIF
import models.layers.surrogate_gradients as gradients
from models.layers.feedback_alignment import TrainingHook
import logging
__all__ = ["cifar_vgg_stllr", "dvs_vgg_stllr", "dvscifar10_vgg_stllr", "nmnist_vgg_stllr", "ncaltech101_vgg_stllr"]


class VGG(nn.Module):
    def __init__(self, n_inputs: int = 3, labels: int = 10, activation=None, acc_activation=None,
                 feedback_mode: str = 'BP', DFA_size=None, factors=None, pool=None, dropout=0.0, gp=1):
        super(VGG, self).__init__()
        self.labels = labels
        if pool == "AVG":
            self.pool_layer = nn.AvgPool2d
        else:
            self.pool_layer = nn.MaxPool2d

        if DFA_size is None:
            self.DFA_size = [[labels, 64, 32, 32],
                             [labels, 128, 16, 16],
                             [labels, 256, 16, 16],
                             [labels, 256, 8, 8],
                             [labels, 512, 8, 8],
                             [labels, 512, 4, 4],
                             [labels, 512, 4, 4],
                             [labels, 512, 4, 4]
                             ]
        else:
            self.DFA_size = DFA_size
        # Feedback mode BP, DFA, sDFA
        self.feedback_mode = feedback_mode
        if (feedback_mode == "DFA") or (feedback_mode == "sDFA"):
            self.y = torch.zeros(1, labels)
            self.y.requires_grad = False

        else:
            self.y = None

        self.conv1 = Conv2dSTLLR(n_inputs, 64, 3, 1, 1, bias=True, activation=activation, factors=factors, leak=0)
        self.dropout1 = DropoutLIF(dropout)
        self.conv1_hook = TrainingHook(labels, dim_hook=self.DFA_size[0], feedback_mode=feedback_mode)

        self.conv2 = Conv2dSTLLR(64, 128, 3, 1, 1, bias=True, activation=activation, factors=factors, leak=0)
        self.dropout2 = DropoutLIF(dropout)
        self.pool1 = self.pool_layer(2, 2)
        self.conv2_hook = TrainingHook(labels, dim_hook=self.DFA_size[1], feedback_mode=feedback_mode)

        self.conv3 = Conv2dSTLLR(128, 256, 3, 1, 1, bias=True, activation=activation, factors=factors, leak=0)
        self.dropout3 = DropoutLIF(dropout)
        self.conv3_hook = TrainingHook(labels, dim_hook=self.DFA_size[2], feedback_mode=feedback_mode)

        self.conv4 = Conv2dSTLLR(256, 256, 3, 1, 1, bias=True, activation=activation, factors=factors, leak=0)
        self.dropout4 = DropoutLIF(dropout)
        self.pool2 = self.pool_layer(2, 2)
        self.conv4_hook = TrainingHook(labels, dim_hook=self.DFA_size[3], feedback_mode=feedback_mode)

        self.conv8 = Conv2dSTLLR(256, 512, 3, 1, 1, bias=True, activation=activation, factors=factors, leak=0)
        self.dropout5 = DropoutLIF(dropout)
        self.conv8_hook = TrainingHook(labels, dim_hook=self.DFA_size[4], feedback_mode=feedback_mode)

        self.conv9 = Conv2dSTLLR(512, 512, 3, 1, 1, bias=True,  activation=activation, factors=factors, leak=0)
        self.dropout6 = DropoutLIF(dropout)
        self.pool4 = self.pool_layer(2, 2)
        self.conv9_hook = TrainingHook(labels, dim_hook=self.DFA_size[5], feedback_mode=feedback_mode)

        self.conv11 = Conv2dSTLLR(512, 512, 3, 1, 1, bias=True, activation=activation, factors=factors, leak=0)
        self.dropout7 = DropoutLIF(dropout)
        self.conv11_hook = TrainingHook(labels, dim_hook=self.DFA_size[6], feedback_mode=feedback_mode)
        self.conv12 = Conv2dSTLLR(512, 512, 3, 1, 1, bias=True, activation=activation, factors=factors, leak=0)
        self.dropout8 = DropoutLIF(dropout)
        self.conv12_hook = TrainingHook(labels, dim_hook=self.DFA_size[7], feedback_mode=feedback_mode)

        self.globalpool = nn.AdaptiveAvgPool2d((gp, gp))

        self.linear16 = nn.Linear(512*gp*gp, labels, bias=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, Conv2dSTLLR):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, LinearSTLLR):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def reset_states(self):
        self.conv1.reset_state()
        self.conv2.reset_state()
        self.conv3.reset_state()
        self.conv4.reset_state()

        self.conv8.reset_state()
        self.conv9.reset_state()
        self.conv11.reset_state()
        self.conv12.reset_state()

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
        self.update_batch_size(x)

        training = self.training
        batch_size = x.shape[0]

        x1 = self.conv1(x)
        x1 = self.dropout1(x1)
        x1 = self.conv1_hook(x1, target, self.y)

        x1 = self.conv2(x1)
        x1 = self.dropout2(x1)
        x1 = self.pool1(x1)
        x1 = self.conv2_hook(x1, target, self.y)

        x1 = self.conv3(x1)
        x1 = self.dropout3(x1)
        x1 = self.conv3_hook(x1, target, self.y)

        x1 = self.conv4(x1)
        x1 = self.dropout4(x1)
        x1 = self.pool2(x1)
        x1 = self.conv4_hook(x1, target, self.y)

        x1 = self.conv8(x1)
        x1 = self.dropout5(x1)
        x1 = self.conv8_hook(x1, target, self.y)

        x1 = self.conv9(x1)
        x1 = self.dropout6(x1)
        x1 = self.pool4(x1)
        x1 = self.conv9_hook(x1, target, self.y)

        x1 = self.conv11(x1)
        x1 = self.dropout7(x1)
        x1 = self.conv11_hook(x1, target, self.y)

        x1 = self.conv12(x1)
        x1 = self.dropout8(x1)
        x1 = self.conv12_hook(x1, target, self.y)

        x1 = self.globalpool(x1).view(batch_size, -1)
        x1 = self.linear16(x1)

        if x1.requires_grad and (self.y is not None):
            x2 = torch.softmax(x1, dim=1)
            self.y.data.copy_(x2.data)
        # x1 = torch.softmax(x1, dim=1)
        return x1


def cifar_vgg_stllr(args, device):
    if args.activation != "LinearSpike":
        activation = gradients.__dict__[args.activation]
    else:
        activation = None
    act = None
    acc_act = None
    factors = args.factors_stdp

    model = VGG(n_inputs=3, labels=10 if args.dataset == "CIFAR10" else 100, activation=activation, acc_activation=acc_act,
                feedback_mode="BP" if "BP" in args.feedback_mode else args.feedback_mode, factors=factors, pool=args.pooling)
    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'], strict=False)
    return model


def dvs_vgg_stllr(args, device):
    if args.activation != "LinearSpike":
        activation = gradients.__dict__[args.activation]
        logging.info("Activation used: "+args.activation)
    else:
        activation = None
        logging.info("Activation used: None")
    acc_act = None

    factors = args.factors_stdp

    model = VGG(n_inputs=2, labels=11, activation=activation, acc_activation=acc_act,
                feedback_mode="BP" if "BP" in args.feedback_mode else args.feedback_mode,
                factors=factors, pool=args.pooling)

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'], strict=False)
    return model


def nmnist_vgg_stllr(args, device):
    act = None
    acc_act = None

    factors = args.factors_stdp

    labels = 10
    DFA_size = [[labels, 64, 34, 34],
                [labels, 128, 17, 17],
                [labels, 256, 17, 17],
                [labels, 256, 8, 8],
                [labels, 512, 8, 8],
                [labels, 512, 4, 4],
                [labels, 512, 4, 4],
                [labels, 512, 4, 4]
                ]

    model = VGG(n_inputs=2, labels=10, activation=act, acc_activation=acc_act,
                feedback_mode="BP" if "BP" in args.feedback_mode else args.feedback_mode,
                DFA_size=DFA_size, factors=factors, pool=args.pooling)

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'], strict=False)
    return model


def dvscifar10_vgg_stllr(args, device):
    if args.activation != "LinearSpike":
        activation = gradients.__dict__[args.activation]
        logging.info("Activation used: "+args.activation)
    else:
        activation = None
        logging.info("Activation used: None")
    labels = 10
    DFA_size = [[labels, 64, 48, 48],
                [labels, 128, 24, 24],
                [labels, 256, 24, 24],
                [labels, 256, 12, 12],
                [labels, 512, 12, 12],
                [labels, 512, 6, 6],
                [labels, 512, 6, 6],
                [labels, 512, 6, 6]
                ]

    acc_act = None

    factors = args.factors_stdp

    model = VGG(n_inputs=2, labels=10, activation=activation, acc_activation=acc_act,
                feedback_mode="BP" if "BP" in args.feedback_mode else args.feedback_mode,
                factors=factors, pool=args.pooling, dropout=0.1, gp=1, DFA_size=DFA_size)

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'], strict=False)
    return model


def ncaltech101_vgg_stllr(args, device):
    if args.activation != "LinearSpike":
        activation = gradients.__dict__[args.activation]
        logging.info("Activation used: "+args.activation)
    else:
        activation = None
        logging.info("Activation used: None")

    acc_act = None

    factors = args.factors_stdp

    model = VGG(n_inputs=2, labels=101, activation=activation, acc_activation=acc_act,
                feedback_mode="BP" if "BP" in args.feedback_mode else args.feedback_mode,
                factors=factors, pool=args.pooling, dropout=0.1, gp=1)

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'], strict=False)
    return model