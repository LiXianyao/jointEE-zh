import torch
import torch.nn as nn
import torch.nn.functional as F
from util.util import *
from util.consts import VISIABLE

class InceptionCNN(nn.Module):
    def __init__(self, hyps, in_channels=1, out_channels=300, kernel_dim=400, inception_mode=1):
        # checked
        super(InceptionCNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_dim = kernel_dim  # 每个核的宽度（wemb_dim + char_cnn_filter + pemb_dim)
        self.inception_mode = inception_mode
        self.construct_layer(torch.nn.BatchNorm2d, self.get_CNN2d)
        self.dropout = nn.Dropout(p=hyps["inception_dropout"])

    def construct_layer(self, BatchNorm, get_CNN):
        if self.inception_mode == 1:
            self.bn_1 = BatchNorm(self.out_channels)
            self.bn_3 = BatchNorm(self.out_channels)
            self.bn_5 = BatchNorm(self.out_channels)
            self.conv_1 = get_CNN(window_size=1)
            self.conv_3 = get_CNN(window_size=3)
            self.conv_5 = get_CNN(window_size=5)
        elif self.inception_mode == 2:
            self.bn_1 = BatchNorm(self.out_channels)
            self.bn_3 = BatchNorm(self.out_channels)
            self.conv_1 = get_CNN(window_size=1)
            self.conv_3 = get_CNN(window_size=3)

    def get_CNN2d(self, window_size):
        conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                        kernel_size=(window_size, self.kernel_dim), padding=(window_size // 2, 0), stride=1)
        init_cnn_(conv)
        #self.sizeof("cnn {}".format(window_size), conv.weight)
        return conv

    def forward(self, input, pooling_height=1):
        # checked
        if self.inception_mode == 0:
            return input

        batch_size, max_seq, word_embed_size = input.size()

        # batch x 1 x max_seq x word_embed
        input_ = input.unsqueeze(1)

        if self.inception_mode == 1:
            # batch x out x max_seq x 1
            conv_1 = self.bn_1(self.conv_1(input_))
            conv_3 = self.bn_3(self.conv_3(input_))
            conv_5 = self.bn_5(self.conv_5(input_))
            #conv_1 = self.conv_1(input_)
            #conv_3 = self.conv_3(input_)
            #conv_5 = self.conv_5(input_)
            input_1 = torch.tanh(conv_1)
            input_3 = torch.tanh(conv_3)
            input_5 = torch.tanh(conv_5)

            # batch x out_channels x max_seq x 3
            pooling_input = torch.cat([input_1, input_3, input_5], 3)
            # batch x out_channels x max_seq x 1
            output = F.max_pool2d(pooling_input, kernel_size=(pooling_height, pooling_input.size(3)))
        elif self.inception_mode == 2:
            conv_1 = self.bn_1(self.conv_1(input_))
            conv_3 = self.bn_3(self.conv_3(input_))
            input_1 = torch.tanh(conv_1)
            input_3 = torch.tanh(conv_3)
            pooling_input = torch.cat([input_1, input_3], 3)
            # batch x out_channels x max_seq x 1
            output = F.max_pool2d(pooling_input, kernel_size=(pooling_height, pooling_input.size(3)))

        # batch x out x max_seq -> batch x max_seq x out
        output = output.squeeze(3).transpose(1, 2)
        output = self.dropout(output)
        #assert output.size() == input.size()
        return output

    def sizeof(self, name, tensor):
        if not VISIABLE: return
        log("shape of tensor '{}' is {} ".format(name, tensor.size()))