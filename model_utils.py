
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable
import math


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class EncoderPrenet(nn.Module):
    def __init__(self,dim, dropout = 0.1):
        super(EncoderPrenet,self).__init__()

        self.layer1 = LinearNorm(dim,dim)
        self.layer2 = LinearNorm(dim,dim)
        
        self.drop = nn.Dropout(dropout)


    def forward(self,x):
        x = self.drop(self.layer1(x))
        x = self.layer2(x)

        return x
    
class DecoderPrenet(nn.Module):
    def __init__(self,in_dim,out_dim,dropout = 0.1):
        super(DecoderPrenet,self).__init__()
        
        self.layer1 = LinearNorm(in_dim,int(out_dim//2))
        self.layer2 = LinearNorm(int(out_dim//2),out_dim)
        
        self.drop = nn.Dropout(dropout)
        
        
    def forward(self, x):
        x = self.drop(self.layer1(x))
        x = self.layer2(x)
        
        return x


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """
    def __init__(self, configs, dropout):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(configs['Audio_Configs']['num_mels'], configs['Postnet_Configs']['postnet_embedding_dim'],
                         kernel_size=configs['Postnet_Configs']['postnet_kernel_size'], stride=1,
                         padding=int((configs['Postnet_Configs']['postnet_kernel_size'] - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(configs['Postnet_Configs']['postnet_embedding_dim']))
        )

        for i in range(1, configs['Postnet_Configs']['postnet_n_convolutions'] - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(configs['Postnet_Configs']['postnet_embedding_dim'],
                             configs['Postnet_Configs']['postnet_embedding_dim'],
                             kernel_size=configs['Postnet_Configs']['postnet_kernel_size'], stride=1,
                             padding=int((configs['Postnet_Configs']['postnet_kernel_size'] - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(configs['Postnet_Configs']['postnet_embedding_dim']))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(configs['Postnet_Configs']['postnet_embedding_dim'], 
                         configs['Audio_Configs']['num_mels'],
                         kernel_size=configs['Postnet_Configs']['postnet_kernel_size'], stride=1,
                         padding=int((configs['Postnet_Configs']['postnet_kernel_size'] - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(configs['Audio_Configs']['num_mels']))
            )

    def forward(self, x):
        x = x.transpose(1,2)
        
        for i in range(len(self.convolutions) - 1):
            x = self.dropout(torch.tanh(self.convolutions[i](x)))
        x = self.dropout(self.convolutions[-1](x))

        return x.transpose(1,2)

class HeadPredictor(nn.Module):
    def __init__(self, configs, dropout=0.1):
        super(HeadPredictor, self).__init__()
        
        self.dec_prenet = DecoderPrenet(configs['EncDec_Configs']['embed_dim'],configs['Audio_Configs']['num_mels'])
        

        self.post_net = Postnet(configs,dropout)

        self.stop_linear = nn.Linear(configs['EncDec_Configs']['embed_dim'], 1)

    def forward(self, x):
        # all_mel_linear
        x_mel_linear = self.dec_prenet(x)

        #stop_linear
        stoplinear_output = self.stop_linear(x)

        x_postnet = self.post_net(x_mel_linear)

        mel_out = x_mel_linear + x_postnet


        return x_mel_linear, mel_out, stoplinear_output