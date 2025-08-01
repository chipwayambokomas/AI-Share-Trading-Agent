# src/models/graphwavenet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()
    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = NConv()
        c_in_new = (order * support_len + 1) * c_in
        self.mlp = nn.Conv2d(c_in_new, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class GraphWaveNet(nn.Module):
    def __init__(self, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2):
        super(GraphWaveNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        self.supports = supports
        receptive_field = 1
        if supports is None:
            self.supports = []
        support_len = len(self.supports)
        if gcn_bool and addaptadj:
            support_len += 1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels, kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(GCN(dilation_channels, residual_channels, dropout, support_len=support_len))
        if self.addaptadj and supports is not None:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)
        self.receptive_field = receptive_field

    def forward(self, input):
        input = input.permute(0, 3, 2, 1)
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0
        new_supports = self.supports
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]
        for i in range(self.blocks * self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            if self.gcn_bool and self.supports is not None:
                x = self.gconv[i](x, new_supports)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x