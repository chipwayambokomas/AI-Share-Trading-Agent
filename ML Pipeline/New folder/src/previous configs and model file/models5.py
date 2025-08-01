# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# TCN & MLP MODELS (Unchanged)
# =============================================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class TCN_Forecaster(nn.Module):
    def __init__(self, input_channels, output_size, num_channels, kernel_size, dropout):
        super(TCN_Forecaster, self).__init__()
        self.tcn = TemporalConvNet(input_channels, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
    def forward(self, x):
        y = self.tcn(x.permute(0, 2, 1))
        output = self.linear(y[:, :, -1])
        return output

class MLP_Forecaster(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout=0.2, **kwargs):
        super(MLP_Forecaster, self).__init__()
        layers = []
        last_size = input_size
        for layer_size in hidden_layers:
            layers.append(nn.Linear(last_size, layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_size = layer_size
        layers.append(nn.Linear(last_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        return self.network(x_flat)

# =============================================================================
# GRAPHWAVENET MODEL ARCHITECTURE
# =============================================================================

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

# =============================================================================
# DSTAGNN MODEL ARCHITECTURE
# =============================================================================
from scipy.sparse.linalg import eigs

def scaled_Laplacian(W):
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    try:
        lambda_max = eigs(L, k=1, which='LR')[0].real
    except:
        lambda_max = np.linalg.eigvalsh(L).max()
    return (2 * L) / lambda_max - np.identity(W.shape[0])

def cheb_polynomial(L_tilde, K):
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials

class SScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(SScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        return scores

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, num_of_d):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.num_of_d = num_of_d

    def forward(self, Q, K, V, attn_mask, res_att):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) + res_att
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        attn = F.softmax(scores, dim=3)
        context = torch.matmul(attn, V)
        return context, scores

class SMultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, device):
        super(SMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.device = device
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)

    def forward(self, input_Q, input_K, attn_mask):
        batch_size = input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        attn = SScaledDotProductAttention(self.d_k)(Q, K, attn_mask)
        return attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, num_of_d, device):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.num_of_d = num_of_d
        self.device = device
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask, res_att):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2, 3)
        K = self.W_K(input_K).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2, 3)
        V = self.W_V(input_V).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_v).transpose(2, 3)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, res_attn = ScaledDotProductAttention(self.d_k, self.num_of_d)(Q, K, V, attn_mask, res_att)
        context = context.transpose(2, 3).reshape(batch_size, self.num_of_d, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return self.layer_norm(output + residual), res_attn

class cheb_conv_withSAt(nn.Module):
    def __init__(self, K, cheb_polynomials, in_channels, out_channels, num_of_vertices):
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.relu = nn.ReLU(inplace=True)
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])
        self.mask = nn.ParameterList([nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, spatial_attention, adj_pa):
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k]
                mask = self.mask[k]
                myspatial_attention = spatial_attention[:, k, :, :] + adj_pa.mul(mask)
                myspatial_attention = F.softmax(myspatial_attention, dim=1)
                T_k_with_at = T_k.mul(myspatial_attention)
                theta_k = self.Theta[k]
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)
                output = output + rhs.matmul(theta_k)
            outputs.append(output.unsqueeze(-1))
        return self.relu(torch.cat(outputs, dim=-1))

class Embedding(nn.Module):
    def __init__(self, nb_seq, d_Em, num_of_features, Etype, device):
        super(Embedding, self).__init__()
        self.nb_seq = nb_seq
        self.Etype = Etype
        self.num_of_features = num_of_features
        self.pos_embed = nn.Embedding(nb_seq, d_Em)
        self.norm = nn.LayerNorm(d_Em)
        self.device = device

    def forward(self, x, batch_size):
        if self.Etype == 'T':
            pos = torch.arange(self.nb_seq, dtype=torch.long).to(self.device)
            pos = pos.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_of_features, self.nb_seq)
            embedding = x.permute(0, 2, 3, 1) + self.pos_embed(pos)
        else:
            pos = torch.arange(self.nb_seq, dtype=torch.long).to(self.device)
            pos = pos.unsqueeze(0).expand(batch_size, self.nb_seq)
            embedding = x + self.pos_embed(pos)
        Emx = self.norm(embedding)
        return Emx

class GTU(nn.Module):
    def __init__(self, in_channels, time_strides, kernel_size):
        super(GTU, self).__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=(1, kernel_size), stride=(1, time_strides))

    def forward(self, x):
        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, : self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x_gtu = torch.mul(self.tanh(x_p), self.sigmoid(x_q))
        return x_gtu

class DSTAGNN_block(nn.Module):
    def __init__(self, in_channels, K, nb_chev_filter, time_strides, cheb_polynomials, adj_pa, num_of_vertices, num_of_timesteps, d_model, d_k, d_v, n_heads, device):
        super(DSTAGNN_block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.adj_pa = torch.FloatTensor(adj_pa).to(device)
        self.pre_conv = nn.Conv2d(num_of_timesteps, d_model, kernel_size=(1, in_channels))
        self.EmbedT = Embedding(num_of_timesteps, num_of_vertices, in_channels, 'T', device)
        self.EmbedS = Embedding(num_of_vertices, d_model, in_channels, 'S', device)
        self.TAt = MultiHeadAttention(d_model=num_of_vertices, d_k=d_k, d_v=d_v, n_heads=n_heads, num_of_d=in_channels, device=device)
        self.SAt = SMultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=K, device=device)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter, num_of_vertices)
        self.gtu3 = GTU(nb_chev_filter, time_strides, 3)
        self.gtu5 = GTU(nb_chev_filter, time_strides, 5)
        self.gtu7 = GTU(nb_chev_filter, time_strides, 7)
        self.residual_conv = nn.Conv2d(in_channels, nb_chev_filter, kernel_size=(1, 1), stride=(1, time_strides))
        
        def get_out_len(in_len, kernel, stride):
            return (in_len - (kernel - 1) - 1) // stride + 1
        
        gtu3_out_len = get_out_len(num_of_timesteps, 3, time_strides)
        gtu5_out_len = get_out_len(num_of_timesteps, 5, time_strides)
        gtu7_out_len = get_out_len(num_of_timesteps, 7, time_strides)
        fc_input_size = gtu3_out_len + gtu5_out_len + gtu7_out_len
        
        self.fcmy = nn.Sequential(nn.Linear(fc_input_size, num_of_timesteps//time_strides), nn.Dropout(0.05))
        self.ln = nn.LayerNorm(nb_chev_filter)

    def forward(self, x, res_att):
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        
        if num_of_features == 1:
            TEmx = self.EmbedT(x, batch_size)
        else:
            TEmx = x.permute(0, 2, 3, 1)
        TATout, re_At = self.TAt(TEmx, TEmx, TEmx, None, res_att)
        
        x_TAt = self.pre_conv(TATout.permute(0, 2, 3, 1))[:, :, :, -1].permute(0, 2, 1)
        SEmx_TAt = self.EmbedS(x_TAt, batch_size)
        STAt = self.SAt(SEmx_TAt, SEmx_TAt, None)
        
        spatial_gcn = self.cheb_conv_SAt(x, STAt, self.adj_pa)
        
        X = spatial_gcn.permute(0, 2, 1, 3)
        x_gtu = [self.gtu3(X), self.gtu5(X), self.gtu7(X)]
        time_conv = torch.cat(x_gtu, dim=-1)
        time_conv = self.fcmy(time_conv)
        
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        
        output = self.ln((F.relu(x_residual + time_conv)).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        
        return output, re_At

class DSTAGNN(nn.Module):
    def __init__(self, device, nb_block, in_channels, K, nb_chev_filter, time_strides, adj_mx, num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads):
        super(DSTAGNN, self).__init__()
        self.n_heads = n_heads
        cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in cheb_polynomial(scaled_Laplacian(adj_mx), K)]
        self.BlockList = nn.ModuleList([DSTAGNN_block(in_channels, K, nb_chev_filter, time_strides, cheb_polynomials, adj_mx, num_of_vertices, len_input, d_model, d_k, d_v, n_heads, device)])
        self.BlockList.extend([DSTAGNN_block(nb_chev_filter, K, nb_chev_filter, 1, cheb_polynomials, adj_mx, num_of_vertices, len_input//time_strides, d_model, d_k, d_v, n_heads, device) for _ in range(nb_block-1)])
        self.final_conv = nn.Conv2d(int(nb_chev_filter * nb_block), 128, kernel_size=(1, len_input//time_strides))
        self.final_fc = nn.Linear(128, num_for_predict)
        self.device = device

    def forward(self, x):
        x = x.permute(0, 2, 3, 1) 
        need_concat = []
        res_att = torch.zeros(x.size(0), x.size(2), self.n_heads, x.size(3), x.size(3)).to(self.device)
        
        for block in self.BlockList:
            x, res_att = block(x, res_att)
            need_concat.append(x)
        
        final_x = torch.cat(need_concat, dim=2)
        
        output1 = self.final_conv(final_x.permute(0, 2, 1, 3)) 
        output1 = output1.squeeze(-1).permute(0, 2, 1)
        output = self.final_fc(output1)
        return output

def make_dstagnn_model(**kwargs):
    """
    Helper function to create a DSTAGNN model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes = kwargs['num_nodes']
    in_dim = kwargs['in_dim']
    out_dim = kwargs['out_dim']
    input_window = kwargs['input_window']
    adj_matrix = kwargs['adj_matrix']
    dstagnn_args = kwargs['dstagnn_args']

    model = DSTAGNN(
        device=device,
        nb_block=dstagnn_args['nb_block'],
        in_channels=in_dim,
        K=dstagnn_args['K'],
        nb_chev_filter=dstagnn_args['nb_chev_filter'],
        time_strides=dstagnn_args['time_strides'],
        adj_mx=adj_matrix,
        num_for_predict=out_dim,
        len_input=input_window,
        num_of_vertices=num_nodes,
        d_model=dstagnn_args['d_model'],
        d_k=dstagnn_args['d_k'],
        d_v=dstagnn_args['d_v'],
        n_heads=dstagnn_args['n_heads']
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model.to(device)
