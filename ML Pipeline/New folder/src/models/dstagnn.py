# src/models/dstagnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse.linalg import eigs

# (Helper functions scaled_Laplacian, cheb_polynomial remain the same)
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


# (All Attention classes and other helpers remain the same)
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

    # --- FIX 1 of 2: Change the forward method signature ---
    def forward(self, x):
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # Initialize the residual attention tensor with the correct shape for this block
        res_att = torch.zeros(batch_size, num_of_features, self.TAt.n_heads, num_of_timesteps, num_of_timesteps).to(self.adj_pa.device)

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

        return output

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

        # --- FIX 2 of 2: Remove the residual attention logic from the main forward loop ---
        for block in self.BlockList:
            x = block(x)
            need_concat.append(x)

        final_x = torch.cat(need_concat, dim=2)

        output1 = self.final_conv(final_x.permute(0, 2, 1, 3))
        output1 = output1.squeeze(-1).permute(0, 2, 1)
        output = self.final_fc(output1)
        return output

# (make_dstagnn_model function remains the same)
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