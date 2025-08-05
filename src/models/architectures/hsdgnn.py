import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class HSDGNN_block(nn.Module):
    """
    The core building block of the HSDGNN model, responsible for capturing
    spatio-temporal dependencies.
    """
    def __init__(self, num_nodes, input_dim, rnn_units, embed_dim):
        super(HSDGNN_block, self).__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.rnn_units = rnn_units
        self.input_dim = input_dim

        self.gru1 = nn.GRU(embed_dim, rnn_units)
        self.gru2 = nn.GRU(rnn_units, rnn_units)

        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, rnn_units, rnn_units))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, rnn_units))

        self.diff = nn.Conv2d(rnn_units * 2, rnn_units, kernel_size=(1, 1), bias=True)
        self.dropout = nn.Dropout(p=0.1)

        self.x_embedding = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(1, 16)), ('sigmoid1', nn.Sigmoid()),
                ('fc2', nn.Linear(16, 2)), ('sigmoid2', nn.Sigmoid()),
                ('fc3', nn.Linear(2, embed_dim))
            ])
        )
        self.fc1 = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(input_dim, 16)), ('sigmoid1', nn.Sigmoid()),
                ('fc2', nn.Linear(16, 2)), ('sigmoid2', nn.Sigmoid()),
                ('fc3', nn.Linear(2, embed_dim))
            ])
        )
        self.fc2 = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(rnn_units, 16)), ('sigmoid1', nn.Sigmoid()),
                ('fc2', nn.Linear(16, 2)), ('sigmoid2', nn.Sigmoid()),
                ('fc3', nn.Linear(2, embed_dim))
            ])
        )

    def dynamic_dependency(self, output_of_gru, node_embeddings_all):
        filter_emb = self.fc2(output_of_gru)
        nodevec = torch.tanh(torch.mul(node_embeddings_all[0], filter_emb))
        supports1 = torch.stack([torch.eye(self.num_nodes)] * output_of_gru.shape[0]).to(node_embeddings_all[0].device)
        dynamic_adj_matrix = F.relu(torch.matmul(nodevec, nodevec.permute(0, 1, 3, 2)))
        x_g1 = torch.einsum("tnm,tbmc->tbnc", supports1, output_of_gru)
        x_g2 = torch.einsum("tbnm,tbmc->tbnc", dynamic_adj_matrix, output_of_gru)
        x_g = x_g1 + x_g2
        weights = torch.einsum('nd,dcr->ncr', node_embeddings_all[1], self.weights_pool)
        bias = torch.matmul(node_embeddings_all[1], self.bias_pool)
        x_g = torch.einsum('tbnc,ncr->tbnr', x_g, weights) + bias
        return x_g, dynamic_adj_matrix

    def forward(self, x, node_embeddings_all):
        B, T, N = x.shape[0], x.shape[1], x.shape[2]
        node_embed = self.x_embedding(x.unsqueeze(-1))
        supports_node2 = F.relu(torch.matmul(node_embed, node_embed.permute(0, 1, 2, 4, 3)))
        x2 = torch.einsum("btnji,btni->btnj", supports_node2, x)
        input_for_fc1 = x + x2
        input_for_gru1 = self.fc1(input_for_fc1).permute(1, 0, 2, 3)
        h0_1 = torch.zeros(1, B * N, self.rnn_units).to(input_for_gru1.device)
        output_of_gru1, _ = self.gru1(input_for_gru1.reshape(T, B * N, self.embed_dim), h0_1)
        output_of_gru1 = output_of_gru1.reshape(T, B, N, self.rnn_units)
        diff_signal, dynamic_adj = self.dynamic_dependency(output_of_gru1, node_embeddings_all)
        diffusion_input = torch.cat([output_of_gru1, diff_signal], dim=3).permute(1, 3, 2, 0)
        input_for_gru2 = self.diff(diffusion_input).permute(3, 0, 2, 1)
        input_for_gru2 = self.dropout(input_for_gru2)
        h0_2 = torch.zeros(1, B * N, self.rnn_units).to(input_for_gru2.device)
        output_of_gru2, _ = self.gru2(input_for_gru2.reshape(T, B * N, self.rnn_units), h0_2)
        final_output = output_of_gru2.reshape(T, B, N, self.rnn_units).permute(1, 0, 2, 3)
        return final_output, dynamic_adj

class HSDGNN(nn.Module):
    def __init__(self, args):
        super(HSDGNN, self).__init__()
        self.rnn_units = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.steps_per_day = args.steps_per_day
        self.node_embeddings = nn.Parameter(torch.randn(args.num_nodes, args.embed_dim))
        self.T_i_D_emb = nn.Parameter(torch.empty(args.steps_per_day, args.embed_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, args.embed_dim))
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)
        self.encoder1 = HSDGNN_block(args.num_nodes, args.input_dim, args.rnn_units, args.embed_dim)
        self.encoder2 = HSDGNN_block(args.num_nodes, args.input_dim, args.rnn_units, args.embed_dim)
        self.encoder3 = HSDGNN_block(args.num_nodes, args.input_dim, args.rnn_units, args.embed_dim)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.end_conv1 = nn.Conv2d(1, args.horizon * args.output_dim, kernel_size=(1, args.rnn_units), bias=True)
        self.end_conv2 = nn.Conv2d(1, args.horizon * args.output_dim, kernel_size=(1, args.rnn_units), bias=True)
        self.end_conv3 = nn.Conv2d(1, args.horizon * args.output_dim, kernel_size=(1, args.rnn_units), bias=True)
        self.end_conv1_b = nn.Conv2d(1, args.lag, kernel_size=(1, args.rnn_units), bias=True)
        self.end_conv2_b = nn.Conv2d(1, args.lag, kernel_size=(1, args.rnn_units), bias=True)

    def forward(self, source, return_adjs=False):
        # --- START OF FIX ---
        # The input `source` contains our time-series features.
        source_features = source
        B, T, N, _ = source.shape

        # The original model uses time-in-day and day-in-week features which we don't have.
        # This was causing a crash by misinterpreting stock prices as time indices.
        # We now adapt by creating a static node embedding structure that matches
        # the expected dimensions of the HSDGNN_block.

        # Replicate the static node embeddings across the batch and time dimensions
        # to create a "dynamic" part that is stable.
        node_embedding_dynamic_part = self.node_embeddings.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)

        # The HSDGNN_block expects a list containing two embedding tensors.
        # We provide the expanded one and the original static one to match the required structure.
        node_embeddings_all = [node_embedding_dynamic_part.permute(1, 0, 2, 3), self.node_embeddings]
        # --- END OF FIX ---

        output_1, adj1 = self.encoder1(source_features, node_embeddings_all)
        output_1_last = self.dropout1(output_1[:, -1:, :, :])
        prediction1 = self.end_conv1(output_1_last)
        
        source1_b = self.end_conv1_b(output_1_last)
        source2 = source_features - source1_b

        output_2, _ = self.encoder2(source2, node_embeddings_all)
        output_2_last = self.dropout2(output_2[:, -1:, :, :])
        prediction2 = self.end_conv2(output_2_last)

        source2_b = self.end_conv2_b(output_2_last)
        source3 = source2 - source2_b

        output_3, _ = self.encoder3(source3, node_embeddings_all)
        output_3_last = self.dropout3(output_3[:, -1:, :, :])
        prediction3 = self.end_conv3(output_3_last)

        final_prediction = prediction1 + prediction2 + prediction3
        
        if return_adjs:
            return final_prediction, adj1
        else:
            return final_prediction