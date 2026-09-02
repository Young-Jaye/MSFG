import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class Decoder(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(nhid2, nhid1),
            nn.BatchNorm1d(nhid1),
            nn.ReLU(),
        )
        self.pi = nn.Linear(nhid1, nfeat)
        self.disp = nn.Linear(nhid1, nfeat)
        self.mean = nn.Linear(nhid1, nfeat)
        self.disp_act = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.mean_act = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.disp_act(self.disp(x))
        mean = self.mean_act(self.mean(x))
        return pi, disp, mean


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        weights = self.project(z)
        beta = torch.softmax(weights, dim=1)
        return (beta * z).sum(1), beta


class FuseMGCN_Net(nn.Module):
    """
    方案B：
    - 视图：Fused 图 / Spatial 图 / Feature 图 / Shared(common) 图
    - Shared 用 fused 与 spatial 的 common encoder + learnable gate 混合
    """

    def __init__(self, nfeat_gene, nhid1, nhid2, dropout, att_hidden=16):
        super().__init__()
        self.SGCN = GCN(nfeat_gene, nhid1, nhid2, dropout)
        self.FGCN = GCN(nfeat_gene, nhid1, nhid2, dropout)
        self.UGCN = GCN(nfeat_gene, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat_gene, nhid1, nhid2, dropout)

        self.att = Attention(nhid2, hidden_size=att_hidden)
        self.mlp = nn.Sequential(nn.Linear(nhid2, nhid2))
        self.zinb_decoder = Decoder(nfeat_gene, nhid1, nhid2)
        self.shared_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_gene, fadj, sadj, fuse_adj):
        emb_s = self.SGCN(x_gene, sadj)
        emb_f = self.FGCN(x_gene, fadj)
        emb_u = self.UGCN(x_gene, fuse_adj)

        com_u = self.CGCN(x_gene, fuse_adj)
        com_s = self.CGCN(x_gene, sadj)

        w = torch.sigmoid(self.shared_gate)
        shared = w * com_u + (1.0 - w) * com_s

        emb_stack = torch.stack([emb_u, emb_s, emb_f, shared], dim=1)
        emb, att_weights = self.att(emb_stack)
        emb = self.mlp(emb)
        pi, disp, mean = self.zinb_decoder(emb)
        return com_u, com_s, emb, pi, disp, mean, att_weights
