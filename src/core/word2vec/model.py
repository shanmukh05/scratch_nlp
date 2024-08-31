import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOW(nn.Module):
    def __init__(self, config_dict):
        super(CBOW, self).__init__()

        self.embed_dim = config_dict["model"]["embed_dim"]
        self.num_vocab = 1 + config_dict["dataset"]["num_vocab"]

        self.cxt_embedding = nn.Embedding(self.num_vocab, self.embed_dim)
        self.lbl_embedding = nn.Embedding(self.num_vocab-1, self.embed_dim)

    def forward(self, l_cxt, r_cxt, l_lbl, r_lbl):
        l_cxt_emb = self.compute_cxt_embed(l_cxt)
        r_cxt_emb = self.compute_cxt_embed(r_cxt)
        l_lbl_emb = self.lbl_embedding(torch.LongTensor(l_lbl)-self.num_vocab)
        r_lbl_emb = self.lbl_embedding(torch.LongTensor(r_lbl)-self.num_vocab)
        
        l_loss = torch.mul(l_cxt_emb, l_lbl_emb).squeeze()
        l_loss = torch.sum(l_loss, dim=1)
        l_loss = F.logsigmoid(-1 * l_loss)
        r_loss = torch.mul(r_cxt_emb, r_lbl_emb).squeeze()
        r_loss = torch.sum(r_loss, dim=1)
        r_loss = F.logsigmoid(r_loss)

        loss = torch.sum(l_loss) + torch.sum(r_loss)
        return -1 * loss

    def compute_cxt_embed(self, cxt):
        lbl_emb = self.cxt_embedding(torch.LongTensor(cxt))
        return torch.mean(lbl_emb, dim=1)