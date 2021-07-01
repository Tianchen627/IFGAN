from torch import nn
import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np
import math

def build_model(args, user_total, item_total, entity_total, relation_total, i_map=None, e_map=None, new_map=None, share_total=0):
    model_cls = DistMult
    return model_cls(entity_total,
                     relation_total,
                     args)


class DistMult(torch.nn.Module):
    def __init__(self, entity_total, relation_total, args):
        super(DistMult, self).__init__()
        self.entity_total = entity_total
        self.relation_total = relation_total

        self.embedding_size = int(args.embedding_size/4)
        # self.add_noise = args.add_noise
        self.sigma = args.sigma
        self.beta = 1.0

        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.emb_def()
        self.device = torch.device('cuda' if args.use_cuda else 'cpu')
        self = self.to(self.device)


    def emb_def(self):
        self.ent_embeddings = torch.nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_embeddings = torch.nn.Embedding(2 *self.relation_total, self.embedding_size)
        xavier_normal_(self.ent_embeddings.weight.data)
        xavier_normal_(self.rel_embeddings.weight.data)
        # self.gen_fuc = nn.Sequential(
        #     nn.Linear(self.embedding_size * 2, self.embedding_size),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        self.conv1 = torch.nn.Conv2d(1, 8, (3, 3), 1, 0)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm1d(self.embedding_size)
        self.fc = torch.nn.Linear(192,self.embedding_size)
        self.hidden_drop = torch.nn.Dropout(0.3)
        self.feature_map_drop = torch.nn.Dropout2d(0.2)

    def encode_kg(self, ent_emb, rel_emb):
        return ent_emb * rel_emb

    def form_query(self, e1, rel):
        e1_embedded_all = self.get_emb_all()
        ent_emb = e1_embedded_all[e1]
        rel_emb = self.rel_embeddings(rel)
        # ent_emb = self.inp_drop(ent_emb)
        # rel_emb = self.inp_drop(rel_emb)

        e1_embedded= ent_emb.view(-1, 1, 5, 5)
        rel_embedded = rel_emb.view(-1, 1, 5, 5)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        # hr_encoded = self.encode_kg(ent_emb, rel_emb)
        # hr_encoded = self.query_add_noise(hr_encoded)
        # hr_encoded = self.gen_fuc(hr_encoded)
        return x

    def query_judge(self, query_now, tails):
        tail_emb = self.ent_embeddings(tails)
        if len(tail_emb.size()) == 3:
            query_now = query_now.unsqueeze(1)  # (B, 1, emb)
        score = torch.sum(query_now * tail_emb, dim=-1)
        return score

    def query_add_noise(self, query):
        np_noise = np.random.normal(0.0, self.sigma, (query.size(0), self.embedding_size))
        noise = Variable(torch.FloatTensor(np_noise).to(self.device))
        return torch.cat([query, noise], dim=-1)

    def forward(self, e1, rel):
        e1_embedded_all = self.get_emb_all()
        ent_emb = e1_embedded_all[e1]
        rel_emb = self.rel_embeddings(rel)
        # ent_emb = self.inp_drop(ent_emb)
        # rel_emb = self.inp_drop(rel_emb)

        e1_embedded= ent_emb.view(-1, 1, 5, 5)
        rel_embedded = rel_emb.view(-1, 1, 5, 5)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        # hr_encoded = self.encode_kg(ent_emb, rel_emb)
        # hr_encoded = self.query_add_noise(hr_encoded)
        # hr_encoded = self.gen_fuc(hr_encoded)

        #x = torch.mm(hr_encoded, e1_embedded_all.transpose(1, 0))
        #pred = torch.sigmoid(x)
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        return x

    def get_candidates(self, pretrain=False):
        return self.ent_embeddings.weight

    def evaluate(self, e1, rel, e1_embedded_all, pretrain=False):
        #e1_embedded_all = self.get_emb_all()
        ent_emb = e1_embedded_all[e1]
        rel_emb = self.rel_embeddings(rel)

        e1_embedded= ent_emb.view(-1, 1, 5, 5)
        rel_embedded = rel_emb.view(-1, 1, 5, 5)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        #hr_encoded = self.encode_kg(ent_emb, rel_emb)
        #Evaluate without noise
        #hr_encoded = self.gen_fuc(hr_encoded)


        #x = torch.mm(hr_encoded, e1_embedded_all.transpose(1, 0))
        # pred = torch.sigmoid(x)
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        return x

    def forward_triple(self, heads, rels, tails):
        '''

        :param heads: (B, )
        :param rels:  (B, )
        :param tails: (B, N)
        :return:
        '''
        batch_size = heads.size(0)
        num_sample = tails.size(1)
        e1_embedded_all = self.get_emb_all()
        head_emb = e1_embedded_all[heads]
        rel_emb = self.rel_embeddings(rels)
        tail_emb = e1_embedded_all[tails].view(batch_size, num_sample, self.embedding_size)
        # hr_encoded = self.encode_kg(head_emb, rel_emb)
        # hr_encoded = self.query_add_noise(hr_encoded)
        # hr_encoded = self.gen_fuc(hr_encoded)
        # hr_encoded = hr_encoded.view(batch_size, self.embedding_size, 1)
        #scores = torch.bmm(tail_emb, hr_encoded).squeeze()#(B, N, emb) (B, emb, 1) - > (B, N, 1)

        e1_embedded= head_emb.view(-1, 1, 5, 5)
        rel_embedded = rel_emb.view(-1, 1, 5, 5)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        hr_encoded = x.view(batch_size, self.embedding_size, 1)
        scores = torch.bmm(tail_emb, hr_encoded).squeeze()#(B, N, emb) (B, emb, 1) - > (B, N, 1)
        return scores

    def get_emb_all(self):
        return self.ent_embeddings.weight

    def forward_kg(self, e1, rel, all_e_ids=None):
        return self.forward(e1, rel)