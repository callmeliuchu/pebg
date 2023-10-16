import os
import torch
import numpy as np
import math
from scipy import sparse
from torch.nn import functional as F


# 定义PNN层
class PNN1(torch.nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, dropout):
        super(PNN1, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim * embed_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear3 = torch.nn.Linear(hidden_dim // 2, 1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        x = self.linear3(x)
        return x


import torch
import torch.nn as nn


def pnn1(inputs, embed_size, hidden_dim, keep_prob):
    inputs = [inp.cuda() for inp in inputs]  # 把输入移动到GPU

    num_inputs = len(inputs)
    num_pairs = int(num_inputs * (num_inputs - 1) / 2)

    xw = torch.cat(inputs, 1)
    xw3d = xw.view(-1, num_inputs, embed_size)

    row = []
    col = []
    for i in range(num_inputs - 1):
        for j in range(i + 1, num_inputs):
            row.append(i)
            col.append(j)

    p = xw3d[:, row, :].transpose(1, 0)
    q = xw3d[:, col, :].transpose(1, 0)

    p = p.view(-1, num_pairs, embed_size)
    q = q.view(-1, num_pairs, embed_size)

    ip = (p * q).sum(-1).view(-1, num_pairs)
    l = torch.cat([xw, ip], 1)

    h = nn.Linear(l.size(1), hidden_dim)(l)
    h = nn.ReLU()(h)
    h = nn.Dropout(p=1 - keep_prob)(h)

    p = nn.Linear(hidden_dim, 1)(h).view(-1)

    return h, p


# 加载数据
data_folder = 'assist09'
con_sym = ';'

pro_skill_coo = sparse.load_npz(os.path.join(data_folder, 'pro_skill_sparse.npz'))
skill_skill_coo = sparse.load_npz(os.path.join(data_folder, 'skill_skill_sparse.npz'))
pro_pro_coo = sparse.load_npz(os.path.join(data_folder, 'pro_pro_sparse.npz'))

pro_num, skill_num = pro_skill_coo.shape
print(f'问题数目{pro_num}, 技能数目{skill_num}')

pro_skill = pro_skill_coo.toarray()
pro_pro = pro_pro_coo.toarray()
skill_skill = skill_skill_coo.toarray()
pro_skill_tensor = torch.from_numpy(pro_skill)
skill_skill_tensor = torch.from_numpy(skill_skill)
pro_pro_tensor = torch.from_numpy(pro_pro)

pro_feat = np.load(os.path.join(data_folder, 'pro_feat.npz'))['pro_feat']
print('问题特征形状:', pro_feat.shape)

diff_feat_dim = pro_feat.shape[1] - 1
embed_dim = 64
hidden_dim = 128
dropout = 0.5
lr = 0.001
batch_size = 256
epochs = 200

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 定义模型
class PEGB(torch.nn.Module):
    def __init__(self, pro_num, skill_num, diff_feat_dim, embed_dim):
        super(PEGB, self).__init__()

        self.pro_embeddings = torch.nn.Embedding(pro_num, embed_dim)
        self.skill_embeddings = torch.nn.Embedding(skill_num, embed_dim)
        self.diff_embeddings = torch.nn.Linear(diff_feat_dim, embed_dim)

        # self.pnn1 = pnn1(3, embed_dim, hidden_dim, dropout)

    def forward(self, pro, diff_feat, pro_skill, pro_pro, skill_skill):
        pro_embed = self.pro_embeddings(pro)
        skill_embed = self.skill_embeddings(skill_skill)
        diff_feat_embed = self.diff_embeddings(diff_feat)

        # pro-skill
        pro_skill_logits = (pro_embed @ skill_embed.t()).view(-1)
        pro_skill_loss = F.binary_cross_entropy_with_logits(pro_skill_logits, pro_skill.view(-1))

        # pro-pro
        pro_pro_logits = (pro_embed @ pro_embed.t()).view(-1)
        # print(pro_pro_logits.shape,pro_pro.shape)
        pro_pro_loss = F.binary_cross_entropy_with_logits(pro_pro_logits, pro_pro.contiguous().view(-1))

        # skill-skill
        skill_skill_logits = (skill_embed @ skill_embed.t()).view(-1)
        skill_skill_loss = F.binary_cross_entropy_with_logits(skill_skill_logits, skill_skill_tensor.view(-1))

        # 特征融合
        skill_embed = pro_skill @ skill_embed / pro_skill.sum(1, keepdim=True)
        h,p= pnn1([pro_embed, skill_embed, diff_feat_embed],embed_dim, hidden_dim, 0.5)
        # pro_final_embed = None
        mse = ((p-pro_feat[:,-1])**2).mean()

        return pro_skill_loss, pro_pro_loss, skill_skill_loss, mse


model = PEGB(pro_num, skill_num, diff_feat_dim, embed_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for i in range(0, pro_num, batch_size):
        batch_pro = torch.arange(i, min(i + batch_size, pro_num)).long().to(device)
        batch_pro_skill = torch.Tensor(pro_skill[i:i + batch_size]).to(device)
        batch_pro_pro = torch.Tensor(pro_pro[i:i + batch_size,i:i + batch_size]).to(device)
        batch_diff_feat = torch.Tensor(pro_feat[i:i + batch_size, :-1]).to(device)
        batch_skill_skill = torch.arange(skill_num).to(device)

        pro_skill_loss, pro_pro_loss, skill_skill_loss, mse = model(batch_pro,
                                                                  batch_diff_feat,
                                                                  batch_pro_skill,
                                                                  batch_pro_pro,
                                                                  batch_skill_skill)
        loss = pro_skill_loss + pro_pro_loss + skill_skill_loss + mse

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= math.ceil(pro_num / batch_size)
    print(f'Epoch {epoch}, Loss {train_loss:.4f}')

# 保存训练好的embedding
model.eval()
with torch.no_grad():
    pro_embeddings = model.pro_embeddings.weight.cpu().numpy()
    skill_embeddings = model.skill_embeddings.weight.cpu().numpy()

    batch_pro = torch.arange(pro_num).long()
    batch_pro_skill = torch.Tensor(pro_skill)
    batch_diff_feat = torch.Tensor(pro_feat[:, :-1])
    batch_skill_skill = torch.arange(skill_num).to(device)
    pro_final_embeddings = model(batch_pro, batch_diff_feat, batch_pro_skill, pro_pro_tensor, batch_skill_skill)[3].cpu().numpy()

# 处理技能embedding
with open(os.path.join(data_folder, 'skill_id_dict.txt'), 'r') as f:
    skill_id_dict = eval(f.read())

joint_skill_num = len(skill_id_dict)
skill_embeddings_new = np.zeros((joint_skill_num, skill_embeddings.shape[1]))
skill_embeddings_new[:skill_num] = skill_embeddings

for s in skill_id_dict:
    if con_sym in s:
        tmp_skill_id = skill_id_dict[s]
        tmp_skills = [skill_id_dict[t] for t in s.split(con_sym)]
        skill_embeddings_new[tmp_skill_id] = np.mean(skill_embeddings[tmp_skills], axis=0)

np.savez(os.path.join(data_folder, 'embedding_%d.npz' % epochs),
         pro_repre=pro_embeddings,
         skill_repre=skill_embeddings_new,
         pro_final_repre=pro_final_embeddings)