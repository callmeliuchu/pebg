#!/usr/bin/env python
# coding: utf-8

# In[1]:


def pnn1(inputs, embed_size, hidden_dim, keep_prob):
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
    p = p.contiguous().view(-1, num_pairs, embed_size)
    q = q.contiguous().view(-1, num_pairs, embed_size)
    ip = (p * q).sum(-1).view(-1, num_pairs)
    l = torch.cat([xw, ip], 1)
    h = nn.Linear(l.size(1), hidden_dim)(l)
    h = nn.ReLU()(h)
    h = nn.Dropout(p=1 - keep_prob)(h)
    p = nn.Linear(hidden_dim, 1)(h).view(-1)
    return h, p


# In[ ]:





# In[2]:


import os
import torch
import numpy as np
import math
from scipy import sparse
from torch.nn import functional as F


import torch
import torch.nn as nn




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
pro_feat_tensor = torch.from_numpy(pro_feat)
print('问题特征形状:', pro_feat.shape)

diff_feat_dim = pro_feat.shape[1] - 1
embed_dim = 64
hidden_dim = 64
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
        
        self.final_embedding = torch.zeros(pro_num, embed_dim)
    
    def final_pro_embedding(self):
        batch_pro = torch.arange(pro_num).long().to(device)
        batch_pro_skill = torch.Tensor(pro_skill).to(device)
        batch_pro_pro = torch.Tensor(pro_pro).to(device)
        batch_diff_feat = torch.Tensor(pro_feat[:,:-1]).to(device)
        batch_skill_skill = torch.arange(skill_num).to(device)
        pro_embed = self.pro_embeddings(batch_pro)
        skill_embed = self.skill_embeddings(batch_skill_skill)
        diff_feat_embed = self.diff_embeddings(batch_diff_feat)
        print(pro_embed.shape, skill_embed.shape, diff_feat_embed.shape)
        skill_embed = batch_pro_skill @ skill_embed / batch_pro_skill.sum(1, keepdim=True)
        h,p= pnn1([pro_embed, skill_embed, diff_feat_embed],embed_dim, hidden_dim, 0.5)
        return h



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
#         print(pro_embed.shape,skill_embed.shape,diff_feat_embed.shape)
        h,p= pnn1([pro_embed, skill_embed, diff_feat_embed],embed_dim, hidden_dim, 1.0)
#         h,p = self.pnn([pro_embed, skill_embed, diff_feat_embed])
        # pro_final_embed = None
#         print(p.shape,pro_feat_tensor.shape)
        self.final_embedding[pro] = h
        mse = ((p-diff_feat[:,-1])**2).mean()

        return pro_skill_loss, pro_pro_loss, skill_skill_loss,mse,h


model = PEGB(pro_num, skill_num, diff_feat_dim, embed_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)




# In[3]:


def final_pro_embedding(self):
    batch_pro = torch.arange(pro_num).long().to(device)
    batch_pro_skill = torch.Tensor(pro_skill).to(device)
    batch_pro_pro = torch.Tensor(pro_pro).to(device)
    batch_diff_feat = torch.Tensor(pro_feat[:,:-1]).to(device)
    batch_skill_skill = torch.arange(skill_num).to(device)
    pro_embed = self.pro_embeddings(batch_pro)
    skill_embed = self.skill_embeddings(batch_skill_skill)
    diff_feat_embed = self.diff_embeddings(batch_diff_feat)
    print(pro_embed.shape, skill_embed.shape, diff_feat_embed.shape)
    skill_embed = batch_pro_skill @ skill_embed / batch_pro_skill.sum(1, keepdim=True)
    h,p= pnn1([pro_embed, skill_embed, diff_feat_embed],embed_dim, hidden_dim, 1.0)
    return h


# In[4]:


final_pro_embedding(model)[0]


# In[ ]:


for epoch in range(1000):
    model.train()
    train_loss = 0

    for i in range(0, pro_num, batch_size):
        batch_pro = torch.arange(i, min(i + batch_size, pro_num)).long().to(device)
        batch_pro_skill = torch.Tensor(pro_skill[i:i + batch_size]).to(device)
        batch_pro_pro = torch.Tensor(pro_pro[i:i + batch_size,i:i + batch_size]).to(device)
        batch_diff_feat = torch.Tensor(pro_feat[i:i + batch_size, :-1]).to(device)
        batch_skill_skill = torch.arange(skill_num).to(device)

        pro_skill_loss, pro_pro_loss, skill_skill_loss, mse,h = model(batch_pro,
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


# In[ ]:


model.pro_embeddings.weight.shape, model.skill_embeddings.weight.shape, model.diff_embeddings.weight.shape


# In[ ]:


x=model.final_pro_embedding()


# In[ ]:


model.pro_embeddings.weight


# In[ ]:


# 保存训练好的embedding
model.eval()
with torch.no_grad():
    pro_embeddings = model.pro_embeddings.weight.cpu().numpy()
    skill_embeddings = model.skill_embeddings.weight.cpu().numpy()

    batch_pro = torch.arange(pro_num).long()
    batch_pro_skill = torch.Tensor(pro_skill)
    batch_diff_feat = torch.Tensor(pro_feat[:, :-1])
    batch_skill_skill = torch.arange(skill_num).to(device)

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


# In[ ]:


pro_embeddings.shape


# In[ ]:


skill_embeddings.shape


# In[ ]:


pro_skill_tensor


# In[ ]:


torch.sigmoid(model.pro_embeddings.weight @ model.skill_embeddings.weight.t())


# In[ ]:


pro_skill_tensor


# In[ ]:


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 准备数据
# data_tensor = model.pro_embeddings.weight
data_tensor = model.pro_embeddings.weight

# 初始化t-SNE模型
tsne = TSNE(n_components=2, perplexity=30, learning_rate=100, n_iter=1000)

# 执行t-SNE降维
embedded_data = tsne.fit_transform(data_tensor.detach().numpy())

# 可视化结果
# plt.scatter(embedded_data[:, 0], embedded_data[:, 1])
for label in range(skill_num):
    pids = pro_skill_tensor[:,label].nonzero().view(-1).detach().numpy()
#     print(pids)
    red =  np.random.randint(0, 256)
    green = np.random.randint(0, 256)
    blue = np.random.randint(0, 256)
    color = (red / 255, green / 255, blue / 255)
    class_data = embedded_data[pids]
    plt.scatter(class_data[:, 0], class_data[:, 1],c=color, label=str(label))
plt.show()


# In[105]:


pro_skill_tensor


# In[106]:


torch.sigmoid(model.pro_embeddings.weight @ model.skill_embeddings.weight.t())


# In[107]:


pro_skill_tensor


# In[112]:


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 准备数据
# data_tensor = model.pro_embeddings.weight
data_tensor = model.pro_embeddings.weight

# 初始化t-SNE模型
tsne = TSNE(n_components=2, perplexity=30, learning_rate=100, n_iter=1000)

# 执行t-SNE降维
embedded_data = tsne.fit_transform(data_tensor.detach().numpy())

# 可视化结果
# plt.scatter(embedded_data[:, 0], embedded_data[:, 1])
for label in range(skill_num):
    pids = pro_skill_tensor[:,label].nonzero().view(-1).detach().numpy()
#     print(pids)
    red =  np.random.randint(0, 256)
    green = np.random.randint(0, 256)
    blue = np.random.randint(0, 256)
    color = (red / 255, green / 255, blue / 255)
    class_data = embedded_data[pids]
    plt.scatter(class_data[:, 0], class_data[:, 1],c=color, label=str(label))
plt.show()


# In[ ]:





# In[ ]:




