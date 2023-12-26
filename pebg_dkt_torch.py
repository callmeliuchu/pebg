import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from scipy import sparse
from tqdm import tqdm
import os
import time, math
train_flag = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_test_split(data, split=0.6):
    n_samples = data[0].shape[0]
    split_point = int(n_samples * split)
    train_data = [d[:split_point] for d in data]
    test_data = [d[split_point:] for d in data]
    return train_data, test_data


# 数据处理
data_folder = "assist09"
data = np.load(os.path.join(data_folder, 'ass9.npz'))
y, skill, problem, real_len = data['y'], data['skill'], data['problem'], data['real_len']
skill_num, pro_num = data['skill_num'], data['problem_num']
print('problem number %d, skill number %d' % (pro_num, skill_num))

# 划分训练集和测试集
train_data, test_data = train_test_split([y, skill, problem, real_len])
train_y, train_skill, train_problem, train_real_len = train_data
test_y, test_skill, test_problem, test_real_len = test_data

# 嵌入初始化
# embed_data = np.load(os.path.join(data_folder, 'embedding_200.npz'))
# _, _, pre_pro_embed = embed_data['pro_repre'], embed_data['skill_repre'], embed_data['pro_final_repre']
pre_pro_embed = data['problem_embedding']
print('xxxxx',pre_pro_embed.shape, pre_pro_embed.dtype)


print('yyyyyyyyy',pre_pro_embed.shape, pre_pro_embed.dtype)

# 超参数
epochs = 100
bs = 64
embed_dim = pre_pro_embed.shape[1]
hidden_dim = 64
lr = 0.001
use_pretrain = True
train_embed = False


import torch

def concat_zero(x):
    xx = x[:-1]
    yy = x[-1]
    zero_tensor = torch.zeros(embed_dim, dtype=torch.float32)
    o = torch.cat([xx, zero_tensor]) if yy > 0. else torch.cat([zero_tensor, xx])
    return o

# 模型定义
class LSTM(nn.Module):
    def __init__(self, pro_num, embed_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.pro_embeddings = nn.Embedding(pro_num, embed_dim, padding_idx=0)
        # lstm_input_dim = embed_dim + 1
        self.lstm = nn.LSTM(embed_dim*2, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, embed_dim)
        self.sigmoid = nn.Sigmoid()

        if use_pretrain:
            pretrained_weight = torch.from_numpy(pre_pro_embed)
            self.pro_embeddings.weight.data.copy_(pretrained_weight)
            self.pro_embeddings.weight.requires_grad = train_embed

    def forward(self, pro_seq, y_seq, pro_len):
        batch_size, seq_len = pro_seq.shape
        # 创建全零的张量
        zeros_tensor = torch.zeros(batch_size, seq_len, embed_dim)
        # 在指定维度进行拼接，生成需要填充的张量
        ones_tensor = torch.ones(batch_size,seq_len, embed_dim)
        zeros_filled = torch.cat([zeros_tensor, ones_tensor], dim=-1)
        # 根据 y_seq 创建索引张量
        y_indices = y_seq == 1
        # 使用索引张量填充对应位置
        pro_emb = zeros_filled.clone()
        pro_emb[y_indices] = torch.cat([torch.ones(embed_dim),torch.zeros(embed_dim)], dim=-1)
        pro_embeddings_x = self.pro_embeddings(pro_seq)
        pro_emb[:, :, :embed_dim] *= pro_embeddings_x
        pro_emb[:, :, embed_dim:] *= pro_embeddings_x
        lstm_out, (hn, cn) = self.lstm(pro_emb)
        pred = self.linear(lstm_out)
        return self.sigmoid(pred)


model = LSTM(pro_num, embed_dim, hidden_dim).to(device)

# 损失函数和优化器
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def compute_metrics(preds, targets):
    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0
    acc = metrics.accuracy_score(targets, preds)
    auc = metrics.roc_auc_score(targets, preds)
    return acc, auc


if train_flag:

    train_steps = math.ceil(len(train_y) / bs)
    test_steps = math.ceil(len(test_y) / bs)

    best_auc = 0

    plot_data1 = []
    plot_data2 = []
    plot_data3 = []
    for epoch in range(epochs):

        model.train()
        train_loss = 0

        for i in tqdm(range(train_steps)):
            start = i * bs
            end = start + bs
            y_batch = torch.from_numpy(train_y[start:end]).float().to(device)
            # pro_batch = torch.from_numpy(train_problem[start:end]).long().to(device)
            pro_batch = torch.from_numpy(train_skill[start:end]).long().to(device)
            len_batch = torch.from_numpy(train_real_len[start:end]).long().to(device)
            apred = model(pro_batch, y_batch, len_batch)
            problem_ids = torch.cat([pro_batch[i, 1:l] for i, l in enumerate(len_batch)], dim=-1)
            pred = torch.cat([apred[i,:l-1] for i,l in enumerate(len_batch)],dim=0).gather(1,problem_ids.view(-1,1)).flatten()
            target = torch.cat([y_batch[i,1:l] for i,l in enumerate(len_batch)],dim=-1)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= train_steps

        model.eval()
        test_preds, test_targets = [], []
        with torch.no_grad():
            for i in tqdm(range(test_steps)):
                start = i * bs
                end = start + bs
                y_batch = torch.from_numpy(test_y[start:end]).float().to(device)
                # pro_batch = torch.from_numpy(test_problem[start:end]).long().to(device)
                pro_batch = torch.from_numpy(test_skill[start:end]).long().to(device)
                len_batch = torch.from_numpy(test_real_len[start:end]).long().to(device)

                apred = model(pro_batch, y_batch, len_batch).squeeze(-1)
                problem_ids = torch.cat([pro_batch[i, 1:l ] for i, l in enumerate(len_batch)], dim=-1)
                pred = torch.cat([apred[i, :l-1] for i, l in enumerate(len_batch)], dim=0).gather(1,problem_ids.view(-1, 1)).flatten()
                target = torch.cat([y_batch[i, 1:l] for i, l in enumerate(len_batch)], dim=-1)
                test_preds.append(pred.detach().cpu().numpy())
                test_targets.append(target.detach().cpu().numpy())

        test_preds = np.concatenate(test_preds, axis=0)
        test_targets = np.concatenate(test_targets, axis=0)
        test_acc, test_auc = compute_metrics(test_preds, test_targets)

        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.5f}, Test AUC: {test_auc:.3f}, Test ACC: {test_acc:.3f}')
        plot_data1.append(train_loss)
        plot_data2.append(test_auc)
        plot_data3.append(test_acc)

        if test_auc > best_auc:
            best_auc = test_auc
            torch.save(model.state_dict(), os.path.join(data_folder, 'dkt_pytorch.pt'))

    import matplotlib.pyplot as plt

    # 示例数据
    x = list(range(len(plot_data1)))
    y_class1 = plot_data1
    y_class2 = plot_data2
    y_class3 = plot_data3

    # 创建散点图
    plt.scatter(x, y_class1, color='red', label='train_loss')
    plt.scatter(x, y_class2, color='blue', label='Test AUC')
    plt.scatter(x, y_class3, color='green', label='test_acc')

    # 添加标题和坐标轴标签
    plt.title('Scatter Plot with 3 Classes')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 添加图例
    plt.legend()

    # 显示图表
    plt.show()

else:
    model.load_state_dict(torch.load(os.path.join(data_folder, 'dkt_pytorch.pt')))
    model.eval()

    # 得到问题嵌入向量
    pro_embed_trained = model.pro_embeddings.weight.data[1:].cpu().numpy()
    np.savez(os.path.join(data_folder, 'pro_embed_pytorch.npz'), pro_final_repre=pro_embed_trained)