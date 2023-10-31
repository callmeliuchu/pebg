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


def train_test_split(data, split=0.8):
    n_samples = data[0].shape[0]
    split_point = int(n_samples * split)
    train_data = [d[:split_point] for d in data]
    test_data = [d[split_point:] for d in data]
    return train_data, test_data


# 数据处理
data_folder = "assist09_test"
data = np.load(os.path.join(data_folder, data_folder + '.npz'))
y, skill, problem, real_len = data['y'], data['skill'], data['problem'], data['real_len']
skill_num, pro_num = data['skill_num'], data['problem_num']
print('problem number %d, skill number %d' % (pro_num, skill_num))

# 划分训练集和测试集
train_data, test_data = train_test_split([y, skill, problem, real_len])
train_y, train_skill, train_problem, train_real_len = train_data
test_y, test_skill, test_problem, test_real_len = test_data

# 嵌入初始化
embed_data = np.load(os.path.join(data_folder, 'embedding_200.npz'))
_, _, pre_pro_embed = embed_data['pro_repre'], embed_data['skill_repre'], embed_data['pro_final_repre']
print(pre_pro_embed.shape, pre_pro_embed.dtype)

# 超参数
epochs = 200
bs = 128
embed_dim = pre_pro_embed.shape[1]
hidden_dim = 128
lr = 0.001
use_pretrain = True
train_embed = True


# 模型定义
class LSTM(nn.Module):
    def __init__(self, pro_num, embed_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.pro_embeddings = nn.Embedding(pro_num, embed_dim, padding_idx=0)
        lstm_input_dim = embed_dim + 1
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

        if use_pretrain:
            pretrained_weight = torch.from_numpy(pre_pro_embed)
            self.pro_embeddings.weight.data.copy_(pretrained_weight)
            self.pro_embeddings.weight.requires_grad = train_embed

    def forward(self, pro_seq, y_seq, pro_len):
        pro_emb = self.pro_embeddings(pro_seq)
        y_emb = y_seq.unsqueeze(-1)  # [batch, seq_len, 1]
        lstm_input = torch.cat([pro_emb, y_emb], dim=-1)
        lstm_out, (hn, cn) = self.lstm(lstm_input)
        pred = self.linear(lstm_out)
        return pred


model = LSTM(pro_num, embed_dim, hidden_dim).to(device)

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def compute_metrics(preds, targets):
    preds[preds > 0] = 1
    preds[preds <= 0] = 0
    acc = metrics.accuracy_score(targets, preds)
    auc = metrics.roc_auc_score(targets, preds)
    return acc, auc


if train_flag:

    train_steps = math.ceil(len(train_y) / bs)
    test_steps = math.ceil(len(test_y) / bs)

    best_auc = 0
    for epoch in range(epochs):

        model.train()
        train_loss = 0

        for i in tqdm(range(train_steps)):
            start = i * bs
            end = start + bs
            y_batch = torch.from_numpy(train_y[start:end]).float().to(device)
            pro_batch = torch.from_numpy(train_problem[start:end]).long().to(device)
            len_batch = torch.from_numpy(train_real_len[start:end]).long().to(device)
            apred = model(pro_batch, y_batch, len_batch).squeeze(-1)
            pred = torch.cat([apred[i,:l-1] for i,l in enumerate(len_batch)],dim=-1)
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
                pro_batch = torch.from_numpy(test_problem[start:end]).long().to(device)
                len_batch = torch.from_numpy(test_real_len[start:end]).long().to(device)

                apred = model(pro_batch, y_batch, len_batch).squeeze(-1)
                pred = torch.cat([apred[i, :l - 1] for i, l in enumerate(len_batch)], dim=-1)
                target = torch.cat([y_batch[i, 1:l] for i, l in enumerate(len_batch)], dim=-1)
                test_preds.append(pred.detach().cpu().numpy())
                test_targets.append(target.detach().cpu().numpy())

        test_preds = np.concatenate(test_preds, axis=0)
        test_targets = np.concatenate(test_targets, axis=0)
        test_acc, test_auc = compute_metrics(test_preds, test_targets)

        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.5f}, Test AUC: {test_auc:.3f}, Test ACC: {test_acc:.3f}')

        if test_auc > best_auc:
            best_auc = test_auc
            torch.save(model.state_dict(), os.path.join(data_folder, 'dkt_pytorch.pt'))

else:
    model.load_state_dict(torch.load(os.path.join(data_folder, 'dkt_pytorch.pt')))
    model.eval()

    # 得到问题嵌入向量
    pro_embed_trained = model.pro_embeddings.weight.data[1:].cpu().numpy()
    np.savez(os.path.join(data_folder, 'pro_embed_pytorch.npz'), pro_final_repre=pro_embed_trained)