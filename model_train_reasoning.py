import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from model.Tri_GNN import Tri_GNN
from pre_data.GAT_pre_data import Pre_Data_Tri_GNN
from pre_data.Tri_GNN_pre_data import Tri_GNN_Dataset
from Tri_GNN_process_data import get_Tri_GNN_data


def Tri_GNN_get_train_data(train_dataset, val_dataset, test_dataset, bs):
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs*2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs*2, shuffle=False)
    return train_loader, val_loader, test_loader

def Tri_GNN_Train(epoch, model, loss_func, opt, train_dl, valid_dl, test_dl):
    best_val_loss = float('inf') # 初始化为正无穷大
    sche = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.9, patience=5)
    loss_x = []
    tra_loss_lst = []
    val_loss_lst = []
    test_loss_lst = []
    acc_x = []
    tra_acc_lst = []
    val_acc_lst  = []
    test_acc_lst = []
    for ep in range(1, epoch + 1):
        model.train()
        train_loss, correct = 0, 0
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad()

            out = model(xb)
            loss = loss_func(out, yb)
            loss.backward()
            opt.step()

            train_loss += loss.item()
            correct += out.argmax(dim=1).eq(yb).sum().item()
        tra_loss = train_loss / len(train_dl)
        tra_acc = correct  / len(train_dl.dataset)
        val_loss, val_acc = Tri_GNN_Test(model, loss_func, valid_dl)
        sche.step(val_loss)

        test_loss, test_acc = Tri_GNN_Test(model, loss_func, test_dl)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './best_model/Tri_GNN_best_model.pth')
            print('保存模型')
        tra_loss_lst.append(tra_loss)
        val_loss_lst.append(val_loss)
        test_loss_lst.append(test_loss)
        loss_x.append(ep)
        tra_acc_lst.append(tra_acc)
        val_acc_lst.append(val_acc)
        test_acc_lst.append(test_acc)
        acc_x.append(ep)

        print(f'当前epoch：{ep}, 当前学习率：{opt.param_groups[0]["lr"]:.6f}, '
              f'训集损失：{tra_loss:.4f}, 训集精度：{tra_acc:.4f}%, '
              f'验证集损失：{val_loss:.4f}, 验证集精度：{val_acc:.4f}%, '
              f'测试集损失：{test_loss:.4f}, 测试集精度：{test_acc:.4f}%')

    return tra_loss_lst, val_loss_lst, test_loss_lst, loss_x, tra_acc_lst, val_acc_lst, test_acc_lst, acc_x

def Tri_GNN_Test(model, loss_func, dl):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(x)
            loss = loss_func(out, y)
            total_loss += loss.item()
            correct += out.argmax(dim=1).eq(y).sum().item()

    return total_loss / len(dl), correct/ len(dl.dataset)

def model_evalate(model, dl):
    re_rank = 0
    hit_1_num = 0
    hit_3_num = 0
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(x)
            output = torch.softmax(out, dim=1)
            # MRR指标
            re_rank += sum_rank(output, y)
            # hit@1指标
            hit_1_num += output.argmax(dim=1).eq(y).sum().item()
            # hit@3指标
            hit_3_num += hit_3(output, y)

    return re_rank / len(dl.dataset), hit_1_num / len(dl.dataset), hit_3_num / len(dl.dataset)

def sum_rank(out,y):
    sorted_indices = torch.argsort(out, dim=1, descending=True)
    ranks = []
    for i in range(len(y)):
        rank = (sorted_indices[i] == y[i]).nonzero(as_tuple=True)[0].item() + 1  # 排名从1开始
        ranks.append(rank)
    reciprocal_ranks = [1.0 / rank for rank in ranks]
    return sum(reciprocal_ranks)

def hit_3(out, y):
    top3_indices = torch.topk(out, k=3, dim=1).indices
    hits = []
    for i in range(len(y)):
        if y[i] in top3_indices[i]:
            hits.append(1.0)
        else:
            hits.append(0.0)
    return sum(hits)

def get_Tri_GNN_model():
    model = Tri_GNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=1e-4)
    return model, optimizer

def save_list(lst, name):
    name = './data/' + name + '.pkl'
    with open(name, 'wb') as f:
        pickle.dump(lst, f)
    print('保存完成')

# Tri_GNN部分
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_Tri_GNN = Pre_Data_Tri_GNN(root='data/')
edge_matrix = get_Tri_GNN_data(data_Tri_GNN)
edge = edge_matrix
full_dataset = Tri_GNN_Dataset(ann_file='./Tri_GNN_data/Tri_GNN_data_full.txt', edge=edge)

train_size = int(0.8 * len(full_dataset))  # 80%训练
val_size = int(0.1 * len(full_dataset))   # 10%验证
test_size = len(full_dataset) - train_size - val_size  # 剩余10%测试

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))  # 设置随机种子保证可复现

bs = 20

train_dl, valid_dl, test_dl = Tri_GNN_get_train_data(train_dataset, val_dataset, test_dataset, bs)
# 创建Tri_GNN推理模型
model_Tri_GNN, opt = get_Tri_GNN_model()
model_Tri_GNN = model_Tri_GNN.to(device)
# Tri_GNN推理模型训练
loss_func = F.cross_entropy()
tra_loss_lst, val_loss_lst, test_loss_lst, loss_x, tra_acc_lst, val_acc_lst, test_acc_lst, acc_x = Tri_GNN_Train(100, model_Tri_GNN, loss_func, opt, train_dl, valid_dl, test_dl)
save_list(tra_loss_lst, 'tra_loss')
save_list(val_loss_lst, 'val_loss')
save_list(test_loss_lst, 'test_loss')
save_list(loss_x, 'loss_x')
save_list(tra_acc_lst, 'tra_acc')
save_list(val_acc_lst, 'val_acc')
save_list(test_acc_lst, 'test_acc')
save_list(acc_x, 'acc_x')
