import torch
from model.GAT import GATNET
from pre_data.GAT_pre_data import df, Pre_Data_GAT

def GAT_1_Train(model, data):
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-05)
    schudler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.8)
    min_epochs = 10
    best_val_loss = float('inf')
    model.train()
    for epoch in range(1, 151):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(out, data.y)
        loss.backward(retain_graph=True)
        optimizer.step()
        schudler.step()
        # 验证集
        val_loss, val_acc = GAT_1_Test(model, data)
        if val_loss < best_val_loss and epoch + 1 > min_epochs:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './best_model/GAT_best_model.pth')
            print(f'GAT1:epoch:{epoch:03d}, 训练集损失:{loss.item():.4f}, 验证集损失:{val_loss:.4f}, 验证集精度:{val_acc:.4f}%, 保存最佳模型')

@torch.no_grad()
def GAT_1_Test(model, data):
    model.eval()
    out = model(data)
    loss_function = torch.nn.CrossEntropyLoss()
    loss = loss_function(out, data.y)
    _, pred = out.max(dim=1)
    correct = int(pred.eq(data.y).sum().item())
    acc = 100 * correct / int(len(data.y))
    model.train()

    return loss.item(), acc

# GAT部分
# 获取GAT自身特征提取模型需要的数据
GAT_full_data = Pre_Data_GAT(root='data/')
# GAT模型准备
head_max = df.head_entity_id.max() + 1
tail_max = df.tail_entity_id.max() + 1
if head_max >= tail_max:
    num_embed = head_max
else:
    num_embed = tail_max
print(num_embed)
# 创建GAT自身特征提取模型
model_GAT = GATNET(num_embed, 64, 50, 30, 15, 5)
# GAT自身特征模型训练
GAT_1_Train(model_GAT, GAT_full_data)
# 加载GAT自身特征最佳模型并测试准确率
model_GAT.load_state_dict(torch.load('./best_model/GAT_best_model.pth'))
Tri_GNN_matrix = model_GAT(GAT_full_data)
# 保存GAT提取的矩阵
torch.save(Tri_GNN_matrix, './pre_data/Tri_GNN_matrix.pt')
# print(len(Tri_GNN_matrix))
_, pred = Tri_GNN_matrix.max(dim=1)
correct_1 = pred.eq(GAT_full_data.y).sum().item()
acc_1 = 100 * correct_1 / int(len(GAT_full_data.y))
print(f'GAT_1准确率：{acc_1:.4f}%')
