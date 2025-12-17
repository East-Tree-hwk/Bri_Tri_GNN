import copy
from pre_data.Tri_GNN_pre_data import *

def find_unconnected_pairs(h_nodes, t_nodes):
    en_nodes = list(set(h_nodes + t_nodes))
    num_nodes = len(en_nodes)
    # 创建邻接矩阵
    node_to_idx = {node: idx for idx, node in enumerate(en_nodes)}
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(len(h_nodes)):
        u = h_nodes[i]
        v = t_nodes[i]
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        adj_matrix[u_idx, v_idx] = 1
        adj_matrix[v_idx, u_idx] = 1
    # print(adj_matrix)
    pairs = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i][j] == 0:
                pairs.append((i, j))
    # print(pairs)
    return pairs

def head_tail_label_lst(label_ten, h_ten, t_ten):
    # 获取正向和逆向标签
    label_lst = label_ten
    label_set = set(label_lst.copy())
    c_label_lst = label_lst.copy()
    label_lst = (2 * len(label_set) - 1) - torch.tensor(label_lst.copy())
    a_label_lst = label_lst.tolist()

    h_t_lst = torch.cat((h_ten, t_ten), dim=1).tolist()
    t_h_lst = torch.cat((t_ten, h_ten), dim=1).tolist()
    edge_matrix = copy.deepcopy(h_t_lst)
    torch.save(edge_matrix, './pre_data/edge_matrix.pt')

    # 正向数据集
    h_t_label_lst = []
    for idx in range(len(c_label_lst)):
        c_lst = []
        h_t = h_t_lst[idx]
        c_lst.append(h_t)
        c_label = c_label_lst[idx]
        c_lst.append(c_label)
        h_t_label_lst.append(c_lst)

    # 逆向数据集
    t_h_label_lst = []
    for idx in range(len(a_label_lst)):
        a_lst = []
        t_h = t_h_lst[idx]
        a_lst.append(t_h)
        a_label = a_label_lst[idx]
        a_lst.append(a_label)
        t_h_label_lst.append(a_lst)

    return h_t_label_lst, t_h_label_lst, edge_matrix

def StrChange(edge, dtype=None):
    if dtype == 'a':
        num1 = edge[1]
        num2 = edge[0]
        return '[' + str(num1) + ', ' + str(num2) + ']' + '=' + '1000'
    elif dtype == 'c':
        num1 = edge[0]
        num2 = edge[1]
        return '[' + str(num1) + ', ' + str(num2) + ']' + '=' + '1000'
    else:
        num1 = edge[0]
        num2 = edge[1]
        return str(num1) + '=' + str(num2)

def write_file(h_t_label_lst, t_h_label_lst, pairs):
    f_line = len(h_t_label_lst)
    with open('./Tri_GNN_data/Tri_GNN_data_full.txt', 'w', encoding='utf-8') as file:
        for idx in range(0, f_line):
            c_ful_lst = h_t_label_lst[idx]
            file.write(''.join(StrChange(c_ful_lst)) + '\n')
            a_ful_lst = t_h_label_lst[idx]
            file.write(''.join(StrChange(a_ful_lst)) + '\n')
    print('Tri_GNN_data_Full.TXT加载完成')

    with open('./Tri_GNN_data/Tri_GNN_data_predict_c.txt', 'w', encoding='utf-8') as file:
        for edge in pairs:
            file.write(''.join(StrChange(edge, 'c')) + '\n')
    print('Tri_GNN_data_Train_Predict_c.TXT加载完成')

    with open('./Tri_GNN_data/Tri_GNN_data_predict_a.txt', 'w', encoding='utf-8') as file:
        for edge in pairs:
            file.write(''.join(StrChange(edge, 'a')) + '\n')
    print('Tri_GNN_data_Train_Predict_a.TXT加载完成')

def get_Tri_GNN_data(data_Tri_GNN):
    # 提取头部、尾部实体以及标签
    label_ten = data_Tri_GNN.y.tolist()
    h_ten = data_Tri_GNN.edge_index[0].unsqueeze(1)
    t_ten = data_Tri_GNN.edge_index[1].unsqueeze(1)

    # 训练集、验证集、测试集、总数据集的准备
    h_t_label_lst, t_h_label_lst ,edge_matrix = head_tail_label_lst(label_ten, h_ten, t_ten)

    # 预测集的准备
    h_nodes = h_ten.squeeze().tolist()
    t_nodes = t_ten.squeeze().tolist()
    pairs = find_unconnected_pairs(h_nodes, t_nodes)

    write_file(h_t_label_lst, t_h_label_lst, pairs)

    return edge_matrix
