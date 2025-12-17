import torch, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import InMemoryDataset, Data


df = pd.read_csv('./exported_data.dat', header=None, low_memory=False)
df.columns = ['head_entity_id', 'tail_entity_id', 'relation_id', 'head_label', 'tail_label']

class Pre_Data_GAT(InMemoryDataset):
    def __init__(self, root, transfrom=None, pre_transform=None):
        super(Pre_Data_GAT, self).__init__(root, transfrom, pre_transform)
        if os.path.exists(self.processed_paths[0]):
            self.data = torch.load(self.processed_paths[0], weights_only=False)
        else:
            print('Processed file not found. Processing dataset...')
            self.process()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['GAT_data.dataset'] # 输入处理后的数据名称

    def download(self):
        pass

    def process(self):
        graph = df

        entity = {}
        head_num_lst = []
        tail_num_lst = []
        entity_label = []

        head_nodes = list(graph.head_entity_id)
        tail_nodes = list(graph.tail_entity_id)
        entity_nodes = list(set(head_nodes + tail_nodes))
        print(len(entity_nodes))
        entity_nodes = sorted(entity_nodes)
        entity_nodes_matrix = torch.tensor(entity_nodes.copy())
        torch.save(entity_nodes_matrix, './pre_data/edge_nodes_matrix.pt')

        for id in entity_nodes:
            e_label = graph.loc[graph.head_entity_id == id, ['head_label']].head_label.drop_duplicates().values
            e_label = list(e_label)
            if e_label == []:
                e_label = graph.loc[graph.tail_entity_id == id, ['tail_label']].tail_label.drop_duplicates().values
                e_label = list(e_label)
            element = e_label[0]
            entity_label.append(element)

        entity['id'] = entity_nodes.copy()
        entity['label'] = entity_label
        entity['num'] = LabelEncoder().fit_transform(entity['id'])
        entity = pd.DataFrame(entity)

        graph = graph.reset_index(drop=True)
        entity = entity.reset_index(drop=True)
        for id in graph['head_entity_id']:
            head_num = list(entity.loc[entity.id == id, ['num']].num)
            head_num_lst.append(head_num[-1])
        for id in graph['tail_entity_id']:
            tail_num = list(entity.loc[entity.id == id, ['num']].num)
            tail_num_lst.append(tail_num[-1])

        graph['head_num_lst'] = head_num_lst
        graph['tail_num_lst'] =  tail_num_lst

        source_nodes = graph.head_num_lst
        target_nodes = graph.tail_num_lst

        node_features = entity_nodes.copy()

        node_features = torch.LongTensor(node_features).unsqueeze(1)
        edge_index = np.stack([source_nodes, target_nodes], axis=0)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x = node_features

        y = torch.tensor(entity.label.values, dtype=torch.long)

        full_data = Data(x=x, edge_index=edge_index, y=y)

        torch.save(full_data, self.processed_paths[0])
        print('Data processing complete and file saved')

class Pre_Data_Tri_GNN(InMemoryDataset):
    def __init__(self, root, transfrom=None, pre_transform=None):
        super(Pre_Data_Tri_GNN, self).__init__(root, transfrom, pre_transform)
        if os.path.exists(self.processed_paths[0]):
            self.data = torch.load(self.processed_paths[0], weights_only=False)
        else:
            print('Processed file not found. Processing dataset...')
            self.process()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['Tri_GNN_data.dataset'] # 输入处理后的数据名称

    def download(self):
        pass

    def process(self):
        graph = df

        entity = {}
        head_num_lst = []
        tail_num_lst = []


        head_nodes = list(graph.head_entity_id)
        tail_nodes = list(graph.tail_entity_id)
        entity_nodes = list(set(head_nodes + tail_nodes))
        entity_nodes = sorted(entity_nodes)

        entity['id'] = entity_nodes
        entity['num'] = LabelEncoder().fit_transform(entity['id'])
        entity = pd.DataFrame(entity)

        graph = graph.reset_index(drop=True)
        entity = entity.reset_index(drop=True)
        for id in graph['head_entity_id']:
            head_num = list(entity.loc[entity.id == id, ['num']].num)
            head_num_lst.append(head_num[-1])
        for id in graph['tail_entity_id']:
            tail_num = list(entity.loc[entity.id == id, ['num']].num)
            tail_num_lst.append(tail_num[-1])

        graph['head_num_lst'] = head_num_lst
        graph['tail_num_lst'] =  tail_num_lst

        source_nodes = graph.head_num_lst
        target_nodes = graph.tail_num_lst

        node_features = entity_nodes

        node_features = torch.LongTensor(node_features).unsqueeze(1)
        edge_index = np.stack([source_nodes, target_nodes], axis=0)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x = node_features
        y = torch.IntTensor(graph.relation_id.values)
        data = Data(x=x, edge_index=edge_index, y=y)
        torch.save(data, self.processed_paths[0])
        print('Data processing complete and file saved')
