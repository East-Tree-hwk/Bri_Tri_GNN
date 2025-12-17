import torch
import json
import numpy as np
from collections import defaultdict, deque
from torch.utils import data


class Tri_GNN_Dataset(data.Dataset):
    def __init__(self, ann_file, edge, transform=None):
        self.Tri_GNN_matrix = torch.load('./pre_data/Tri_GNN_matrix.pt')
        self.ann_file = ann_file
        self.edges = edge
        self.entity_label = self.load_annotations()
        self.entity = [entity for entity in list(self.entity_label.keys())]
        self.label = [label for label in list(self.entity_label.values())]
        self.transform = transform

    def __len__(self):
        return len(self.entity)

    def __getitem__(self, idx):
        entities_lst = []
        entities = json.loads(self.entity[idx])# 将strlst转成lst
        label = self.label[idx]
        edges = self.edges
        a = entities[0]
        b = entities[1]
        paths = self.find_paths_between_entities(edges, a, b)
        c_lst = self.select_middle_entities(paths)
        entity_h = self.Tri_GNN_matrix[a]
        entity_t = self.Tri_GNN_matrix[b]
        for c in c_lst:
            entity_m = self.Tri_GNN_matrix[c]
            entities = torch.cat((entity_h, entity_t, entity_m), dim=0)
            entities_lst.append(entities)
        label = torch.from_numpy(np.array(label))
        return entities_lst, label

    def load_annotations(self):
        data_infos = {}
        with open(self.ann_file) as f:
            samples = [x.strip().split('=') for x in f.readlines()]
            for entitynum, gt_label in samples:
                data_infos[entitynum] = np.array(gt_label, dtype=np.int64)
        return data_infos

    @staticmethod
    def bulid_graph(edges):
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        return graph

    @staticmethod
    def bfs_path(graph, start, goal, max_depth=3):
        queue = deque([(start, [start], 0)])
        visited = {start}

        if goal in graph.get(start, []):
            graph[start].remove(goal)

        while queue:
            node, path, depth = queue.popleft()

            if depth > max_depth:
                continue

            if node == goal:
                return path

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor], depth + 1))
        return []

    def find_paths_between_entities(self, edges, a, b):
        graph = self.bulid_graph(edges)
        path = self.bfs_path(graph, a, b)
        return path

    @staticmethod
    def select_middle_entities(path):
        middle_lst = []
        for p in path:
            path_len = len(p)
            if path_len == 0:
                return 0
            elif path_len == 2:
                return path[0]
            elif path_len == 3:
                return path[1]
            elif path_len % 2 != 0:
                idx = (path_len - 1) // 2
            else:
                idx = path_len // 2
            middle_lst.append(path[idx])
        return middle_lst
