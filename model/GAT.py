import torch.nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATNET(torch.nn.Module):
    def __init__(self, num_embed, embed_dim, h1_feats, h2_feats, h3_feats, out_feats):
        super(GATNET, self).__init__()
        self.entity_embedding = torch.nn.Embedding(num_embeddings=num_embed, embedding_dim=embed_dim)
        self.conv1 = GATConv(embed_dim, h1_feats, heads=5, concat=True)
        self.conv2 = GATConv(h1_feats*5, h2_feats, heads=5, concat=True)
        self.conv3 = GATConv(h2_feats*5, h3_feats, heads=5, concat=True)
        self.conv4 = GATConv(h3_feats*5, out_feats, heads=2, concat=False)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.entity_embedding(x)
        x = x.squeeze(1)
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.leaky_relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)

        return x
