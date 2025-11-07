import torch
import torch_geometric
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import lightning.pytorch as pl
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn as nn
from torch_geometric.utils import  add_self_loops, softmax
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn import MessagePassing, GCNConv, GINConv,GATConv, GATv2Conv, GraphConv, GINEConv
# from torchmetrics.classification import BinaryF1Score, BinaryAUROC
from torchmetrics import AUROC, F1Score, Accuracy

### MLP 

class MLPModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.n_feat = in_channels
        self.n_classes = out_channels
        self.n_hidden = hidden_channels
        self.n_layers = num_layers

        self.layers = [Linear(self.n_feat, self.n_hidden)]
        if self.n_layers > 2:
            for _ in range(self.n_layers-2):
                self.layers.append(Linear(self.n_hidden, self.n_hidden))
        self.layers.append(Linear(self.n_hidden, self.n_classes))
        self.layers = torch.nn.Sequential(*self.layers)
    
    def forward(self, x, *args, **kwargs):
        for layer in range(self.n_layers):
            x = self.layers[layer](x)
            if layer == self.n_layers - 1:
                #remove relu for the last layer
                x = F.dropout(x, self.dropout, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.dropout, training=self.training)
        return x

class MLPModule(pl.LightningModule):
    def __init__(self, c_out=2, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self.model = MLPModel(**model_kwargs)

    def forward(self, data, mode="train"):
        x = data.x
        out = self.model(x)

        loss = self.loss_module(out, data.y)
        pred = out.argmax(dim=1)
        acc = (pred == data.y).sum().float() / len(pred)
        return loss, acc

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log("test_acc", acc)

### GCN ########################################################################

gnn_layer_by_name = {"GCN": GCNConv, "GraphConv": GraphConv, 'GAT': GATConv, 'GIN': GINEConv, 'GAT2': GATv2Conv}

################################################################################


class GNNModel(torch.nn.Module):
    def __init__(self, layer_name, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5, **kwargs):
        super().__init__()
        self.layer_name = layer_name
        self.dropout = dropout
        self.n_feat = in_channels
        self.n_classes = out_channels
        self.n_hidden = hidden_channels
        self.n_layers = num_layers

        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = self.n_feat, self.n_hidden
#          if layer_name == 'GIN':
#             print('getting here A')
#             for _ in range(self.n_layers - 1):
#                 NN = torch.nn.Sequential(
#                             torch.nn.Linear(in_channels, out_channels),
#                             torch.nn.ReLU()
#                             )
#                 print(type(NN))
#                 layers += [
#                     gnn_layer(nn = NN, edge_dim = 1, **kwargs),
#                     torch.nn.ReLU(inplace=True),
#                     torch.nn.Dropout(self.dropout),
#                     torch.nn.BatchNorm1d(out_channels)
#                 ]
#                 in_channels = self.n_hidden
#             NN_final = torch.nn.Sequential(
#             torch.nn.Linear(in_channels, self.n_classes),
#             torch.nn.ReLU(),
#             )
#             layers += [gnn_layer(nn = NN_final, edge_dim = 1,**kwargs)]
#         else:
        for _ in range(self.n_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=out_channels,edge_dim = 1,  **kwargs),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(self.dropout),
                torch.nn.BatchNorm1d(out_channels)
            ]
            in_channels = self.n_hidden
        layers += [gnn_layer(in_channels=in_channels, out_channels=self.n_classes ,edge_dim = 1, **kwargs)]
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_info):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for layer in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(layer, torch_geometric.nn.MessagePassing):
                if (self.layer_name == 'GAT') | (self.layer_name == 'GAT2'): 
                    x = layer(x = x, edge_index = edge_index, edge_attr = edge_info)
                else:
                    x = layer(x, edge_index, edge_weight=edge_info)
                    
            else:
                x = layer(x)
        return x
    

    

"""Attention pooling module"""
class Attention_module(Aggregation):
    def __init__(self, D1 = 20, D2 = 10):
        super(Attention_module, self).__init__()
        self.attention_Tanh = [
            nn.Linear(D1, D2),
            nn.Tanh()]
        
        self.attention_Sigmoid = [
            nn.Linear(D1, D2),
            nn.Sigmoid()]

        self.attention_Tanh = nn.Sequential(*self.attention_Tanh)
        self.attention_Sigmoid = nn.Sequential(*self.attention_Sigmoid)
        self.attention_Concatenate = nn.Linear(D2, 1)

    def forward(self, x, index=None, ptr=None, dim_size = None, dim= -2): # 20->10->2
        tanh_res = self.attention_Tanh(x)
        sigmoid_res = self.attention_Sigmoid(x)
        Attention_score = tanh_res.mul(sigmoid_res)
        Attention_score = self.attention_Concatenate(Attention_score)  # N x n_classes

        # return Attention_score, x
        gate = softmax(Attention_score, index, ptr, dim_size, dim)
        return self.reduce(gate * x, index, ptr, dim_size, dim)

"""Initial weights"""
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class GraphLevelGNN(pl.LightningModule):
    def __init__(self, model_name, num_PPI_type, c_out=2, graph_pooling="mean", embedding=True, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        c_hidden = model_kwargs.get('in_channels', 16) # Output dimension of GCN layers
        out_channels = model_kwargs.get('out_channels', 16)

        if embedding:
            self.x_embedding = torch.nn.Linear(num_PPI_type, c_hidden)
            torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)
        else: 
            self.x_embedding = nn.Identity()

        if model_name == "MLP":
            print('Using MLP')
            self.model = MLPModel(**model_kwargs)
        else:
            print('Using GNN')
            print(model_name)
            self.model = GNNModel(layer_name=model_name, **model_kwargs)
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(out_channels, out_channels//2), 
                                        torch.nn.Dropout(0.5), torch.nn.Linear(out_channels//2, c_out))
        
        self.loss_module = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="binary")
        self.train_auroc = AUROC(task="binary")
        self.train_f1 = F1Score(task="binary")
        self.valid_acc = Accuracy(task="binary")
        self.valid_auroc = AUROC(task="binary")
        self.valid_f1 = F1Score(task="binary")
#         self.test_acc = Accuracy(task="binary")
#         self.test_auroc = AUROC(task="binary")
#         self.test_f1 = F1Score(task="binary")

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Linear(out_channels, 1))
        elif graph_pooling == "attention2":
            self.pool = Attention_module(D1 = out_channels, D2=out_channels//2)
        else:
            raise ValueError("Invalid graph pooling type.")


    def forward(self, data, mode="train"):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_weight
        batch = data.batch if 'batch' in data else torch.zeros((len(data.x),)).long().to(data.x.device)

        x = self.x_embedding(x.float())
        
        x = self.model(x, edge_index, edge_info=edge_attr)
        
        x = self.pool(x, batch) 
        out = self.head(x)

        # pred = out.argmax(dim=1)
        # acc = (pred == data.y).sum().float() / len(pred)
        # f1 = self.metricf1(pred, data.y)
        # auc = self.metricauc(pred, data.y)
        return out
    # def forward(self, x, edge_index, edge_attr, data, mode="train"):
    #     # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_weight
    #     batch = data.batch if 'batch' in data else torch.zeros((len(data.x),)).long().to(data.x.device)

    #     x = self.x_embedding(x.float())
        
    #     x = self.model(x, edge_index, edge_info=edge_attr)
        
    #     x = self.pool(x, batch) 
    #     out = self.head(x)

    #     # pred = out.argmax(dim=1)
    #     # acc = (pred == data.y).sum().float() / len(pred)
    #     # f1 = self.metricf1(pred, data.y)
    #     # auc = self.metricauc(pred, data.y)
    #     return out

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        y = batch['y']
        out = self.forward(batch)
        loss = self.loss_module(out, batch.y)

        y_hat = out.argmax(dim=1)
        self.train_acc(y_hat, y)
        self.train_auroc(out[:,1], y)
        self.train_f1(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_auc", self.train_auroc, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['y']
        out = self.forward(batch)
        loss = self.loss_module(out, batch.y)

        y_hat = out.argmax(dim=1)
        self.valid_acc(y_hat, y)
        self.valid_auroc(out[:,1], y)
        self.valid_f1(y_hat, y)

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.valid_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.valid_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss
# ##########################################################################################
#     def test_step(self, batch, batch_idx):
#         y = batch['y']
#         out = self.forward(batch)

#         y_hat = out.argmax(dim=1)
#         self.test_acc(y_hat, y)
#         self.test_auroc(out[:,1], y)
#         self.test_f1(y_hat, y)

#         self.log("test_loss", loss, on_step=True, on_epoch=True)
#         self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
#         self.log("test_auc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)
#         self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
#         return loss
# ##########################################################################################