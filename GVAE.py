import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import scanpy as sc
import squidpy as sq

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn.sequential import Sequential
from torch_geometric.sampler import BaseSampler
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix
import sklearn.manifold as manifold
import umap.umap_ as umap

import sys
import os
import os.path as osp
import requests
import tarfile
import argparse
import random
import pickle
from random import sample
from datetime import datetime

from tqdm import tqdm

#Build argument parser
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-v', "--variational", action='store_true', help="Whether to use a variational AE model", default=False)
arg_parser.add_argument('-d', "--dataset", help="Which dataset to use", required=True)
arg_parser.add_argument('-e', "--epochs", type=int, help="How many training epochs to use", default=1)
arg_parser.add_argument('-c', "--cells", type=int, default=-1,  help="How many cells to sample per epoch.")
arg_parser.add_argument('-t', '--type', type=str, choices=['GCN', 'GAT', 'SAGE', 'Linear'], help="Model type to use (GCN, GAT, SAGE, Linear)", default='GCN')
arg_parser.add_argument('-w', '--weight', action='store_true', help="Whether to use distance-weighted edges")
arg_parser.add_argument('-n', '--normalization', choices=["Laplacian", "Normal", "None"], default="None", help="Adjanceny matrix normalization strategy (Laplacian, Normal, None)")
arg_parser.add_argument('-ct', '--add_cell_types', action='store_true', help='Whether to include cell type information')
arg_parser.add_argument('-rm', '--remove_same_type_edges', action='store_true', help="Whether to remove edges between same cell types")
arg_parser.add_argument('-rms', '--remove_subtype_edges', action='store_true', help='Whether to remove edges between subtypes of the same cell')
arg_parser.add_argument('-aggr', '--aggregation_method', choices=['max', 'mean', 'lstm'], help='Which aggregation method to use for GraphSAGE')
arg_parser.add_argument('-th', '--threshold', type=float, help='Distance threshold to use when constructing graph. If neighbors is specified, threshold is ignored.', default=-1)
arg_parser.add_argument('-ng', '--neighbors', type=int, help='Number of neighbors per cell to select when constructing graph. If threshold is specified, neighbors are ignored.', default=-1)
arg_parser.add_argument('-ls', '--latent', type=int, help='Size of the latent space to use', default=4)
arg_parser.add_argument('-hid', '--hidden', type=str, help='Specify hidden layers', default='64,32')
arg_parser.add_argument('-gs', '--graph_summary', action='store_true', help='Whether to calculate a graph summary', default=True)
args = arg_parser.parse_args()
#Define device based on cuda availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Set training mode to true
TRAINING = True


class SAGEEncoder(nn.Module):
    """GraphSAGE-based encoder class

    Inherits:
        nn.Module: Pytorch base class for neural networks

    Attributes:
        input_size: int
            Neural network input layer size
        hidden_1: int
            Size of the first hidden layer in the network
        hidden_2: int
            Size of the second hidden layer in the network
        latent_size: int
            Size of the latent space in the network
        aggregation_method: str
            Neighborhood aggregation method to use in the
            GraphSAGE convolutions (e.g. mean, max, lstm).

    Methods:
        forward(x, edge_index):
            Feeds input x through the encoder layers.
    """

    def __init__(self, input_size, hidden_layers, latent_size, aggregation_method):
        """Initialization function for GraphSAGE-based encoder, constructs 2 GraphSAGE
           convolutional layers, based on the specified layer sizes and aggregation
           method.

        Parameters:
            input_size: int
                Neural network input layer size
            hidden_1: int
                Size of the first hidden layer in the network
            hidden_2: int
                Size of the second hidden layer in the network
            latent_size: int
                Size of the latent space in the network
            aggregation_method: str
                Neighborhood aggregation method to use in the
                GraphSAGE convolutions (e.g. mean, max, lstm).

        """
        super().__init__()

        self.conv1 = SAGEConv(input_size, hidden_layers[0], aggr=aggregation_method)
        hlayers = []
        for i in range(len(hidden_layers)-1):
            hlayers.append((SAGEConv(hidden_layers[i], hidden_layers[i+1], aggr=aggregation_method), 'x, edge_index -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
        self.hlayers = Sequential('x, edge_index', hlayers)
        self.conv2 = SAGEConv(hidden_layers[-1], latent_size, aggr=aggregation_method)
        self.directconv = SAGEConv(input_size, latent_size, aggr=aggregation_method)

        self.num_hidden_layers = len(hidden_layers)

    def forward(self, x, edge_index):
        """Feeds input x constrained by connectivity captured in edge_index through
        the GraphSAGE-based encoder layers. Also applies dropout if training mode
        is on.

        Parameters:
            x: tensor
                Tensor containing the gene expression matrix of the dataset
            edge_index: Tensor
                Pytorch geometric edge_index tensor which contains the connectivity
                of the cells in the input graph

        Returns:
            Tensor with output of the second convolutional layer (latent space)
        """
        if len(hidden_layers) < 1:
            return self.directconv(x, edge_index)
        x = self.conv1(x, edge_index).relu()
        if TRAINING:
            F.dropout(x, p=0.2)
        x = self.hlayers(x, edge_index)
        return self.conv2(x, edge_index)

class VSAGEEncoder(nn.Module):
    """GraphSAGE-based variational encoder class

    Inherits:
        nn.Module: Pytorch base class for neural networks

    Attributes:
        input_size: Neural network input layer size
        hidden_1: Size of the first hidden layer in the network
        hidden_2: Size of the second hidden layer in the network
        latent_size: Size of the latent space in the network
        aggregation_method: Neighborhood aggregation method to use in the
                            GraphSAGE convolutions (e.g. mean, max, lstm).

    Methods:
        forward: Feeds input x through the variational encoder layers.
    """
    def __init__(self, input_size, hidden_layers, latent_size, aggregation_method):
        """Initialization function for GraphSAGE-based variational encoder, constructs 1 GraphSAGE
           convolutional layers, a mu and a log-std GraphSAGE convolutional layer,
           based on the specified layer sizes and aggregation method. Additionally,
           a normal distribution is intialized with mean=0, std=1.

        Parameters:
            input_size: int
                Neural network input layer size
            hidden_1: int
                Size of the first hidden layer in the network
            hidden_2: int
                Size of the second hidden layer in the network
            latent_size: int
                Size of the latent space in the network
            aggregation_method: str
                Neighborhood aggregation method to use in the
                GraphSAGE convolutions (e.g. mean, max, lstm).

        """
        super().__init__()
        self.conv1 = SAGEConv(input_size, hidden_layers[0], aggr=aggregation_method)
        hlayers = []
        for i in range(len(hidden_layers)-1):
            hlayers.append((SAGEConv(hidden_layers[i], hidden_layers[i+1], aggr=aggregation_method), 'x, edge_index -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
        self.hlayers = Sequential('x, edge_index', hlayers)
        self.conv_mu = SAGEConv(hidden_layers[-1], latent_size, aggr=aggregation_method)
        self.conv_logstd = SAGEConv(hidden_layers[-1], latent_size, aggr=aggregation_method)
        self.directconv_mu = SAGEConv(input_size, latent_size, aggr=aggregation_method)
        self.directconv_logstd = SAGEConv(input_size, latent_size, aggr=aggregation_method)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()

        self.num_hidden_layers = len(hidden_layers)

    def forward(self, x, edge_index):
        """Feeds input x constrained by connectivity captured in edge_index through
        the GraphSAGE-based encoder layers. Also applies dropout if training mode
        is on. Latent space vector z is sampled based from a normal distribution
        along with the mu and sigma outputted by the convolutional layers for the mu and std.
        Additionally, the KL-divergence is calculated using the sampled sigma and mu.

        Parameters:
            x: tensor
                Tensor containing the gene expression matrix of the dataset
            edge_index: Tensor
                Pytorch geometric edge_index tensor which contains the connectivity
                of the cells in the input graph

        Returns:
            z: tensor
                Latent space vector sampled
            kl: tensor
                KL-divergence calculated using the sampled sigma and mu
        """
        if self.num_hidden_layers < 1:
            mu = self.directconv_mu(x, edge_index)
            sigma = torch.exp(self.directconv_logstd(x, edge_index))
        else:
            x = self.conv1(x, edge_index).relu()
            if TRAINING:
                F.dropout(x, p=0.2)

            x = self.hlayers(x, edge_index)

            mu = self.conv_mu(x, edge_index)
            sigma = torch.exp(self.conv_logstd(x, edge_index))
        z = mu + sigma * self.N.sample(mu.shape)
        kl  = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, kl

class GATEncoder(nn.Module):
    """Graph Attention Network-based encoder class

    Inherits:
        nn.Module: Pytorch base class for neural networks

    Attributes:
        input_size: Neural network input layer size
        hidden_1: Size of the first hidden layer in the network
        hidden_2: Size of the second hidden layer in the network
        latent_size: Size of the latent space in the network

    Methods:
        forward: Feeds input x through the encoder layers.
    """
    def __init__(self, input_size, hidden_layers, latent_size):
        """Initialization function for GAT-based encoder, constructs 2 GAT
           convolutional layers, based on the specified layer sizes.

        Parameters:
            input_size: int
                Neural network input layer size
            hidden_1: int
                Size of the first hidden layer in the network
            hidden_2: int
                Size of the second hidden layer in the network
            latent_size: int
                Size of the latent space in the network

        """
        super().__init__()
        self.conv1 = GATConv(input_size, hidden_layers[0])
        hlayers = []
        for i in range(len(hidden_layers)-1):
            hlayers.append((GATConv(hidden_layers[i], hidden_layers[i+1]), 'x, edge_index, weight -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
        self.hlayers = Sequential('x, edge_index, weight', hlayers)
        self.conv2 = GATConv(hidden_layers[-1], latent_size)
        self.directconv = GATConv(input_size, latent_size)

        self.num_hidden_layers = len(hidden_layers)

    def forward(self, x, edge_index, weight):
        if self.num_hidden_layers < 1:
            return self.directconv(x, edge_index, weight)
        x = self.conv1(x, edge_index, weight).relu()
        if TRAINING:
            x = F.dropout(x, p=0.2)

        x = self.hlayers(x, edge_index, weight)
        return self.conv2(x, edge_index, weight)


class VGATEncoder(nn.Module):
    def __init__(self, input_size, hidden_layers, latent_size):
        super().__init__()
        self.conv = GATConv(input_size, hidden_layers[0])
        hlayers = []
        for i in range(len(hidden_layers)-1):
            hlayers.append((GATConv(hidden_layers[i], hidden_layers[i+1]), 'x, edge_index, weight -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
        self.hlayers = Sequential('x, edge_index, weight', hlayers)
        self.conv_mu = GATConv(hidden_layers[-1], latent_size)
        self.conv_logstd = GATConv(hidden_layers[-1], latent_size)
        self.directconv_mu = GATConv(input_size, latent_size)
        self.directconv_logstd = GATConv(input_size, latent_size)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()

        self.num_hidden_layers = len(hidden_layers)

    def forward(self, x, edge_index, weight):
        if self.num_hidden_layers < 1:
            mu = self.directconv_mu(x, edge_index, weight)
            sigma = torch.exp(self.directconv_logstd(x, edge_index, weight))
        else:
            x = self.conv1(x, edge_index, weight).relu()
            if TRAINING:
                x = F.dropout(x, p=0.2)

            x = self.hlayers(x, edge_index, weight)
            mu = self.conv_mu(x, edge_index, weight)
            sigma = torch.exp(self.conv_logstd(x, edge_index, weight))
        z = mu + sigma * self.N.sample(mu.shape)
        kl  = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, kl

class GCNEncoder(nn.Module):
    """Graph Convolutional Network-based encoder class

    Inherits:
        nn.Module: Pytorch base class for neural networks

    Attributes:
        input_size: Neural network input layer size
        hidden_1: Size of the first hidden layer in the network
        hidden_2: Size of the second hidden layer in the network
        latent_size: Size of the latent space in the network

    Methods:
        forward: Feeds input x through the encoder layers.
    """
    def __init__(self, input_size, hidden_layers, latent_size):
        """Initialization function for GCN-based encoder, constructs 2 GCN
           convolutional layers, based on the specified layer sizes.

        Parameters:
            input_size: int
                Neural network input layer size
            hidden_1: int
                Size of the first hidden layer in the network
            hidden_2: int
                Size of the second hidden layer in the network
            latent_size: int
                Size of the latent space in the network

        """
        super().__init__()
        self.conv1 = GCNConv(input_size, hidden_layers[0])
        hlayers = []
        for i in range(len(hidden_layers)-1):
            hlayers.append((GCNConv(hidden_layers[i], hidden_layers[i+1]), 'x, edge_index, weight -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
        self.hlayers = Sequential('x, edge_index, weight', hlayers)
        self.conv2 = GCNConv(hidden_layers[-1], latent_size)
        self.directconv = GCNConv(input_size, latent_size)

        self.num_hidden_layers = len(hidden_layers)

    def forward(self, x, edge_index, weight):
        if self.num_hidden_layers < 1:
            return self.directconv(x, edge_index, weight)
        x = self.conv1(x, edge_index, weight).relu()
        if TRAINING:
            x = F.dropout(x, p=0.2)

        x = self.hlayers(x, edge_index, weight)

        return self.conv2(x, edge_index, weight)

class VGCNEncoder(nn.Module):
    def __init__(self, input_size, hidden_layers, latent_size):
        super().__init__()
        self.conv1 = GCNConv(input_size, hidden_layers[0])
        hlayers = []
        for i in range(len(hidden_layers)-1):
            hlayers.append((GCNConv(hidden_layers[i], hidden_layers[i+1]), 'x, edge_index, weight -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
        self.hlayers = Sequential('x, edge_index, weight', hlayers)
        self.conv_mu = GCNConv(hidden_layers[-1], latent_size)
        self.conv_logstd = GCNConv(hidden_layers[-1], latent_size)
        self.directconv_mu = GCNConv(input_size, latent_size)
        self.directconv_logstd = GCNConv(input_size, latent_size)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()

        self.num_hidden_layers = len(hidden_layers)

    def forward(self, x, edge_index, weight):
        if self.num_hidden_layers < 1:
            mu = self.directconv_mu(x, edge_index, weight)
            sigma = self.directconv_logstd(x, edge_index, weight)
        else:
            x = self.conv1(x, edge_index, weight).relu()
            if TRAINING:
                x = F.dropout(x, p=0.2)

            x = self.hlayers(x, edge_index, weight)

            mu = self.conv_mu(x, edge_index, weight)
            sigma = torch.exp(self.conv_logstd(x, edge_index, weight))
        z = mu + sigma * self.N.sample(mu.shape)
        kl  = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, kl


class LinearEncoder(nn.Module):
    """Linear MLP-based encoder class

    Inherits:
        nn.Module: Pytorch base class for neural networks

    Attributes:
        input_size: Neural network input layer size
        hidden_1: Size of the first hidden layer in the network
        hidden_2: Size of the second hidden layer in the network
        latent_size: Size of the latent space in the network

    Methods:
        forward: Feeds input x through the encoder layers.
    """
    def __init__(self, input_size, hidden_layers, latent_size):
        """Initialization function for Linear encoder, constructs 2
           linear layers, based on the specified layer sizes.

        Parameters:
            input_size: int
                Neural network input layer size
            hidden_1: int
                Size of the first hidden layer in the network
            hidden_2: int
                Size of the second hidden layer in the network
            latent_size: int
                Size of the latent space in the network

        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_layers[0])
        self.hlayers = nn.Sequential()
        for i in range(len(hidden_layers)-1):
            self.hlayers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.hlayers.append(nn.Dropout(p=0.2))
        self.linear2 = nn.Linear(hidden_layers[-1], latent_size)
        self.directlinear = nn.Linear(input_size, latent_size)

        self.num_hidden_layers = len(hidden_layers)

    def forward(self, x):
        if self.num_hidden_layers < 1:
            return self.directlinear(x)
        x = self.linear1(x).relu()
        if TRAINING:
            x = F.dropout(x, p=0.2)

        x = self.hlayers(x)
        return self.linear2(x)

class VLinearEncoder(nn.Module):
    def __init__(self, input_size, hidden_layers, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_layers[0])
        self.hlayers = nn.Sequential()
        for i in range(len(hidden_layers)-1):
            self.hlayers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.hlayers.append(nn.Dropout(p=0.2))
        self.linear_mu = nn.Linear(hidden_layers[-1], latent_size)
        self.linear_logstd = nn.Linear(hidden_1, latent_size)
        self.directlinear_mu = nn.Linear(input_size, latent_size)
        self.directlinear_logstd = nn.Linear(input_size, latent_size)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()

        self.num_hidden_layers = len(hidden_layers)


    def forward(self, x):
        if self.num_hidden_layers < 1:
            mu = self.directlinear_mu(x)
            sigma = torch.exp(self.directlinear_logstd(x))
        else:
            x = self.linear1(x).relu()
            if TRAINING:
                x = F.dropout(x, p=0.2)

            x = self.hlayer(x)
            mu = self.linear_mu(x)
            sigma = torch.exp(self.linear_logstd(x))
        z = mu + sigma * self.N.sample(mu.shape)
        kl  = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, kl



class Decoder(nn.Module):
    """Linear decoder classself.linear2(x_hat).relu()
        if TRAINING:
            x_hat = F.dropout(x_hat, p=0.2)

    Inherits:
        nn.Module: Pytorch base class for neural networks

    Attributes:
        output_size: Output layer size of the decoder, equal to the input size of the encoder
        hidden_1: Size of the first hidden layer in the network
        hidden_2: Size of the second hidden layer in the network
        latent_size: Size of the latent space in the network

    Methods:
        forward: Feeds latent space vector for one cell z through the decoder layers.
    """
    def __init__(self, output_size, hidden_layers, latent_size):
        """Initialization function for the decoder, constructs 3 linear
           decoder layers, based on the specified layer sizes.

        Parameters:
            output_size: int
                Output size of the decoder, equal to the input size of the encoder
            hidden_1: int
                Size of the first hidden layer in the network
            hidden_2: int
                Size of the second hidden layer in the network
            latent_size: int
                Size of the latent space in the network

        """
        super().__init__()
        self.linear1 = nn.Linear(latent_size, hidden_layers[-1])
        self.hlayers = nn.Sequential()
        for i in range(len(hidden_layers), 1, -1):
            self.hlayers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i-2]))
            self.hlayers.append(nn.Dropout(p=0.2))
        self.linear2 = nn.Linear(hidden_layers[0], output_size)
        self.directlinear = nn.Linear(latent_size, output_size)

        self.num_hidden_layers = len(hidden_layers)

    def forward(self, z):
        if self.num_hidden_layers < 1:
            return self.directlinear(z)
        x_hat = self.linear1(z).relu()
        if TRAINING:
            x_hat = F.dropout(x_hat, p=0.2)
        x_hat = self.hlayers(x_hat)
        return self.linear2(x_hat)

class GAE(nn.Module):
    """Graph AutoEncoder Aggregation class

    Inherits:
        nn.Module: Pytorch base class for neural networks

    Attributes:
        encoder: Encoder model to use
        decoder: Decoder model to use

    Methods:
        forward: Feeds input x through the encoder to retrieve latent space z and
                 feed this to the decoder to retrieve predicted expression x_hat.
    """
    def __init__(self, encoder, decoder):
        """Initialization function for GCN-based encoder, constructs 2 GCN
           convolutional layers, based on the specified layer sizes.

        Parameters:
            encoder: class
                Encoder model architecture to use, can be any of the encoder classes
            decoder: class
                Decoder model to use (decoder class specified above)

        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, edge_index=None, cell_id=None, weight=None):
        if args.variational == False:
            if args.type == "Linear":
                z = self.encoder(x)
            elif args.type == "GCN":
                z = self.encoder(x, edge_index, weight)
            elif args.type == 'GAT':
                z = self.encoder(x, edge_index, weight)
            elif args.type == 'SAGE':
                z = self.encoder(x, edge_index)
            x_hat = decoder(z[cell_id, :])
            return x_hat
        else:
            if args.type == "Linear":
                z, kl= self.encoder(x)
            elif args.type == "GCN":
                z, kl = self.encoder(x, edge_index, weight)
            elif args.type == 'GAT':
                z, kl = self.encoder(x, edge_index, weight)
            elif args.type == 'SAGE':
                z, kl = self.encoder(x, edge_index)
            x_hat = decoder(z[cell_id, :])
            return x_hat, kl


@torch.no_grad()
def plot_latent(model, pyg_graph, anndata, cell_types, device, name, number_of_cells, celltype_key):
    TRAINING = False
    plt.figure()
    if args.variational:
        if args.type == 'GCN' or args.type == 'GAT':
            z, kl = model.encoder(pyg_graph.expr.to(device), pyg_graph.edge_index.to(device),
                              pyg_graph.weight.to(device))
        elif args.type == 'SAGE':
            z, kl = model.encoder(pyg_graph.expr.to(device), pyg_graph.edge_index.to(device))
        else:
            z, kl = model.encoder(pyg_graph.expr.to(device))
        z = z.to('cpu').detach().numpy()

    else:
        if args.type == 'GCN'or args.type == 'GAT':
            z = model.encoder(pyg_graph.expr.to(device), pyg_graph.edge_index.to(device),
                                      pyg_graph.weight.to(device))
        elif args.type == 'SAGE':
            z = model.encoder(pyg_graph.expr.to(device), pyg_graph.edge_index.to(device))
        else:
            z = model.encoder(pyg_graph.expr.to(device))
        z = z.to('cpu').detach().numpy()
    tsne = manifold.TSNE(n_components=2)
    tsne_z =tsne.fit_transform(z[:number_of_cells,:])
    plot = sns.scatterplot(x=tsne_z[:,0], y=tsne_z[:,1], hue=list(anndata[:number_of_cells,:].obs[celltype_key]))
    plot.legend(fontsize=3)
    fig = plot.get_figure()
    fig.savefig(f'tsne_latentspace_{name}.png', dpi=200)
    plt.close()


    mapper = umap.UMAP()
    umap_z = mapper.fit_transform(z[:number_of_cells,:])
    plot = sns.scatterplot(x=umap_z[:,0], y=umap_z[:,1], hue=list(anndata[:number_of_cells,:].obs[celltype_key]))
    plot.legend(fontsize=3)
    fig = plot.get_figure()
    fig.savefig(f'umap_latentspace_{name}.png', dpi=200)


def train_model(model, train_data, x, cell_id, weight):
    model.train()
    optimizer.zero_grad()
    if args.variational:
        x_hat, kl = model(train_data.expr, train_data.edge_index, cell_id, weight)
    else:
        x_hat = model(train_data.expr, train_data.edge_index, cell_id, weight)

    loss = (1/train_data.expr.size(dim=1)) * ((x - x_hat)**2).sum()
    if args.variational:
        loss += (1 / train_data.num_nodes) * kl
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def apply_on_dataset(model, dataset, name, celltype_key):
    dataset = construct_graph(dataset)
    G = convert_to_graph(dataset.obsp['spatial_distances'], dataset.X, dataset.obs[celltype_key], name)
    pyG_graph = pyg.utils.from_networkx(G)
    pyG_graph.to(device)

    true_expr = dataset.X
    pred_expr = np.zeros(shape=(dataset.X.shape[0], dataset.X.shape[1]))
    print(true_expr.shape, pred_expr.shape)

    total_loss = 0
    for cell in tqdm(G.nodes()):
        batch = pyG_graph.clone()
        batch.expr[cell, :].fill_(0.0)
        assert batch.expr[cell, :].sum() == 0
        loss, x_hat = validate(model, batch, pyG_graph.expr[cell], cell, pyG_graph.weight)
        pred_expr[cell, :] = x_hat.cpu().detach().numpy()
        total_loss += loss

    dataset.obs['total_counts'] = np.sum(dataset.X, axis=1)

    sc.pl.spatial(dataset, use_raw=False, spot_size=0.1, color=['total_counts'],
                  title="Spatial distribution of true expression",
                  save=f"true_expr_spatial_{name}_all_genes", size=1, show=False)
    dataset.X = pred_expr
    dataset.obs['total_pred'] = np.sum(dataset.X, axis=1)
    sc.pl.spatial(dataset, use_raw=False, spot_size=0.1, color=['total_pred'],
                  title='Spatial distribution of predicted expression',
                  save=f"predicted_expr_spatial_{name}_all_genes", size=1, show=False)

    dataset.layers['error'] = np.absolute(true_expr - pred_expr)
    dataset.obs['total_error'] = np.sum(dataset.layers['error'], axis=1)
    dataset.obs['relative_error'] = dataset.obs['total_error'] / dataset.obs['total_counts']
    sc.pl.spatial(dataset, layer='error', spot_size=0.1, title='Spatial distribution of total prediction error',
                  save=f"total_error_spatial_{name}", color=['total_error'], size=1, show=False)
    sc.pl.spatial(dataset, layer='error', spot_size=0.1, title='Spatial distribution of relative prediction error',
                  save=f"relative_error_spatial_{name}", color=['relative_error'], size=1, show=False)

    i = 0
    for gene in dataset.var_names:
        sc.pl.spatial(dataset, use_raw=False, color=[gene], spot_size=0.1,
                      title=f'Spatial distribution of predicted expression of {gene}',
                      save=f"predicted_expr_spatial_{name}_{gene}", size=1, show=False)
        sc.pl.spatial(dataset, layer='error', color=[gene], spot_size=0.1,
                      title=f'Spatial distribution of prediction error of {gene}',
                      save=f"error_spatial_{name}_{gene}", size=1, show=False)
        i += 1
        if i == 1:
            break

    print(dataset.var_names)
    #Calculate total error for each gene
    total_error_per_gene = np.sum(dataset.layers['error'], axis=0)
    print(total_error_per_gene)
    #Get error relative to the amount of genes present
    average_error_per_gene = total_error_per_gene/dataset.shape[1]
    print(average_error_per_gene)
    #Get error relative to amount of expression for that gene over all cells
    relative_error_per_gene = total_error_per_gene / np.sum(dataset.X, axis=0)
    print(relative_error_per_gene)

    error_per_gene = {}
    for i, gene in enumerate(dataset.var_names):
        error_per_gene[gene] = [total_error_per_gene[i],
                                average_error_per_gene[i],
                                relative_error_per_gene[i]]

    with open(f"error_per_gene_{name}.pkl", 'wb') as f:
        pickle.dump(error_per_gene, f)

    error_gene_df = pd.from_dict(error_per_gene.reset_index(), orient='index',
                                 columns=['total', 'average', 'relative']).sort_values(by='relative', axis=0, ascending=False)
    print(error_gene_df)
    top10 = error_gene_df[:10, :]
    print(top10)

    sns.barplot(top10, x='index', y='relative', orient='h')
    plt.xlabel('Relative prediction error')
    plt.ylabel('Gene')
    plt.legend()
    plt.savefig(f'figures/gene_error_{name}.png', dpi=300)
    plt.close()

    error_per_cell_type = {}
    for cell_type in dataset.obs[celltype_key].unique():
        total_error = np.sum(dataset[dataset.obs[celltype_key] == cell_type].obs['total_error'])
        average_error = total_error / dataset[dataset.obs[celltype_key] == cell_type].shape[0]
        error_per_cell_type[cell_type] = average_error
        print(f"{cell_type} : {average_error}")

    error_celltype_df = pd.from_dict(error_per_cell_type, orient='index',
                                     column='average error').sort_values(by='average_error', axis=0, ascending=False)
    sns.barplot(error_celltype_df.reset_index(), x='index', y='relative'
                label='Prediction error', orient='v')
    plt.legend()
    plt.xlabel('Prediction error')
    plt.ylabel('Cell type')
    plt.savefig(f"figures/cell_type_error_{name}.png", dpi=300)
    plt.close()
    with open(f"error_per_celltype_{name}.pkl", 'wb') as f:
        pickle.dump(error_per_cell_type, f)











@torch.no_grad()
def validate(model, val_data, x, cell_id, weight):
    model.eval()
    if args.variational:
        x_hat, kl = model(val_data.expr, val_data.edge_index, cell_id, weight)
    else:
        x_hat = model(val_data.expr, val_data.edge_index, cell_id, weight)

    loss = (1/val_data.expr.size(dim=1)) * ((x - x_hat)**2).sum()
    if args.variational:
        loss += (1 / val_data.num_nodes) * kl
    return float(loss), x_hat

def convert_to_graph(adj_mat, expr_mat, cell_types=None, name='graph'):
    if args.normalization == 'Normal':
        adj_mat = normalize_adjacency_matrix(adj_mat.toarray())
        G = nx.from_numpy_matrix(adj_mat)
    else:
        #Make graph from adjanceny matrix
        G = nx.from_numpy_matrix(adj_mat.toarray())

    nx.set_node_attributes(G, {i: {"expr" : x, 'cell_type' : y} for i, x in enumerate(expr_mat) for i, y in enumerate(cell_types)})

    if args.remove_same_type_edges:
        G = remove_same_cell_type_edges(G)

    if args.remove_subtype_edges:
        G = remove_similar_celltype_edges(G)

    if args.graph_summary:
        graph_summary(G, name)

    if args.add_cell_types == False:
        G = remove_node_attributes(G, 'cell_type')

    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1/G[edge[0]][edge[1]]['weight']


    #Check graph
    print(G)
    print(G.nodes[2])
    for e in G.edges(0):
        print(G[e[0]][e[1]])
    return G

def remove_same_cell_type_edges(G):
    for node in G.nodes():
        cell_type = G.nodes[node]['cell_type']
        neighbors = list(G.neighbors(node))
        for neighbor in neighbors:
            if G.nodes[neighbor]['cell_type'] == cell_type:
                G.remove_edge(neighbor, node)
    return G


def remove_node_attributes(G, attr):
    for node in G.nodes():
        del G.nodes[node][attr]
    return G

def remove_similar_celltype_edges(G):
    for node in G.nodes():
        cell_type = G.nodes[node]['cell_type']
        neighbors = list(G.neighbors(node))
        for neighbor in neighbors:
            overlap_size = 0
            for i, char in enumerate(G.nodes[neighbor]['cell_type']):
                if len(cell_type) > i:
                    if cell_type[i] == char:
                        overlap_size += 1
            if overlap_size > 3:
                G.remove_edge(neighbor, node)

    return G

def normalize_adjacency_matrix(M):
    d = np.sum(M, axis=1)
    d = 1/np.sqrt(d)
    D = np.diag(d)
    return D @ M @ D

def plot_loss_curve(data, xlabel, name):
    plt.plot(list(data.keys()), list(data.values()))
    plt.xlabel(xlabel)
    plt.ylabel('MSE')
    plt.title("GNN-based model MSE")
    plt.savefig(name, dpi=300)
    plt.close()

def plot_val_curve(train_loss, val_loss, name):
    plt.plot(list(train_loss.keys()), list(train_loss.values()), label='training')
    plt.plot(list(val_loss.keys()), list(val_loss.values()), label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Validation curve')
    plt.legend()
    plt.savefig(name, dpi=300)
    plt.close()

def construct_graph(dataset):
    if args.threshold != float(-1):
        threshold = args.threshold
        sq.gr.spatial_neighbors(dataset, coord_type='generic', spatial_key='spatial',
                                radius=float(threshold), n_neighs=100)
    else:
        n_neighs = args.neighbors
        sq.gr.spatial_neighbors(dataset, coord_type='generic', spatial_key='spatial',
                                n_neighs=int(n_neighs), delaunay=False)
    return dataset

def plot_degree(degree_dist, type='degree', graph_name=''):
    #Plot log-log scaled degree distribution
    plt.ylabel('Node frequency')
    plt.hist(degree_dist, bins=np.arange(degree_dist[0].min(), degree_dist[0].max()+1))
    plt.title('Distribution of node {}'.format(type))
    plt.savefig('degree_dist_'+graph_name+'.png', dpi=300)
    plt.close()

def plot_degree_connectivity(conn_dict, graph_name=''):
    plt.ylabel('Average connectivity')
    plt.hist(conn_dict, bins=np.arange(np.array(list(conn_dict.keys())).min(), np.array(list(conn_dict.keys())).max()+1))
    plt.title('Average degree connectivity')
    plt.savefig('degree_con_'+graph_name+'.png', dpi=300)
    plt.close()

def plot_edge_weights(edge_dict, name):
    plot = sns.histplot(edge_dict)
    plot.set(xlabel='Weight (distance)', ylabel='Edge frequency')
    plt.title('Edge weight frequencies')
    plt.savefig('dist_weight_'+name+".png", dpi=300)
    plt.close()

def graph_summary(G, name):
    summary_dict = {}
    summary_dict['name'] = name
    summary_dict['params'] = dict(vars(args))
    edges = G.number_of_edges()
    summary_dict['edges'] = edges
    #get number of nodes
    nodes = G.number_of_nodes()
    summary_dict['nodes'] = nodes
    #Get density
    density = nx.density(G)
    summary_dict['density'] = density
    #Get average clustering coefficient
    clust_cf = nx.average_clustering(G)
    summary_dict['clustcf'] = clust_cf
    #Plot the edge weight distribution
    edge_dist = {}
    for u,v,w in G.edges(data=True):
        w = int(w['weight'])
        if w not in edge_dist:
            edge_dist[w] = 0
        edge_dist[w] += 1
    summary_dict['edge_dist'] = edge_dist
    plot_edge_weights(edge_dist, name)
    #Compute the average degree connectivity
    average_degree_connectivity = nx.average_degree_connectivity(G)
    summary_dict['average_degree_connectivity'] = average_degree_connectivity
    plot_degree_connectivity(average_degree_connectivity, name)
    #Compute the degree assortativity and cell type assortativity
    degree_assortativity = nx.degree_assortativity_coefficient(G)
    celltype_assortativity = nx.attribute_assortativity_coefficient(G, 'cell_type')
    summary_dict['degree_assortativity'] = degree_assortativity
    summary_dict['celltype_assortativity'] = celltype_assortativity
    #Get indegree/outdegree counts
    degrees = sorted((d for n,d in G.degree()), reverse=True)
    #Make distribution in form degree:count
    degree_dist = np.unique(degrees, return_counts=True)
    summary_dict['degree_dist'] = degree_dist
    #Plot the degree distribution
    plot_degree(degree_dist, 'degree', name)

    with open(f'graph_summary_{name}.pkl', 'wb') as f:
        pickle.dump(summary_dict, f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Get current directory, make sure output directory exists
dirpath = os.getcwd()
outpath = dirpath + "/output"
if not os.path.exists(outpath):
    os.mkdir("output")

if not os.path.exists(dirpath+"/data"):
    os.mkdir("data")

if args.dataset == 'resolve':
    if not os.path.exists(dirpath+"/data/resolve.h5ad"):
        print("Downloading RESOLVE dataset:")
        link = requests.get("https://dl01.irc.ugent.be/spatial/adata_objects/adataA1-1.h5ad")
        with open('data/resolve.h5ad', 'wb') as f:
            f.write(link.content)
    dataset = sc.read_h5ad("data/resolve.h5ad")
    name = 'resolve'
    organism = 'mouse'
    celltype_key = 'maxScores'

elif args.dataset == 'merfish':
    dataset = sq.datasets.merfish()
    organism='mouse'
    name='mouse_merfish'

elif args.dataset == 'seqfish':
    dataset = sq.datasets.seqfish()
    organism='mouse'
    name='mouse_seqfish'
    celltype_key = 'celltype_mapped_refined'

elif args.dataset == 'nanostring':
    dataset = sq.read.nanostring(path="data/Lung5_Rep1",
                       counts_file="Lung5_Rep1_exprMat_file.csv",
                       meta_file="Lung5_Rep1_metadata_file.csv",
                       fov_file="Lung5_Rep1_fov_positions_file.csv")
    organism = 'human'
    name= 'Lung5_Rep1'

print("Dataset:")
print(dataset)

if not isinstance(dataset.X, np.ndarray):
    dataset.X = dataset.X.toarray()

val_i = random.sample(range(len(dataset.obs)), k=1000)
val, train = dataset[val_i], dataset[[x for x in range(len(dataset.obs)) if x not in val_i]]
print(val.X.shape, train.X.shape)

if args.threshold != -1 or args.neighbors != -1 or args.dataset != 'resolve':
    print("Constructing graph...")
    train = construct_graph(train)
    val = construct_graph(val)

print("Converting graph to PyG format...")

if args.weight:
    G_train = convert_to_graph(train.obsp['spatial_distances'], train.X, train.obs[celltype_key], name+'_train')
    G_val = convert_to_graph(val.obsp['spatial_distances'], val.X, val.obs[celltype_key], name+"_val")
else:
    G_train = convert_to_graph(train.obsp['spatial_connectivities'], train.X, train.obs[celltype_key], name+"_train")
    G_val = convert_to_graph(val.obsp['spatial_connectivities'], val.X, val.obs[celltype_key], name+"_val")

pyg_train, pyg_val = pyg.utils.from_networkx(G_train), pyg.utils.from_networkx(G_val)
#TODO: Split into train and validation
print(pyg_train, pyg_val)
pyg_train.to(device)
pyg_val.to(device)

#Set layer sizes
input_size, hidden_layers, latent_size = pyg_train.expr.size(dim=1), [int(x) for x in args.hidden.split(',')], args.latent

#Build model architecture based on given arguments
if not args.variational and args.type == 'GCN':
    encoder = GCNEncoder(input_size, hidden_layers, latent_size)
elif not args.variational and args.type == 'GAT':
    encoder = GATEncoder(input_size, hidden_layers, latent_size)
elif not args.variational and args.type == 'SAGE':
    encoder = SAGEEncoder(input_size, hidden_layers, latent_size, args.aggregation_method)
elif not args.variational and args.type == 'Linear':
    encoder = LinearEncoder(input_size, hidden_layers, latent_size)
elif args.variational and args.type == 'GCN':
    encoder = VGCNEncoder(input_size, hidden_layers, latent_size)
elif args.variational and args.type == 'GAT':
    encoder = VGATEncoder(input_size, hidden_layers, latent_size)
elif args.variational and args.type == 'SAGE':
    encoder = VSAGEEncoder(input_size, hidden_layers, latent_size, args.aggregation_method)
elif args.variational and args.type == 'Linear':
    encoder = VLinearEncoder(input_size, hidden_layers, latent_size)

#Build Decoder
decoder = Decoder(input_size, hidden_layers, latent_size)
#Build model
model = GAE(encoder, decoder)
print("Model:")
print(model)

#Send model to GPU
model = model.to(device)

pyg.transforms.ToDevice(device)

#Set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Set number of nodes to sample per epoch
if args.cells == -1:
    k = G_train.number_of_nodes()
else:
    k = args.cells

loss_over_cells = {}
train_loss_over_epochs = {}
val_loss_over_epochs = {}
#Set normalization for training data expression
#normalizer = NormalizeFeatures(["expr"])
#Train the model
for epoch in range(1, args.epochs+1):
    total_loss_over_cells = 0
    for i, cell in tqdm(enumerate(random.sample(G_train.nodes(), k=k))):
        batch = pyg_train.clone()
        batch.expr[cell, :].fill_(0.0)
        assert batch.expr[cell, :].sum() == 0
        loss = train_model(model, batch, pyg_train.expr[cell], cell, pyg_train.weight)
        total_loss_over_cells += loss
        if i % 50 == 0 and i != 0:
            print(f"Cells seen: {i}, average MSE:{total_loss_over_cells/i}")
            loss_over_cells[i] = total_loss_over_cells/i

    total_val_loss = 0
    for cell in tqdm(random.sample(G_val.nodes(), k=G_val.number_of_nodes())):
        val_batch = pyg_val.clone()
        val_batch.expr[cell, :].fill_(0.0)
        assert val_batch.expr[cell, :].sum() == 0
        val_loss, _ = validate(model, val_batch, pyg_val.expr[cell], cell, pyg_val.weight)
        total_val_loss += val_loss

    train_loss_over_epochs[epoch] = total_loss_over_cells/k
    val_loss_over_epochs[epoch] = total_val_loss/G_val.number_of_nodes()
    print(f"Epoch {epoch}, average training loss:{train_loss_over_epochs[epoch]}, average validation loss:{val_loss_over_epochs[epoch]}")

#Save trained model
torch.save(model, f"model_{args.type}.pt")
try:
    torch.onnx.export(model, pyg_graph.expr, f'{args.type}_model.onnx', export_params=True,
                  input_names=['neighborhood expression+spatial graph'], output_Names=['cell expression'])
except:
    'ONNX FAILED'

if args.variational:
    subtype = 'variational'
else:
    subtype = 'non-variational'

#Plot results
plot_loss_curve(loss_over_cells, 'cells', f'loss_curve_cells_{name}_{type}_{subtype}.png')
plot_val_curve(train_loss_over_epochs, val_loss_over_epochs, f'val_loss_curve_epochs_{name}_{type}_{subtype}.png')
plot_latent(model, pyg_val, dataset, list(dataset.obs[celltype_key].unique()),
            device, name=f'{name}_{type}_{subtype}', number_of_cells=500, celltype_key=celltype_key)

#Apply on dataset
apply_on_dataset(model, dataset, 'test', celltype_key)
