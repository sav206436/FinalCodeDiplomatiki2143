#1 Importing Libraries (Import Statements): This section imports various Python libraries and packages necessary for data processing, visualization, and machine learning.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch
import string
import re
from matplotlib import pyplot as plt    
from sklearn.metrics import roc_auc_score
import tqdm
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

#2 Loading Data: The code reads movie and ratings data from .csv files using Pandas and stores them in dataframes.        
movies_df = pd.read_csv("C:/Users/Savvas/Desktop/me2143diplomatiki/movielens100kdataset/ml-latest-small/movies.csv",index_col='movieId')#A DataFrame containing movie information, where the "movieId" column is set as the index. "movieId" is the row identifier for each row in the DataFrame
ratings_df = pd.read_csv("C:/Users/Savvas/Desktop/me2143diplomatiki/movielens100kdataset/ml-latest-small/ratings.csv")# A DataFrame containing user ratings data.

#3 Data Exploration: This section explores the loaded data.
fig = plt.figure()
ax = ratings_df.rating.value_counts(True).sort_index().plot.bar(figsize=(8,6))
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
plt.xlabel('Ratings from 0,5 to 5.0', fontsize=12)
plt.ylabel('% share of ratings', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
fig.savefig('Ratings_distribution.png')#X,Y diagram with %share of ratings of all users

#4 Data Preprocessing: Genres of movies are split and converted into indicator variables. These indicator variables indicate whether a movie belongs to a specific genre (e.g., Action, Adventure, Drama). This information is stored in the movie_feat variable, which is a tensor.
genres = movies_df['genres'].str.get_dummies('|')#split the genre information into indicator variables. The | separator is used to split genres since genres are typically separated by pipe '|' symbols in such datasets
print(genres[["Action", "Adventure", "Drama", "Horror"]].head())#prints the first few rows of the 'genres' DataFrame, specifically showing the columns for "Action," "Adventure," "Drama," and "Horror" genres.
movie_feat = torch.from_numpy(genres.values).to(torch.float)# the binary genre indicator variables are converted into a PyTorch tensor. This tensor, 'movie_feat,' represents the genre features of movies. Each row of the tensor corresponds to a movie, and each column represents a different genre. The values are converted to float type to ensure compatibility with PyTorch operations.
assert movie_feat.size() == (9742, 20)#This line checks whether the 'movie_feat' tensor has the expected shape of (9742, 20). This assertion verifies that the tensor has been correctly constructed with 9742 movies and 20 genre columns.
print (movie_feat[1:4])#prints the genre features for movies 2 to 4 in the dataset. It shows the genre indicator values for each of these movies.

#5 User and Movie ID Mapping: This section maps unique user and movie IDs to consecutive values. It creates mappings for user and movie data, which is used for constructing edges connecting users to movies.
# Create a mapping from unique user indices to range [0, num_user_nodes):
unique_user_id = ratings_df['userId'].unique()

unique_user_id = pd.DataFrame(data={
    'userId': unique_user_id,
    'mappedID': pd.RangeIndex(len(unique_user_id)),
})
print("Mapping of userIds to consecutive values:")
print("--------------------------------------------------")
print(unique_user_id.head())
print()
# Create a mapping from unique movie indices to range [0, num_movie_nodes):
unique_movie_id = ratings_df['movieId'].unique()
unique_movie_id = pd.DataFrame(data={
    'movieId': unique_movie_id,
    'mappedID': pd.RangeIndex(len(unique_movie_id)),
})
print("Mapping of movieIds to consecutive values:")
print("--------------------------------------------------")
print(unique_movie_id.head())

#6 Creating User-Movie Edges: Edges connecting users to movies are created based on the mapped user and movie IDs. The resulting edge indices are stored in edge_index_user_to_movie
# Perform merge to obtain the edges from users and movies:
ratings_user_id = pd.merge(ratings_df['userId'], unique_user_id,
                            left_on='userId', right_on='userId', how='left')
print("Merges create user-movie edges as ratings_user_id:")
print("--------------------------------------------------")
print(ratings_user_id.head())
ratings_user_id = torch.from_numpy(ratings_user_id['mappedID'].values)

ratings_movie_id = pd.merge(ratings_df['movieId'], unique_movie_id,
                            left_on='movieId', right_on='movieId', how='left')
ratings_movie_id = torch.from_numpy(ratings_movie_id['mappedID'].values)

# With this, we are ready to construct our `edge_index` in COO format
# following PyG semantics:
edge_index_user_to_movie = torch.stack([ratings_user_id, ratings_movie_id], dim=0)
assert edge_index_user_to_movie.size() == (2, 100836)
print()
print("Final edge_index pointing from users to movies:")
print("--------------------------------------------------")
print(edge_index_user_to_movie)

#7 Creating a Heterogeneous Graph: A heterogeneous graph is constructed to represent the data. It includes user and movie nodes, their features, and user-movie interactions. This part also ensures bidirectional edges are present for message passing.
data = HeteroData()
# Save node indices:
data["user"].node_id = torch.arange(len(unique_user_id))
data["movie"].node_id = torch.arange(len(movies_df))
# Add the node features and edge indices:
data["movie"].x = movie_feat
data["user", "rates", "movie"].edge_index = edge_index_user_to_movie
# We also need to make sure to add the reverse edges from movies to users
# in order to let a GNN be able to pass messages in both directions.
# We can leverage the `T.ToUndirected()` transform for this from PyG:
data = T.ToUndirected()(data)

#8 Data Splitting: The dataset is split into training, validation, and testing sets. Some edges are reserved for validation and testing purposes using a random link split strategy.
# For this, we first split the set of edges into
# training (80%), validation (10%), and testing edges (10%).
# Across the training edges, we use 70% of edges for message passing,
# and 30% of edges for supervision.
# We further want to generate fixed negative edges for evaluation with a ratio of 2:1.
# Negative edges during training will be generated on-the-fly.
# We can leverage the `RandomLinkSplit()` transform for this from PyG:
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=("user", "rates", "movie"),
    rev_edge_types=("movie", "rev_rates", "user"),
)
train_data, val_data, test_data = transform(data)

train_data["user", "rates", "movie"].edge_label

# In the first hop, we sample at most 20 neighbors.
# In the second hop, we sample at most 10 neighbors.
# In addition, during training, we want to sample negative edges on-the-fly with
# a ratio of 2:1.
# We can make use of the `loader.LinkNeighborLoader` from PyG:

#9 Creating a Data Loader: A data loader is created to efficiently sample data for training. It uses the LinkNeighborLoader from PyTorch Geometric for sampling and loading data.
# Define seed edges:
edge_label_index = train_data["user", "rates", "movie"].edge_label_index
edge_label = train_data["user", "rates", "movie"].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(("user", "rates", "movie"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)

#10 Defining the GNN Model: This part defines a Graph Neural Network (GNN) model. It consists of two GraphSAGE (SAGEConv) layers for message passing.
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

#11 Defining a Classifier: A classifier is defined to make edge-level predictions by computing dot-products between source and destination node embeddings.
# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)

#12 Creating the Main Model: The main recommendation model is created. It combines user and movie embeddings, applies the GNN, and makes predictions using the classifier.
class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.movie_lin = torch.nn.Linear(20, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.movie_emb = torch.nn.Embedding(data["movie"].num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()
    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
        }
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )
        return pred
       
model = Model(hidden_channels=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the validation seed edges:
edge_label_index = val_data["user", "rates", "movie"].edge_label_index
edge_label = val_data["user", "rates", "movie"].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10],
    edge_label_index=(("user", "rates", "movie"), edge_label_index),
    edge_label=edge_label,
    batch_size=3 * 128,
    shuffle=False,
)

# Initialize lists to store AUC values, loss values and epochs
loss_values = []
auc_values = []
epochs = []

#13 Training the Model: The recommendation model is trained using the training data. It calculates the loss and updates model parameters using backpropagation. AUC (Area Under the Curve) values are collected for each epoch to evaluate the model's performance.
#And collect the AUC values for each epoch
for epoch in range(1, 5):
    total_loss = total_examples = 0
    print(f"Training")
    for sampled_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        sampled_data.to(device)
        pred = model(sampled_data)
        ground_truth = sampled_data["user", "rates", "movie"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
    # Compute AUC for the current epoch and store it
    preds = []
    ground_truths = []
    print(f"AUC calculation")
    for sampled_data in tqdm.tqdm(val_loader):
        with torch.no_grad():
            sampled_data.to(device)
            preds.append(model(sampled_data))
            ground_truths.append(sampled_data["user", "rates", "movie"].edge_label)
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    print(f"AUC: {auc:.4f}")
    # Store the AUC value, loss and epoch
    loss_values.append(total_loss / total_examples)
    auc_values.append(auc)
    epochs.append(epoch)

#14 Plotting AUC and Loss: At the end of training, the code creates two plots: one for AUC values over each epoch and another for loss values over each epoch. These plots provide insights into the model's training progress.
# Create an x, y graph to show AUC values over each epoch
plt.figure(figsize=(10, 6))
plt.plot(epochs, auc_values, marker='o', linestyle='-', color='b')
plt.title('Area Under the ROC Curve (AUC) value over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Area Under the ROC Curve (AUC) value')
plt.grid(True)

# Create an x-y graph to show loss values over each epoch
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_values, marker='o', linestyle='-', color='r')
plt.title('Binary Cross Entropy Loss with Logits (Loss) value over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross Entropy Loss with Logits (Loss) value')
plt.grid(True)

#Show the plots
plt.show()