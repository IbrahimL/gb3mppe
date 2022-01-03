import argparse
import tensorflow as tf
import numpy as np
from lib.models.MMG import MMG
from lib.dataset.campus import *
import matplotlib.pyplot as plt

def train(model, data, gt_data, EPOCHS = 10):
    adjacency, node_features, edge_features= data
    loss_t = []
    print("----TRAINING---")
    for epoch in range(EPOCHS):
        r_seed = np.random.randint(0, 10000)
        adjacency = tf.random.shuffle(adjacency, seed=r_seed)
        node_features = tf.random.shuffle(node_features, seed=r_seed)
        edge_features = tf.random.shuffle(edge_features, seed=r_seed)
        gt_data = tf.random.shuffle(gt_data, seed=r_seed)
        for k in range(0,390,10):
            a_sparse= tf.sparse.from_dense(adjacency[k:k+10])
            n_features=node_features[k:k+10]
            e_features=edge_features[k:k+10]
            g_graphs=gt_data[k:k+10]
            loss = model.train_step([a_sparse, n_features, e_features], g_graphs)
        print("epoch {}, loss={}".format(epoch, loss))
        loss_t.append(loss)
    return loss_t

def main():
    parser = argparse.ArgumentParser(description='Train the mmg model.')
    parser.add_argument('n_epochs', metavar='n_epochs', type=int, default=10,
                    help='number of training epochs')
    parser.add_argument('show', metavar='show', type=bool, default=False,
                    help='show the training curve')
    args = parser.parse_args()
    show = args.show
    n_epochs = args.n_epochs
    # Architecture parameters
    # MMG
    mmg_hidden_dim = [256, 128, 64]
    mmg_output_dim = 1
    campus_dataset = Campus('../../data/Campus/node_features.pkl', '../../data/Campus/edge_features.pkl', '../../data/Campus/GT_graphs.pkl')
    node_features, adjacency, edge_features, gt_graphs = campus_dataset._generate_graphs()
    # we take only the first 400 images for simplicity of implementation and to test the training
    adjacency= adjacency[0:400]
    node_features=node_features[0:400]
    edge_features=edge_features[0:400]
    gt_graphs=gt_graphs[0:400]
    # cast all the tensors to the same data type
    adjacency=tf.cast(adjacency,tf.float32)
    node_features=tf.cast(node_features,tf.float32)
    edge_features=tf.cast(edge_features,tf.float32)
    gt_graphs = tf.cast(gt_graphs,tf.float32)

    print("adjacency shape", adjacency.shape)
    print("node features shape", node_features.shape)
    print("edge features shape", edge_features.shape)
    
    # MMG Init
    mmg_model = MMG(mmg_hidden_dim, mmg_output_dim)
    
    loss = train(model=mmg_model,
                 data=[adjacency, node_features, edge_features],
                 gt_data=gt_graphs,
                 EPOCHS=n_epochs)
    if show:
        fig, ax = plt.subplots()
        ax.plot(loss)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Training curve")
        plt.show()
    
if __name__ == "__main__":
    loss = main()