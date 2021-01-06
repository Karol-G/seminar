import hdbscan
import numpy as np
from sklearn.datasets import make_blobs
import seaborn as sns
import matplotlib.pyplot as plt

def filter_linkage(linkage, labels, label):
    length = len(labels)
    indices = np.nonzero([labels == label])[1].tolist()
    indices_length = len(indices)
    filtered_linkage = []
    new_index_counter = 0
    new_cluster_counter = indices_length
    original_indices = []
    for i, link in enumerate(linkage):
        if link[0] in indices and link[1] in indices:
            new_link = []
            for j in [0, 1]:
                if link[j] < length:
                    new_link.append(new_index_counter)
                    original_indices.append([int(link[j]), new_index_counter])
                    new_index_counter += 1
                else:
                    new_link.append(new_cluster_counter)
                    original_indices.append([int(link[j]), new_cluster_counter])
                    new_cluster_counter += 1
            new_link.append(link[2])
            new_link.append(link[3])
            filtered_linkage.append(new_link)
            indices.append(i+length)
    filtered_linkage = np.asarray(filtered_linkage)
    original_indices = np.asarray(original_indices)
    return filtered_linkage, original_indices, indices_length

def filter_data(data, original_indices, indices_length):
    filtered_data = [None]*indices_length
    for index in original_indices:
        if index[1] < indices_length:
            filtered_data[index[1]] = data[index[0]]
    filtered_data = np.asarray(filtered_data)
    return filtered_data


def plot_data(plot_data, plot_labels):
    labels = np.unique(plot_labels)
    for label in labels:
        label_data = plot_data[plot_labels == label]
        plt.scatter(*label_data.T)
    plt.show()


# Generate blob data
data, _ = make_blobs(1000)

# Cluster all data
clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
cluster_labels = clusterer.fit_predict(data)

plot_data(data, cluster_labels)

# Filter linkage by removing all links that don't belong to cluster_label
cluster_label = 1
#new_linkage, original_indices, indices_length = filter_linkage(clusterer._single_linkage_tree, cluster_labels, cluster_label)

# Convert linkage to flat clusters and return the new cluster labels
#new_labels = hdbscan.hdbscan_._tree_to_labels(0, single_linkage_tree=new_linkage, min_cluster_size=10)[0]
#print(new_labels)

# Filter the original data so that only the data of cluster_label remains
#new_data = filter_data(data, original_indices, indices_length)
label = 1
label_data = data[cluster_labels == label]
new_labels = clusterer.fit_predict(label_data)

plot_data(label_data, new_labels)