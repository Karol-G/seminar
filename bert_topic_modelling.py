from bertopic._bertopic import BERTopic, check_documents_type, check_embeddings_shape
from scipy.sparse.csr import csr_matrix
import pandas as pd
import json
from tqdm import tqdm
import pickle
import numpy as np
import time


def fit_transform(model, data_save_path, documents, embeddings=None, show_progress_bar=False):
    check_documents_type(documents)
    check_embeddings_shape(embeddings, documents)

    documents = pd.DataFrame({"Document": documents,
                              "ID": range(len(documents)),
                              "Topic": None})

    # Extract embeddings
    print("Extract BERT sentence embeddings...")
    if not any([isinstance(embeddings, np.ndarray), isinstance(embeddings, csr_matrix)]):
        start_time = time.time()
        embeddings = model._extract_embeddings(documents.Document, show_progress_bar)
        print("Elapsed time: ", time.time() - start_time)
        with open(data_save_path + "embeddings.pkl", 'wb') as handle:
         pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        model.custom_embeddings = True

    # Reduce dimensionality with UMAP
    print("Reduce dimensionality with UMAP")
    start_time = time.time()
    umap_embeddings = model._reduce_dimensionality(embeddings)
    print("Elapsed time: ", time.time() - start_time)
    with open(data_save_path + "umap_embeddings.pkl", 'wb') as handle:
        pickle.dump(umap_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Cluster UMAP embeddings with HDBSCAN
    print("Cluster UMAP embeddings with HDBSCAN")
    start_time = time.time()
    documents, probabilities, linkage_tree = model._cluster_embeddings(umap_embeddings, documents)
    print("Elapsed time: ", time.time() - start_time)
    with open(data_save_path + "doc_prob.pkl", 'wb') as handle:
        pickle.dump([documents, probabilities, linkage_tree], handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Extract topics by calculating c-TF-IDF
    print("Extract topics by calculating c-TF-IDF")
    start_time = time.time()
    model._extract_topics(documents)
    print("Elapsed time: ", time.time() - start_time)

    print("Other stuff")
    start_time = time.time()
    if model.nr_topics:
        documents = model._reduce_topics(documents)
        probabilities = model._map_probabilities(probabilities)

    predictions = documents.Topic.to_list()
    print("Elapsed time: ", time.time() - start_time)

    with open(data_save_path + "pred_prob.pkl", 'wb') as handle:
        pickle.dump([predictions, probabilities], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return predictions, probabilities


if __name__ == '__main__':
    keyword = "title"
    data_path = "/gris/gris-f/homelv/kgotkows/datasets/arxiv/arxiv-metadata-oai-snapshot.json"
    data_save_path = "/gris/gris-f/homelv/kgotkows/datasets/arxiv/" + keyword + "4/"
    # data_save_path = "D:/Datasets/visuelle_trendanalyse/arxiv/abstract2/"
    preprocess_data = False
    load_model = True
    print("data_save_path: ", data_save_path)


    if not load_model:
        if preprocess_data:
            data = []
            with open(data_path, 'r') as f:
                for line in tqdm(f):
                    data.append(json.loads(line))

            data = [entry[keyword] for entry in tqdm(data)]

            with open(data_save_path + "arxiv_{}.pkl".format(keyword), 'wb') as handle:
             pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(data_save_path + "arxiv_{}.pkl".format(keyword), 'rb') as handle:
              data = pickle.load(handle)

            data = data[:int(len(data)/4)]
            print(len(data))

            model = BERTopic(embedding_model="/gris/gris-f/homelv/kgotkows/datasets/arxiv/arxiv_trained_embedding/model/", huggingface_model=True, verbose=True, calculate_probabilities=False)
            # topics, probabilities = model.fit_transform(data, show_progress_bar=True)
            topics, probabilities = fit_transform(model, data_save_path, data, show_progress_bar=True)
            model.save(data_save_path + "arxiv_{}.model".format(keyword))
            print(model.get_topic(0))
    else:
        model = BERTopic(verbose=True)
        model = model.load(data_save_path + "arxiv_{}.model".format(keyword))

        topics = model.get_topic_freq()

        for index, row in topics.iterrows():
            topic_index = row["Topic"]
            if topic_index != -1:
                print(model.get_topic(topic_index))
            if index >= 20:
                break

        #model.visualize_topics()

        # with open(data_save_path + "doc_prob.pkl", 'rb') as handle:
        #     documents, probabilities, linkage_tree = pickle.load(handle)
        #
        # with open(data_save_path + "arxiv_{}.pkl".format(keyword), 'rb') as handle:
        #     data = pickle.load(handle)
        # data = data[:int(len(data)/4)]
        #
        # print("data len: ", len(data))
        #
        # # print(linkage_tree)
        # print(linkage_tree)
        # print(linkage_tree.shape)
        #
        # clusters = 0
        #
        # for i in tqdm(range(len(linkage_tree))):
        #     if linkage_tree[i][0] >= len(data) or linkage_tree[i][1] >= len(data):
        #         clusters += 1
        #
        # print("clusters: ", clusters)

        # TODO: From X / umap_embeddings / documents remove all entries that dont belong to class C
        # TODO: From single_linkage_tree do the same (look into get_clusters)
        # TODO: Call hdbscan._tree_to_labels
