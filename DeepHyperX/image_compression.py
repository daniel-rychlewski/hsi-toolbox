import time
from datetime import timedelta

import numpy as np

from DeepHyperX.models import autoencoder_train
from DeepHyperX.utils import print_memory_metrics, start_mem_measurement, stop_mem_measurement, get_device


def apply_band_selection_choice(technique, dataset, predictions, mode, hyperparams, wrapped_dataset, custom_epoch_number, n_components, df_column_entry_dict):
    if technique is not None:
        print("\nUsing band selection technique "+technique+" ...\n")
        if technique == "Autoencoder":
            if custom_epoch_number is not None:
                return apply_autoencoder(hyperparams, wrapped_dataset, int(custom_epoch_number))
            else:
                return apply_autoencoder(hyperparams, wrapped_dataset, 10)
        else:
            return apply_band_selection(technique, dataset, predictions, mode, int(n_components), df_column_entry_dict)
    else:
        return


# overview of some techniques: https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/

def apply_band_selection(technique, dataset, predictions, mode, n_components, df_column_entry_dict):
    if df_column_entry_dict is None:
        df_column_entry_dict = {} # couldn't care less, this is a lazy way to make all accesses work

    print("Dataset current shape: " + str(dataset.shape))

    print_memory_metrics("before applying band selection method "+technique, df_column_entry_dict)

    from DeepHyperX.batch import PARAMETER_JSON
    parameterFile = open(PARAMETER_JSON, "r")
    import json
    data = json.load(parameterFile)
    parameterFile.close()

    if technique in ["IncrementalPCA"]: # requires special method
        dataset, _ = applyIncrementalPCA(dataset, n_components)

    elif technique in data["image_compression"]["extraction"]["techniques"]:

        extraction_object = None
        if technique == "PCA":
            from sklearn.decomposition import PCA
            """ HybridSN: Exploring 3D-2D CNN Feature Hierarchy for Hyperspectral Image Classification
            Source code used: https://github.com/gokriznastic/HybridSN/blob/master/Hybrid-Spectral-Net.ipynb
            Paper: https://arxiv.org/abs/1902.06701
            Good parameters: 30 components for Indian Pines, 15 for Salinas and Pavia University
            """
            extraction_object = PCA(n_components=n_components, whiten=True)
        elif technique == "KernelPCA":
            from sklearn.decomposition import KernelPCA
            extraction_object = KernelPCA(kernel="rbf", n_components=n_components, gamma=None, fit_inverse_transform=True, n_jobs=1)
        elif technique == "SparsePCA":
            """Sparse PCA uses the links between the ACP and the SVD to extract the main components by solving a lower-order matrix approximation problem."""
            from sklearn.decomposition import SparsePCA
            extraction_object = SparsePCA(n_components=n_components, alpha=0.0001, n_jobs=-1)
        elif technique == "LDA": # only supervised is supported, y is required
            if mode != "supervised":
                print("warning: mode other than supervised detected for lda, setting it to supervised...\n")
                mode = "supervised"
            # maximally n_classes - 1 columns, https://stackoverflow.com/questions/26963454/lda-ignoring-n-components
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            extraction_object = LinearDiscriminantAnalysis(n_components=n_components)
        elif technique == "SVD":
            from sklearn.decomposition import TruncatedSVD
            extraction_object = TruncatedSVD(n_components=n_components, algorithm='randomized', n_iter=5)
        elif technique == "GRP":
            from sklearn.random_projection import GaussianRandomProjection
            extraction_object = GaussianRandomProjection(n_components=n_components, eps=0.5)
        elif technique == "SRP":
            from sklearn.random_projection import SparseRandomProjection
            extraction_object = SparseRandomProjection(n_components=n_components, density='auto', eps=0.5, dense_output=False)
        elif technique == "MDS":
            """O(n^3), uses lots of memory for distance matrix (doesn't fit in 48GB), doesn't fit in GPU memory either, so basically unusable"""
            from sklearn.manifold import MDS
            extraction_object = MDS(n_components=n_components, n_init=12, max_iter=200, metric=True, n_jobs=16)
        elif technique == "MiniBatch":
            """takes too long"""
            from sklearn.decomposition import MiniBatchDictionaryLearning
            extraction_object = MiniBatchDictionaryLearning(n_components=n_components, batch_size=200,
                                                                alpha=1, n_iter=1)
        elif technique == "LLE":
            # modified LLE requires n_neighbors >= n_components
            """execution takes 20 minutes or so, but it does work, just takes a long time"""
            from sklearn.manifold import LocallyLinearEmbedding
            extraction_object = LocallyLinearEmbedding(n_components=n_components, n_neighbors=100, method='modified',
                                         n_jobs=4)
        elif technique == "ICA":
            from sklearn.decomposition import FastICA
            extraction_object = FastICA(n_components=n_components, algorithm='parallel', whiten=True, max_iter=100)
        elif technique == "FactorAnalysis":
            from sklearn.decomposition import FactorAnalysis
            extraction_object = FactorAnalysis(n_components=n_components)#75
        elif technique == "ISOMAP":
            from sklearn import manifold
            extraction_object = manifold.Isomap(n_neighbors=5, n_components=n_components, n_jobs=-1)
        elif technique == "t-SNE":
            # like PCA, but non-linear (pca is linear)
            from sklearn.manifold import TSNE
            extraction_object = TSNE(n_components=n_components, learning_rate=300, perplexity=30, early_exaggeration=12,
                        init='random')
        elif technique == "UMAP":
            # install umap-learn for this to work
            import umap
            extraction_object = umap.UMAP(n_neighbors=50, min_dist=0.3, n_components=n_components)
        elif technique == "NMF":
            # https://www.kaggle.com/remidi/dimensionality-reduction-techniques
            from sklearn.decomposition import NMF
            extraction_object = NMF(n_components=n_components, init='nndsvdar', random_state=420)
        elif technique == "FAG":
            # super fast and nice
            from sklearn.cluster import FeatureAgglomeration
            extraction_object = FeatureAgglomeration(n_clusters=n_components, linkage='ward')
        else:
            raise ValueError("Unknown feature extraction technique: "+technique)

        start_mem_measurement()
        start = time.time()

        dataset, _ = applyFeatureExtraction(dataset, predictions, extraction_object, mode, merged=(len(dataset.shape) == 4 and len(predictions.shape) == 3))

        time_elapse = time.time() - start

        event = 'applying band selection method (EXTRACTION) '+technique
        formatted_time = str(timedelta(seconds=time_elapse))
        df_column_entry_dict['Time measurement at ' + event + ' [s]'] = time_elapse

        print("\n"+event+" took " + formatted_time + " seconds\n")

        event = "after applying band selection method " + technique
        stop_mem_measurement(event, df_column_entry_dict)
        print_memory_metrics(event, df_column_entry_dict)

    elif technique in data["image_compression"]["selection"]["techniques"]:

        selection_object = None
        if technique == "RandomForest":
            # Random forests or random decision forests are an ensemble learning method for classification, regression and other
            # tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.[1][2] Random decision forests correct for decision trees' habit of overfitting to their training set.[3]:587–588 https://en.wikipedia.org/wiki/Random_forest
            from sklearn.ensemble import RandomForestClassifier
            selection_object = RandomForestClassifier()
        elif technique == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression
            selection_object = LogisticRegression()
        elif technique == "LinearRegression":
            from sklearn.linear_model import LinearRegression
            selection_object = LinearRegression()
        elif technique == "LightGBM":
            from lightgbm import LGBMClassifier
            selection_object = LGBMClassifier()
        else:
            raise ValueError("Unknown feature selection technique: " + technique)

        start_mem_measurement()
        start = time.time()

        dataset, _ = applyFeatureSelection(dataset, predictions, selection_object, n_components, mode, merged=(len(dataset.shape) == 4 and len(predictions.shape) == 3))

        time_elapse = time.time() - start

        event = 'applying band selection method (SELECTION) '+technique
        formatted_time = str(timedelta(seconds=time_elapse))
        df_column_entry_dict['Time measurement at ' + event + ' [s]'] = time_elapse

        print("\n"+event+" took " + formatted_time + " seconds\n")

        event = "after applying band selection method " + technique
        stop_mem_measurement(event, df_column_entry_dict)
        print_memory_metrics(event, df_column_entry_dict)

    print("Dataset new shape: " + str(dataset.shape))

    return dataset

# https://towardsdatascience.com/dimensionality-reduction-toolbox-in-python-9a18995927cd
def applyIncrementalPCA(X, num_components, n_batches = 256):
    """
    :param X:
    :param num_components:
    :return:
    """
    from sklearn.decomposition import IncrementalPCA
    inc_pca = IncrementalPCA(num_components)

    newX = np.reshape(X, (-1, X.shape[2]))

    for X_batch in np.array_split(newX, n_batches):
        inc_pca.partial_fit(X_batch)
    X_ipca = inc_pca.transform(newX)

    X_ipca = np.reshape(X_ipca, (X.shape[0], X.shape[1], -1))

    return X_ipca, inc_pca

def apply_autoencoder(hyperparams, wrapped_dataset, custom_epoch_number=None):
    from DeepHyperX.models import get_model
    model, optimizer, loss, hyperparams = get_model('boulch', **hyperparams)

    CUDA_DEVICE = get_device(0) # hardcoded
    hyperparams['device'] = CUDA_DEVICE

    from torch.utils import data
    loader = data.DataLoader(wrapped_dataset,
                    batch_size=hyperparams['batch_size'],
                    pin_memory=hyperparams['device'],
                    shuffle=True)

    import visdom
    viz_autoencoder = visdom.Visdom(env='autoencoder')

    if custom_epoch_number == None:
        num_epochs = hyperparams['epoch']
    else:
        num_epochs = custom_epoch_number

    # visdom not necessary; val_loader breaks things (5 dimensions instead of 3, however this occurs)
    path, compressed_X = autoencoder_train(model, optimizer, loader, num_epochs=num_epochs)

    print("\nAutoencoder applied, model saved at "+path+"\n")

    return compressed_X

def applyFeatureExtraction(X, y, extraction_object, mode, merged=False):
    if merged:
        X_reshaped = np.reshape(X, (-1, X.shape[3]))
        y_reshaped = np.reshape(y, y.shape[0] * y.shape[1] * y.shape[2])
    else:
        X_reshaped = np.reshape(X, (-1, X.shape[2]))
        y_reshaped = np.reshape(y, y.shape[0] * y.shape[1])

    if mode == "supervised" or mode == "mixed":
        X_compressed = extraction_object.fit_transform(X_reshaped, y_reshaped)
    elif mode == "unsupervised": # todo maybe check vs. supervised for very few examples, not for all
        X_compressed = extraction_object.fit_transform(X_reshaped)
    else:
        raise ValueError("Unknown mode "+mode+", choose between supervised and unsupervised")

    if merged:
        X_compressed = np.reshape(X_compressed, (X.shape[0], X.shape[1], X.shape[2], -1))
    else:
        X_compressed = np.reshape(X_compressed, (X.shape[0], X.shape[1], -1))

    return X_compressed, extraction_object


# https://www.kaggle.com/sz8416/6-ways-for-feature-selection
def applyFeatureSelection(X, y, algorithm, n_components, mode, merged=False):
    if merged:
        newX = np.reshape(X, (-1, X.shape[3]))
        newY = np.reshape(y, y.shape[0] * y.shape[1] * y.shape[2])
    else:
        newX = np.reshape(X, (-1, X.shape[2]))
        newY = np.reshape(y, y.shape[0] * y.shape[1])
    feature = None
    # different ways to select features with wrapper methods: https://stackabuse.com/applying-wrapper-methods-in-python-for-feature-selection/
    # RFE vs. SFS: https://stackoverflow.com/questions/35640168/wrapper-methods-for-feature-selection-machine-learning-in-scikit-learn
    if mode == "forward":
        #todo"""takes forever, maybe change params like n_jobs"""
        from mlxtend.feature_selection import SequentialFeatureSelector
        feature = SequentialFeatureSelector(algorithm, k_features=n_components, forward=True)
    elif mode == "backward_threshold" or mode == 'mixed':
        from sklearn.feature_selection import SelectFromModel
        feature = SelectFromModel(algorithm, max_features=n_components)
    elif mode == "backward_iterate":
        """legacy option, way slower than with a threshold"""
        from sklearn.feature_selection import RFE
        feature = RFE(algorithm, n_features_to_select=n_components)
    else:
        raise ValueError("Unknown feature selection mode "+mode)

    new_X = feature.fit_transform(newX, newY)

    if mode == 'mixed':
        formerX = newX.transpose()
        new_X = new_X.transpose()
        # remove features already present
        unused_X = []
        for i in range(len(formerX)):
            newXrow = formerX[i]

            found = False
            for j in range(len(new_X)):
                checkXrow = new_X[j]
                if (newXrow == checkXrow).all():
                    found = True
                    break
            if not found:
                unused_X.append(newXrow)

        # add missing features
        from sklearn.feature_selection import RFE

        # n_features_to_select means that this many features will REMAIN afterwards, not that they are selected for removal
        feature = RFE(algorithm, n_features_to_select=n_components - new_X.shape[0])

        unused_X = np.array(unused_X).transpose()
        add_me = feature.fit_transform(unused_X, newY)
        # merge
        new_X = np.concatenate((new_X.transpose(), add_me), axis=1)

    if merged:
        new_X = np.reshape(new_X, (X.shape[0], X.shape[1], X.shape[2], -1))
    else:
        new_X = np.reshape(new_X, (X.shape[0], X.shape[1], -1))

    return new_X, feature