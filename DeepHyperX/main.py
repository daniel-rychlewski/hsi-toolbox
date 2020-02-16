# -*- coding: utf-8 -*-
"""
DEEP LEARNING FOR HYPERSPECTRAL DATA.

This script allows the user to run several deep models (and SVM baselines)
against various hyperspectral datasets. It is designed to quickly benchmark
state-of-the-art CNNs on various public hyperspectral datasets.

This code is released under the GPLv3 license for non-commercial and research
purposes only.
For commercial use, please contact the authors.
"""
from __future__ import division
# Python 2/3 compatiblity
from __future__ import print_function

import argparse
import time
from datetime import timedelta

# Numpy, scipy, scikit-image, spectral
import numpy as np
import pandas as pd
# Visualization
import seaborn as sns
import sklearn.model_selection
# Torch
import torch
import torch.utils.data as data
import visdom
from pandas.tests.sparse.frame.test_to_from_scipy import scipy

from DeepHyperX.batch import STORE_EXPERIMENT_LOCATION, \
    PYTHON_INTERPRETER_LOCATION, VIS_PY_LOCATION
from DeepHyperX.datasets import get_dataset, HyperX, open_file, DATASETS_CONFIG
from DeepHyperX.image_compression import apply_band_selection_choice
from DeepHyperX.models import get_model, train, test, save_model
from DeepHyperX.utils import metrics, convert_to_color_, convert_from_color_, \
    display_dataset, display_predictions, explore_spectrums, plot_spectrums, \
    sample_gt, build_dataset, show_results, compute_imf_weights, get_device, start_mem_measurement, stop_mem_measurement
from DeepHyperX.utils import print_memory_metrics


# https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def mean_confidence_interval(data, confidence=0.95):
    try:
        a = 1.0 * np.array(data)
        n = len(a)

        # not necessary anymore, but does not hurt to leave it in
        if a[0].__class__ == torch.Tensor:
            a = list(float(b) for b in a)

        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m-h, m+h
    except ValueError:
        return 0,0

dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default=None, choices=dataset_names,
                    help="Dataset to use.")
parser.add_argument('--model', type=str, default=None,
                    help="Model to train. Available:\n"
                    "SVM (linear), "
                    "SVM_grid (grid search on linear, poly and RBF kernels), "
                    "baseline (fully connected NN), "
                    "hu (1D CNN), "
                    "hamida (3D CNN + 1D classifier), "
                    "lee (3D FCN), "
                    "chen (3D CNN), "
                    "li (3D CNN), "
                    "he (3D CNN), "
                    "luo (3D CNN), "
                    "sharma (2D CNN), "
                    "boulch (1D semi-supervised CNN), "
                    "liu (2D semi-supervised CNN), "
                    "mou (1D RNN)")
parser.add_argument('--folder', type=str, help="Folder where to store the "
                    "datasets (defaults to the current working directory).",
                    default="./Datasets/")
parser.add_argument('--cuda', type=int, default=-1,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--runs', type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument('--restore', type=str, default=None,
                    help="Weights to use for initialization, e.g. a checkpoint")

# Dataset options
group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--training_sample', type=float, default=10,
                    help="Percentage of samples to use for training (default: 10%%)")
group_dataset.add_argument('--sampling_mode', type=str, help="Sampling mode"
                    " (random sampling or disjoint, default: random)",
                    default='random')
group_dataset.add_argument('--train_set', type=str, default=None,
                    help="Path to the train ground truth (optional, this "
                    "supersedes the --sampling_mode option)")
group_dataset.add_argument('--test_set', type=str, default=None,
                    help="Path to the test set (optional, by default "
                    "the test_set is the entire ground truth minus the training)")
# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--epoch', type=int, help="Training epochs (optional, if"
                    " absent will be set by the model)")
group_train.add_argument('--patch_size', type=int,
                    help="Size of the spatial neighbourhood (optional, if "
                    "absent will be set by the model)")
group_train.add_argument('--lr', type=float,
                    help="Learning rate, set by the model if not specified.")
group_train.add_argument('--class_balancing', action='store_true',
                    help="Inverse median frequency class balancing (default = False)")
group_train.add_argument('--batch_size', type=int,
                    help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--test_stride', type=int, default=1,
                     help="Sliding window step stride during inference (default = 1)")
# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true',
                    help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true',
                    help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true',
                    help="Random mixes between spectra")

parser.add_argument('--with_exploration', action='store_true',
                    help="See data exploration visualization")
parser.add_argument('--download', type=str, default=None, nargs='+',
                    choices=dataset_names,
                    help="Download the specified datasets and quits.")

# Mode options
parser.add_argument('--mode', help="Feature extraction / selection mode", choices={'supervised', 'unsupervised', 'forward', 'backward_threshold', 'backward_iterate', 'mixed'}, default=None)

# Model compression options (Intel Distiller is used for quantization)
parser.add_argument('--prune', help="Include this if you want your model to be pruned after training", action='store_true')

# Pruning options
parser.add_argument('--prune_percent', help="Percentage of 0-100 percent that describes how many weights should be pruned", default=None)
parser.add_argument('--prune_epochs', help="Prune epochs for retraining for the iterative pruning", default=10)
parser.add_argument('--alpha', help="alpha parameter for pruning", default=0.5)

# Image compression options
parser.add_argument('--band_selection', help="Choose dimensionality reduction technique for reducing band count of the hyperspectral dataset", default=None)
parser.add_argument('--cumulated_band_selection', help="Apply all three band selections for train,val,test at once for guaranteed one fixed number of bands remaining for train,val,test?", default=False)
parser.add_argument('--autoencoder_epochs', help="epochs for autoencoder usage", default=None)
parser.add_argument('--n_components', help="n components for band selection", default=75)
parser.add_argument('--speedup', help='factor by which epochs are reduced (for all models)', default=1)

# Info arguments for batch execution to note something that influences program flow at some point
parser.add_argument('--formerly_used_technique', help="band selection technique previously applied on the model for the checkpoint that is being read in now", default=None)
parser.add_argument('--old_n_components', help="n_components for band selection technique previously applied on the model for the checkpoint that is being read in now", default=None)

args = parser.parse_args()

CUDA_DEVICE = get_device(int(args.cuda))

# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
# Data augmentation ?
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Dataset name
DATASET = args.dataset
# Model name
MODEL = args.model
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride
# # Model Compression
DO_PRUNE = args.prune
PRUNE_PERCENT = float(args.prune_percent) if args.prune_percent is not None else None
# Image Compression
BAND_SELECTION_TECHNIQUE = args.band_selection
CUMULATED_BAND_SELECTION = args.cumulated_band_selection
N_COMPONENTS = args.n_components
SPEEDUP = args.speedup

FORMER_TECHNIQUE = args.formerly_used_technique
FORMER_COMPONENTS = args.old_n_components

dataframe_grid = []
df_column_entry_dict = {}

# Download dataset
if args.download is not None and len(args.download) > 0:
    for dataset in args.download:
        get_dataset(dataset, target_folder=FOLDER)
    quit()

VISDOM_ENV = MODEL + '_' + DATASET
viz = visdom.Visdom(env=VISDOM_ENV)
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")


hyperparams = vars(args)
# Load the dataset
img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET,
                                                               FOLDER)
# Number of classes
N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]

# Parameters for the SVM grid search
SVM_GRID_PARAMS = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3],
                                       'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
                   {'kernel': ['poly'], 'degree': [3], 'gamma': [1e-1, 1e-2, 1e-3]}]

if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(x):
    return convert_to_color_(x, palette=palette)
def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)


# Instantiate the experiment based on predefined networks
hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'device': CUDA_DEVICE})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

# Show the image and the ground truth
display_dataset(img, gt, RGB_BANDS, LABEL_VALUES, palette, viz)
color_gt = convert_to_color(gt)

if DATAVIZ:
    # Data exploration : compute and show the mean spectrums
    mean_spectrums = explore_spectrums(img, gt, LABEL_VALUES, viz,
                                       ignored_labels=IGNORED_LABELS)
    plot_spectrums(mean_spectrums, viz, title='Mean spectrum/class')

results = []
# run the experiment several times
# pruning and quantization can be done after each respective run
for run in range(N_RUNS):
    df_column_entry_dict.clear()
    if TRAIN_GT is not None and TEST_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = open_file(TEST_GT)
    elif TRAIN_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = np.copy(gt)
        w, h = test_gt.shape
        test_gt[(train_gt > 0)[:w,:h]] = 0
    elif TEST_GT is not None:
        test_gt = open_file(TEST_GT)
    else:
	# Sample random training spectra
        train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)
    print("{} samples selected (over {})".format(np.count_nonzero(train_gt),
                                                 np.count_nonzero(gt)))
    print("Running an experiment with the {} model".format(MODEL),
          "run {}/{}".format(run + 1, N_RUNS))

    display_predictions(convert_to_color(train_gt), viz, caption="Train ground truth")
    display_predictions(convert_to_color(test_gt), viz, caption="Test ground truth")

    path = None # Where is the model saved to (= folder + filename)
    # Pass these data loaders around for pruning later on. Will be set in this model if-else
    train_loader = None
    val_loader = None
    # Distinguish by model type first. If neural network, choose the appropriate one later
    if MODEL == 'SVM_grid':
        print("Running a grid search SVM")
        # Grid search SVM (linear and RBF)
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=IGNORED_LABELS)
        class_weight = 'balanced' if CLASS_BALANCING else None
        from sklearn.svm import SVC
        clf = sklearn.svm.SVC(class_weight=class_weight)
        from sklearn.model_selection import GridSearchCV
        clf = sklearn.model_selection.GridSearchCV(clf, SVM_GRID_PARAMS, verbose=5, n_jobs=4)

        print_memory_metrics("got model/before training", df_column_entry_dict)
        start_mem_measurement()
        start = time.time()

        clf.fit(X_train, y_train)

        time_elapse = time.time() - start

        event = 'model.predict'
        formatted_time = str(timedelta(seconds=time_elapse))
        df_column_entry_dict['Time measurement at '+event+' [s]'] = time_elapse

        print("\n"+event+" took "+ formatted_time + " seconds\n")

        event = "after training"
        stop_mem_measurement(event, df_column_entry_dict)
        print_memory_metrics(event, df_column_entry_dict)

        print("SVM best parameters : {}".format(clf.best_params_))
        print_memory_metrics("before inference", df_column_entry_dict)
        start_mem_measurement()
        start = time.time()
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        time_elapse = time.time() - start

        event = 'model.predict'
        formatted_time = str(timedelta(seconds=time_elapse))
        df_column_entry_dict['Time measurement at '+event+' [s]'] = time_elapse

        print("\n"+event+" took "+ formatted_time + " seconds\n")

        event = "after inference"
        stop_mem_measurement(event, df_column_entry_dict)
        print_memory_metrics(event, df_column_entry_dict)
        path = save_model(clf, MODEL, DATASET)
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == 'SVM':
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=IGNORED_LABELS)
        class_weight = 'balanced' if CLASS_BALANCING else None
        from sklearn.svm import SVC
        clf = sklearn.svm.SVC(class_weight=class_weight)
        print_memory_metrics("got model/before training", df_column_entry_dict)
        start_mem_measurement()
        start = time.time()

        clf.fit(X_train, y_train)

        time_elapse = time.time() - start
        event = 'model.fit'
        formatted_time = str(timedelta(seconds=time_elapse))
        df_column_entry_dict['Time measurement at '+event+' [s]'] = time_elapse
        print("\n"+event+" took "+ formatted_time + " seconds\n")
        event = "after training"
        stop_mem_measurement(event, df_column_entry_dict)
        print_memory_metrics(event, df_column_entry_dict)

        path = save_model(clf, MODEL, DATASET)

        print_memory_metrics("before inference", df_column_entry_dict)
        start_mem_measurement()
        start = time.time()

        prediction = clf.predict(img.reshape(-1, N_BANDS))

        time_elapse = time.time() - start
        event = 'model.predict'
        formatted_time = str(timedelta(seconds=time_elapse))
        df_column_entry_dict['Time measurement at '+event+' [s]'] = time_elapse
        print("\n"+event+" took "+ formatted_time + " seconds\n")

        event = "after inference"
        stop_mem_measurement(event, df_column_entry_dict)
        print_memory_metrics(event, df_column_entry_dict)
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == 'SGD':
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=IGNORED_LABELS)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        class_weight = 'balanced' if CLASS_BALANCING else None
        from sklearn.linear_model import SGDClassifier
        clf = sklearn.linear_model.SGDClassifier(class_weight=class_weight, learning_rate='optimal', tol=1e-3, average=10)
        print_memory_metrics("got model/before training", df_column_entry_dict)
        start_mem_measurement()
        start = time.time()

        clf.fit(X_train, y_train)

        time_elapse = time.time() - start
        event = 'model.fit'
        formatted_time = str(timedelta(seconds=time_elapse))
        df_column_entry_dict['Time measurement at '+event+' [s]'] = time_elapse
        print("\n"+event+" took "+ formatted_time + " seconds\n")
        event = "after training"
        stop_mem_measurement(event, df_column_entry_dict)
        print_memory_metrics("after training", df_column_entry_dict)

        path = save_model(clf, MODEL, DATASET)

        print_memory_metrics("before inference", df_column_entry_dict)
        start_mem_measurement()
        start = time.time()

        prediction = clf.predict(scaler.transform(img.reshape(-1, N_BANDS)))

        time_elapse = time.time() - start
        event = 'model.predict'
        formatted_time = str(timedelta(seconds=time_elapse))
        df_column_entry_dict['Time measurement at '+event+' [s]'] = time_elapse
        print("\n"+event+" took "+ formatted_time + " seconds\n")

        event = "after inference"
        stop_mem_measurement(event, df_column_entry_dict)
        print_memory_metrics(event, df_column_entry_dict)
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == 'nearest':
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=IGNORED_LABELS)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        class_weight = 'balanced' if CLASS_BALANCING else None
        from sklearn.neighbors import KNeighborsClassifier
        clf = sklearn.neighbors.KNeighborsClassifier(weights='distance')
        clf = sklearn.model_selection.GridSearchCV(clf, {'n_neighbors': [1, 3, 5, 10, 20]}, verbose=5, n_jobs=4)
        print_memory_metrics("got model/before training", df_column_entry_dict)
        start_mem_measurement()
        start = time.time()

        clf.fit(X_train, y_train)

        time_elapse = time.time() - start
        event = 'model.fit'
        formatted_time = str(timedelta(seconds=time_elapse))
        df_column_entry_dict['Time measurement at '+event+' [s]'] = time_elapse

        print("\n"+event+" took "+ formatted_time + " seconds\n")
        event = "after training"
        stop_mem_measurement(event, df_column_entry_dict)
        print_memory_metrics(event, df_column_entry_dict)

        path = save_model(clf, MODEL, DATASET)

        print_memory_metrics("before inference", df_column_entry_dict)
        start_mem_measurement()
        start = time.time()

        prediction = clf.predict(img.reshape(-1, N_BANDS))

        time_elapse = time.time() - start
        event = 'model.predict'
        formatted_time = str(timedelta(seconds=time_elapse))
        df_column_entry_dict['Time measurement at '+event+' [s]'] = time_elapse
        print("\n"+event+" took "+ formatted_time + " seconds\n")

        event = "after inference"
        stop_mem_measurement(event, df_column_entry_dict)
        print_memory_metrics(event, df_column_entry_dict)
        prediction = prediction.reshape(img.shape[:2])
    else:
        # Neural network
        if CLASS_BALANCING:
            weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
            hyperparams['weights'] = torch.from_numpy(weights)

        _, _, _, hyperparams = get_model(MODEL, **hyperparams) # todo ugly to get model twice, but only this sets patch size

        # Split train set in train/val
        train_gt, val_gt = sample_gt(train_gt, 0.95, mode='random')
        # Generate the dataset
        train_dataset = HyperX(data=img, gt=train_gt, hyperparams=hyperparams)

        if BAND_SELECTION_TECHNIQUE is not None and not CUMULATED_BAND_SELECTION:
            train_dataset = HyperX(data=apply_band_selection_choice(BAND_SELECTION_TECHNIQUE, train_dataset.data, train_gt, args.mode, hyperparams, train_dataset, custom_epoch_number=args.autoencoder_epochs, n_components=N_COMPONENTS, df_column_entry_dict=df_column_entry_dict), gt=train_gt, hyperparams=hyperparams)

        val_dataset = HyperX(data=img, gt=val_gt, hyperparams=hyperparams)

        if BAND_SELECTION_TECHNIQUE is not None and not CUMULATED_BAND_SELECTION:
            val_dataset = HyperX(data=apply_band_selection_choice(BAND_SELECTION_TECHNIQUE, val_dataset.data, val_gt, args.mode, hyperparams, val_dataset, custom_epoch_number=args.autoencoder_epochs, n_components=N_COMPONENTS, df_column_entry_dict=df_column_entry_dict), gt=val_gt, hyperparams=hyperparams)
            old_n_bands = hyperparams['n_bands']  # back it up for autoencoder validation
            hyperparams['n_bands'] = train_dataset.data.shape[2]  # adjust for the correct model to be constructed - only important in case of band reduction having taken place, otherwise, this line doesn't change anything

        img_dataset = HyperX(data=img, gt=gt, hyperparams=hyperparams)

        if BAND_SELECTION_TECHNIQUE == "Autoencoder":
            hyperparams['n_bands'] = old_n_bands  # boulch_et_al autoencoder needs this correct number of bands for verification
        if BAND_SELECTION_TECHNIQUE is not None and not CUMULATED_BAND_SELECTION:
            img_dataset = HyperX(data=apply_band_selection_choice(BAND_SELECTION_TECHNIQUE, img_dataset.data, gt, args.mode, hyperparams, img_dataset, custom_epoch_number=args.autoencoder_epochs, n_components=N_COMPONENTS, df_column_entry_dict=df_column_entry_dict), gt=gt, hyperparams=hyperparams)

        if BAND_SELECTION_TECHNIQUE is not None and CUMULATED_BAND_SELECTION:
            all_datasets = np.stack((train_dataset.data, val_dataset.data, img_dataset.data), axis=3)
            all_gts = np.stack((train_gt, val_gt, gt), axis=2)
            all_datasets = np.array(all_datasets).reshape(
                (all_datasets.shape[0], all_datasets.shape[1], all_datasets.shape[3], all_datasets.shape[2]))
            all_datasets = apply_band_selection_choice(BAND_SELECTION_TECHNIQUE, all_datasets, all_gts, args.mode, hyperparams,
                                    val_dataset, custom_epoch_number=args.autoencoder_epochs, n_components=N_COMPONENTS,
                                    df_column_entry_dict=df_column_entry_dict)
            train_dataset = HyperX(data=np.array_split(all_datasets, indices_or_sections=3, axis=2)[0].squeeze(), gt=train_gt, hyperparams=hyperparams)
            val_dataset   = HyperX(data=np.array_split(all_datasets, indices_or_sections=3, axis=2)[1].squeeze(), gt=val_gt  , hyperparams=hyperparams)
            img_dataset   = HyperX(data=np.array_split(all_datasets, indices_or_sections=3, axis=2)[2].squeeze(), gt=gt      , hyperparams=hyperparams)

        train_loader = data.DataLoader(train_dataset,
                                       batch_size=hyperparams['batch_size'],
                                       pin_memory=hyperparams['device'],
                                       shuffle=True)

        val_loader = data.DataLoader(val_dataset,
                                     pin_memory=hyperparams['device'],
                                     batch_size=hyperparams['batch_size'])

        hyperparams['speedup'] = float(SPEEDUP) # reduce number of epochs (and by this, execution time as well, especially for repeated batch executions that would have taken weeks)
        if FORMER_COMPONENTS is not None:
            hyperparams['n_bands'] = int(FORMER_COMPONENTS) # adjust for the correct model to be constructed - only important in case of band reduction having taken place, otherwise, this line doesn't change anything
        else:
            hyperparams['n_bands'] = train_dataset.data.shape[2]

        model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)
        print_memory_metrics("got model", df_column_entry_dict)

        print(hyperparams)
        print("Network :")
        with torch.no_grad():
            for input, _ in train_loader:
                break
                # edit torchsummary return value to:
                # return float(total_params), float(trainable_params), float(total_params-trainable_params), total_input_size, total_output_size, total_params_size, total_size
            # total_params, trainable_params, non_trainable_params, total_input_size, total_output_size, total_params_size, total_size = summary(model.to(hyperparams['device']), input.size()[1:])
            #
            # df_column_entry_dict['Total params'] = total_params
            # df_column_entry_dict['Trainable params'] = trainable_params
            # df_column_entry_dict['Non-trainable params'] = non_trainable_params
            #
            # df_column_entry_dict['Input size [MB]'] = total_input_size
            # df_column_entry_dict['Forward/backward pass size [MB]'] = total_output_size
            # df_column_entry_dict['Params size [MB]'] = total_params_size
            # df_column_entry_dict['Estimated Total Size [MB]'] = total_size

        if CHECKPOINT is not None:
            model.load_state_dict(torch.load(CHECKPOINT))#['state_dict'])

        try:
            print_memory_metrics("before training", df_column_entry_dict)
            start_mem_measurement()
            start = time.time()
            path = train(model, optimizer, loss, train_loader, hyperparams['epoch'],
                  scheduler=hyperparams['scheduler'], device=hyperparams['device'],
                  supervision=hyperparams['supervision'], val_loader=val_loader,
                  display=viz)
            time_elapse = time.time() - start
            event = 'model.train'
            formatted_time = str(timedelta(seconds=time_elapse))
            df_column_entry_dict['Time measurement at ' + event + ' [s]'] = time_elapse

            print("\n"+event+" took " + formatted_time + " seconds\n")
            event = "after training"
            stop_mem_measurement(event, df_column_entry_dict)
            print_memory_metrics(event, df_column_entry_dict)
        except KeyboardInterrupt:
            # Allow the user to stop the training
            pass

        print_memory_metrics("before inference", df_column_entry_dict)
        start_mem_measurement()
        start = time.time()

        probabilities = test(model, img_dataset.data, hyperparams)

        time_elapse = time.time() - start
        event = 'model.test'
        formatted_time = str(timedelta(seconds=time_elapse))
        df_column_entry_dict['Time measurement at ' + event + ' [s]'] = time_elapse
        print("\n"+event+" took " + formatted_time + " seconds\n")
        event = "after inference"
        stop_mem_measurement(event, df_column_entry_dict)
        print_memory_metrics(event, df_column_entry_dict)

        prediction = np.argmax(probabilities, axis=-1)

        if EPOCH == 0 and CHECKPOINT is not None:
            path = CHECKPOINT

    # End of model distinction loop, now evaluate the model
    run_results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=N_CLASSES)

    mask = np.zeros(gt.shape, dtype='bool')
    for l in IGNORED_LABELS:
        mask[gt == l] = True
    prediction[mask] = 0

    color_prediction = convert_to_color(prediction)
    display_predictions(color_prediction, viz, gt=convert_to_color(test_gt), caption="Prediction vs. test ground truth")

    results.append(run_results)
    show_results(run_results, viz, label_values=LABEL_VALUES, df_column_entry_dict=df_column_entry_dict)

    # Do model compression
    # Compression techniques
    prune_model_path = None
    quantize_model_path = None
    if DO_PRUNE:
        print("Pruning the model...")
        from DeepHyperX.model_compression import *
        from DeepHyperX.models import *

        # Weights are initialized randomly when constructing the model -> therefore, load all of the weights before passing the model to the prune method
        state_dict = torch.load(path)

        # another get_model not necessary to get variable model
        # model, _, _, hyperparams = get_model(MODEL, **hyperparams) # hyperparams contains all the important information, especially which dataset to use
        model.load_state_dict(state_dict)

        hyperparams['former_technique'] = FORMER_TECHNIQUE
        hyperparams['former_components'] = FORMER_COMPONENTS

        # no eval() because we'll prune the model first - validate() of iterative pruning will set it to evaluation mode when needed
        prune(args, model, PRUNE_PERCENT, train_loader, val_loader, hyperparams, df_column_entry_dict)

    dataframe_grid.append(list(df_column_entry_dict.values()))
frame = pd.DataFrame(dataframe_grid, columns=list(df_column_entry_dict.keys()))
# Compute mean, std, 95% confidence intervals for this data frame for the relevant columns

means = frame.mean()
stddevs = frame.std()

low_dict = {}
high_dict = {}

conf_temp = frame.apply(mean_confidence_interval)
for key, value in zip(list(df_column_entry_dict.keys()), conf_temp.get_values()):
    low_dict[key]  = value[0]
    high_dict[key] = value[1]

frame = frame.append(pd.DataFrame(low_dict, index=[0]), ignore_index=True)
frame = frame.append(pd.DataFrame(high_dict, index=[0]), ignore_index=True)
frame = frame.append(means, ignore_index=True)
frame = frame.append(stddevs, ignore_index=True)

import os
os.makedirs(STORE_EXPERIMENT_LOCATION, exist_ok=True)
frame.to_excel(STORE_EXPERIMENT_LOCATION + MODEL + "_" + DATASET + ".xlsx", index=False)
#frame.to_excel(STORE_EXPERIMENT_LOCATION + MODEL + "_" + DATASET +"_"+ os.path.basename(CHECKPOINT)+ ".xlsx", index=False)

# back up visdom results
start_command = PYTHON_INTERPRETER_LOCATION + " " + VIS_PY_LOCATION + " --save " + VISDOM_ENV + " --file " + STORE_EXPERIMENT_LOCATION + VISDOM_ENV + ".visdom"
os.system(start_command)

# Show cumulative results at the very end in case of multiple runs
if N_RUNS > 1:
    show_results(results, viz, label_values=LABEL_VALUES, aggregated=True, df_column_entry_dict=df_column_entry_dict)