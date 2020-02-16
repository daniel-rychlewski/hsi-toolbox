from numpy.ma import arange

from DeepHyperX.batch import BAND_SELECTION_CHECKPOINTS, DEEPHYPERX_PATH_LINUX, \
    COMPRESS_CLASSIFIER_PATH_LINUX
from DeepHyperX.batch_prune import MODEL_QUANTIZE_RESTORE_PATH

cao_alphas = list(map(lambda x: "0."+str(x), range(1,10,1)))
cao_alphas.extend(map(lambda x: "1."+str(x), range(0,10,1)))

he_alphas = list(map(lambda x: "0."+str(x), range(1,10,1)))
he_alphas.extend(map(lambda x: "1."+str(x), range(0,10,1)))
he_alphas.extend(map(lambda x: "0.0"+str(x), range(1,10,1)))
he_alphas.extend(map(lambda x: "0.00"+str(x), range(1,10,1)))
he_alphas.extend(["0.0001", "0.0005"])

hu_alphas = list(map(lambda x: "0."+str(x), range(1,10,1)))
hu_alphas.extend(map(lambda x: "1."+str(x), range(0,10,1)))

luo_alphas = list(map(lambda x: "1."+str(x)+"e-29", range(1,10,1)))
luo_alphas.extend(map(lambda x: "2."+str(x)+"e-29", range(1,10,1)))
luo_alphas.extend(["1e-29", "2e-29"])
# luo_alphas.extend(cao_alphas) # only linear layer matters in luo, so don't need cao alphas

santara_alphas = list(map(lambda x: "0."+str(x), range(1,10,1)))
santara_alphas.extend(map(lambda x: "1."+str(x), range(0,10,1)))
# convert to string lists because of rounding errors for float

from DeepHyperX.batch import MAIN_PY_LOCATION
from DeepHyperX.batch import PYTHON_INTERPRETER_LOCATION

alphas = {'cao': cao_alphas, 'he': he_alphas, 'hu': hu_alphas, 'luo_cnn': luo_alphas, 'santara': santara_alphas}
models_with_epochs = {"he": "100", "hu": "100", "luo_cnn": "100", "santara": "40", "cao": "100"}
startCommand = PYTHON_INTERPRETER_LOCATION + " " + MAIN_PY_LOCATION + " --training_sample 0.8 --runs 1 --with_exploration --cuda 0 --dataset IndianPines"

import os

def read_for_inference():

    for model, _ in models_with_epochs.items():
        for alpha in alphas[model]:
            # no prune percent as stopping criterion -> will stop when no longer pruned well
            from DeepHyperX.batch_prune import MODEL_PRUNE_RESTORE_PATH
            my_command = startCommand + " --model " + model + " --restore " + MODEL_PRUNE_RESTORE_PATH + model +"_alpha"+alpha+"_pruned.pth --epoch 0 1> " + MODEL_PRUNE_RESTORE_PATH + model + "_alpha"+alpha+"_inference.txt 2>&1"
            # print(my_command)
            os.system(my_command)

def read_after_band_selection(pths_path, the_technique, the_models, quantize_afterwards, the_n_components=[100,140], the_mode="mixed"):#the_n_components=[40,70,100,140,170]
    for model in the_models:
        for technique in [the_technique]:
            # for n_components in range(10,191,10):
            for n_components in the_n_components:
                for alpha in alphas[model]:
                    # no prune percent as stopping criterion -> will stop when no longer pruned well
                    from DeepHyperX.batch_prune import MODEL_PRUNE_RESTORE_PATH
                    my_command = startCommand + " --prune --alpha "+alpha+" --band_selection "+technique+" --n_components "+str(n_components)+" --mode "+the_mode+" --model " + model + " --restore " + pths_path + model +"_" + technique + "_" + str(n_components) + ".pth --epoch 0 --formerly_used_technique "+technique+" --old_n_components "+str(n_components)+" 1> " + MODEL_PRUNE_RESTORE_PATH \
                                 + model + "_alpha"+alpha+"_"+technique+"_"+str(n_components)+"_inference.txt 2>&1"
                    print(my_command)
                    os.system(my_command)

                    if quantize_afterwards:
                        # run distiller quantization
                        for (activations_bits, weights_bits, accumulator_bits) in [(8,16,32), (8,8,32), (8,4,16), (16, 16, 16), (8, 8, 8), (4, 4, 4), (2, 2, 2), (1, 1, 1)]:
                        # for (activations_bits, weights_bits, accumulator_bits) in [(8,16,32)]:
                            quantization_command = PYTHON_INTERPRETER_LOCATION + " " + \
                                                   COMPRESS_CLASSIFIER_PATH_LINUX + \
                                               " --arch " + model +" --dataset IndianPines --epochs 10 -p 1 --lr=0.001 --cuda 0 ../DeepHyperX/Datasets/" + \
                                               " --resume-from '" + DEEPHYPERX_PATH_LINUX + "outputs/Experiments/pruneMe/" + model + "_alpha" + alpha + "_" + technique + "_" + str(n_components) + "' --evaluate --quantize-eval " \
                                               "--qe-bits-acts " + str(activations_bits) +" --qe-bits-wts " + str(weights_bits) + " --qe-bits-accum " + str(accumulator_bits) + " --name " + model + technique + str(n_components) + "_IndianPines_ptquantize --out-dir ../distiller/outputs/band-selection-prune-combo/ --confusion --vs 0.8 " \
                                               "--formerly_used_technique " + technique + " --old_n_components " + str(n_components) + " " \
                                               "1> " + MODEL_QUANTIZE_RESTORE_PATH + model + "_alpha" + alpha + "_" + technique + "_" + str(n_components) + "_quantized_" + str(activations_bits) + "_" + str(weights_bits) + "_" + str(accumulator_bits) + "_inference.txt 2>&1"

                            os.system(quantization_command)

def quantize_bandselectedandpruned_model(folder, model, technique, all_n_components=range(20, 191, 10)):
    for n_components in all_n_components:
        for alpha in alphas[model]:
            for (activations_bits, weights_bits, accumulator_bits) in [(8,16,32), (8,8,32), (8,4,16), (16, 16, 16), (8, 8, 8), (4, 4, 4), (2, 2, 2), (1, 1, 1)]:
                quantization_command = PYTHON_INTERPRETER_LOCATION + " " + \
                                       COMPRESS_CLASSIFIER_PATH_LINUX + \
                                       " --arch " + model + " --dataset IndianPines --epochs 10 -p 1 --lr=0.001 --cuda 0 ../DeepHyperX/Datasets/" + \
                                       " --resume-from '" + folder + model + "_alpha" + alpha + "_" + technique + "_" + str(
                    n_components) + "' --evaluate --quantize-eval " \
                                    "--qe-bits-acts " + str(activations_bits) + " --qe-bits-wts " + str(
                    weights_bits) + " --qe-bits-accum " + str(
                    accumulator_bits) + " --name " + model + technique + str(n_components) + "_IndianPines_ptquantize --out-dir ../distiller/outputs/band-selection-prune-combo/ --confusion --vs 0.8 " \
                                        "--formerly_used_technique " + technique + " --old_n_components " + str(n_components) + " " \
                                               "1> " + MODEL_QUANTIZE_RESTORE_PATH + model + "_alpha" + alpha + "_" + technique + "_" + str(n_components) + "_quantized_" + str(activations_bits) + "_" + str(weights_bits) + "_" + str(accumulator_bits) + "_inference.txt 2>&1"
                print(quantization_command)
                os.system(quantization_command)

if __name__ == '__main__':
    pass
    # read_for_inference()
    # techniques: "PCA", "NMF", "LLE", "UMAP", "Autoencoder", "RandomForest", "LogisticRegression", "LinearRegression"
    # models: "he", "hu", "luo_cnn", "santara", "cao"

    # do when he PCA is complete (190 has passed). rest will be quantized automatically with quantize_afterwards=True - quantize again to be able to distinguish between quantization modes
    # quantize_bandselectedandpruned_model(DEEPHYPERX_PATH_LINUX9+"outputs/Experiments/pruneMe/", "he", "PCA")
    # quantize_bandselectedandpruned_model(DEEPHYPERX_PATH_LINUX9+"outputs/Experiments/pruneMe/", "luo_cnn", "PCA")

    # use after all band selection techniques for luo finish
    # quantize_bandselectedandpruned_model(DEEPHYPERX_PATH_LINUX+"outputs/Experiments/pruneMe/", "luo_cnn", "NMF", all_n_components=[40,70,100,140,170])
    # quantize_bandselectedandpruned_model(DEEPHYPERX_PATH_LINUX+"outputs/Experiments/pruneMe/", "luo_cnn", "UMAP", all_n_components=[40,70,100,140,170])
    # quantize_bandselectedandpruned_model(DEEPHYPERX_PATH_LINUX+"outputs/Experiments/pruneMe/", "luo_cnn", "LogisticRegression")
    # quantize_bandselectedandpruned_model(DEEPHYPERX_PATH_LINUX+"outputs/Experiments/pruneMe/", "luo_cnn", "LinearRegression")

    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="PCA", the_model="he", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="PCA", the_model="hu", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="PCA", the_models=["luo_cnn"], quantize_afterwards=True, the_n_components=[150], the_mode="supervised")
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="PCA", the_models=["luo_cnn"], quantize_afterwards=True, the_n_components=[160], the_mode="supervised")
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="PCA", the_models=["luo_cnn"], quantize_afterwards=True, the_n_components=[180], the_mode="supervised")
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="PCA", the_models=["luo_cnn"], quantize_afterwards=True, the_n_components=[190], the_mode="supervised")
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="PCA", the_model="santara", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="PCA", the_model="cao", quantize_afterwards=True)

    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="NMF", the_models=["he"], quantize_afterwards=True, the_n_components=[170,140])
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="NMF", the_model="hu", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="NMF", the_model="luo_cnn", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="NMF", the_model="santara", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="NMF", the_model="cao", quantize_afterwards=True)

    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="NMF", the_models=["santara"], quantize_afterwards=True, the_n_components=[100], the_mode="supervised")
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="NMF", the_models=["santara"], quantize_afterwards=True, the_n_components=[140], the_mode="supervised")
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="NMF", the_models=["santara"], quantize_afterwards=True, the_n_components=[170], the_mode="supervised")

    # LLE only until 100 - pca, nmf better alternatives and for any n_components
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="LLE", the_model="he", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="LLE", the_model="hu", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="LLE", the_model="luo_cnn", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="LLE", the_model="santara", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="LLE", the_model="cao", quantize_afterwards=True)

    # autoencoder bad OA, PCA better for n=3
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="Autoencoder", the_model="he", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="Autoencoder", the_model="hu", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="Autoencoder", the_model="luo_cnn", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="Autoencoder", the_model="santara", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="Autoencoder", the_model="cao", quantize_afterwards=True)

    # random forest is bad
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="RandomForest", the_model="he", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="RandomForest", the_model="hu", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="RandomForest", the_model="luo_cnn", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="RandomForest", the_model="santara", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="RandomForest", the_model="cao", quantize_afterwards=True)

    # takes long, but our representative for projection based - tsne takes even longer. still, umap is bad
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="UMAP", the_model="he", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="UMAP", the_model="hu", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="UMAP", the_model="luo_cnn", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="UMAP", the_model="santara", quantize_afterwards=True)
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="UMAP", the_model="cao", quantize_afterwards=True)

    # 10 30 50 70 90, until 90
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="LogisticRegression", the_model="he", quantize_afterwards=True, the_n_components=[10,30,50,70,90], the_mode="backward_threshold")
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="LogisticRegression", the_model="hu", quantize_afterwards=True, the_n_components=[10,30,50,70,90], the_mode="backward_threshold")
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="LogisticRegression", the_model="luo_cnn", quantize_afterwards=True, the_n_components=[10,30,50,70,90], the_mode="backward_threshold")
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="LogisticRegression", the_model="santara", quantize_afterwards=True, the_n_components=[10,30,50,70,90], the_mode="backward_threshold")
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="LogisticRegression", the_model="cao", quantize_afterwards=True, the_n_components=[10,30,50,70,90], the_mode="backward_threshold")

    # until 60 linearregression without mixed. 20 30 40 50 60
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="LinearRegression", the_model="he", quantize_afterwards=True, the_n_components=[20,30,40,50,60], the_mode="backward_threshold")
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="LinearRegression", the_model="hu", quantize_afterwards=True, the_n_components=[20,30,40,50,60], the_mode="backward_threshold")
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="LinearRegression", the_model="luo_cnn", quantize_afterwards=True, the_n_components=[20,30,40,50,60], the_mode="backward_threshold")
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="LinearRegression", the_model="santara", quantize_afterwards=True, the_n_components=[20,30,40,50,60], the_mode="backward_threshold")
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="LinearRegression", the_model="cao", quantize_afterwards=True, the_n_components=[20,30,40,50,60], the_mode="backward_threshold")

    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="LinearRegression", the_models=["hu", "santara"], quantize_afterwards=True, the_n_components=[50], the_mode="backward_threshold")
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="LinearRegression", the_models=["luo_cnn", "cao"], quantize_afterwards=True, the_n_components=[50], the_mode="backward_threshold")

    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="LinearRegression", the_models=["hu","santara","cao","he"], quantize_afterwards=True, the_n_components=[70], the_mode="backward_threshold")
    # read_after_band_selection(pths_path=BAND_SELECTION_CHECKPOINTS, the_technique="LinearRegression", the_models=["luo_cnn","hu","santara","cao","he"], quantize_afterwards=True, the_n_components=[90], the_mode="backward_threshold")