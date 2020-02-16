" --restore 'D:/Experiments/allDatasetsAllModels6Runs/he_et_al/IndianPines/2019-07-01 18-39-24.425846_epoch100_0.97.pth'  --epoch 1 --prune_epochs 7"
import json
import os

from DeepHyperX.batch import PARAMETER_JSON, STORE_EXPERIMENT_LOCATION, PYTHON_INTERPRETER_LOCATION, MAIN_PY_LOCATION, \
    DEEPHYPERX_PATH, DEEPHYPERX_PATH_LINUX, COMPRESS_CLASSIFIER_PATH_LINUX
from conversion_testfile import pth_to_onnx

RESTORE_PATH = DEEPHYPERX_PATH + "outputs/Experiments/"
MODEL_PRUNE_RESTORE_PATH = RESTORE_PATH + "pruneMe/"
MODEL_QUANTIZE_RESTORE_PATH = RESTORE_PATH + "quantizeMe/"
BAND_SELECTION_RESTORE_PATH = RESTORE_PATH + "bandSelection/"

def model_prune():
    """Do pruning of the models by varying the alpha parameter. use iterative_pruning integrated in DeepHyperX"""
    startCommand = PYTHON_INTERPRETER_LOCATION+" "+MAIN_PY_LOCATION+" --training_sample 0.8 --runs 1 --with_exploration --cuda 0 --dataset IndianPines"

    # prune epochs = 10% of training epochs
    # models_with_prune_epochs = {"he":"10", "hu":"10", "luo_cnn":"10", "santara":"4", "cao":"10"}
    models_with_prune_epochs = {"luo_cnn":"10"}
    for key, value in models_with_prune_epochs.items():
        import numpy as np
        for alpha in np.arange(1e-29, 3e-29, 1e-30):
        # with 1==1:
        #     alpha = .01
            # no prune percent as stopping criterion -> will stop when no longer pruned well
            my_command = startCommand + " --model " + key + " --prune --alpha "+str(alpha)+" --restore " + MODEL_PRUNE_RESTORE_PATH + key + ".pth --epoch 0 --prune_epochs " + value + " 1> " + MODEL_PRUNE_RESTORE_PATH + key + str(alpha) + ".txt 2>&1"
            # print(my_command)
            os.system(my_command)

def band_prune():
    """Do band selection methods integrated in DeepHyperX, vary n_components and modes"""
    startCommand = PYTHON_INTERPRETER_LOCATION + " " + MAIN_PY_LOCATION + " --training_sample 0.8 --runs 1 --with_exploration --cuda 0 --dataset IndianPines"

    parameterFile = open(PARAMETER_JSON, "r")
    data = json.load(parameterFile)
    parameterFile.close()

    ex_techniques = ["Autoencoder"]
    se_techniques = []
    ex_modes = []
    se_modes = []

    extraction = data["image_compression"]["extraction"]
    extraction_techniques = extraction["techniques"]
    extraction_modes = extraction["modes"]

    selection = data["image_compression"]["selection"]
    selection_techniques = selection["techniques"]
    selection_modes = selection["modes"]

    ex_techniques.extend(extraction_techniques)
    se_techniques.extend(selection_techniques)
    ex_modes.extend(extraction_modes)
    se_modes.extend(selection_modes)

    # for model in ["he", "hu", "luo_cnn", "santara", "cao"]:
    #     for n_components in range(10, 191, 10):  # IndianPines has 200 bands
    #         for mode in ex_modes:
    #             for technique in ex_techniques:
    #                 # cannot just restore model. need to retrain a model every time because the number of components is different every time, so training test evaluate datasets will be as well
    #                 my_command = startCommand + " --model " + model + " --band_selection "+technique+" --mode "+mode+" --n_components "+str(n_components)+" --autoencoder_epochs "+str(10)+" 1> " + BAND_SELECTION_RESTORE_PATH + model +"_"+ technique +"_"+ mode +"_"+ str(n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)
    #         for mode in se_modes:
    #             for technique in se_techniques:
    #                 my_command = startCommand + " --model " + model + " --band_selection "+technique+" --mode "+mode+" --n_components "+str(n_components)+" --autoencoder_epochs "+str(10)+" 1> " + BAND_SELECTION_RESTORE_PATH + model +"_"+ technique +"_"+ mode +"_"+ str(n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)

    # for model in ["hu"]:
    #     for n_components in range(10, 191, 10):  # IndianPines has 200 bands
    #         for mode in ["supervised"]:
    #             for technique in ["PCA"]:
    #                 my_command = startCommand + " --model " + model + " --band_selection "+technique+" --mode "+mode+" --n_components "+str(n_components)+" --autoencoder_epochs "+str(10)+" 1> " + BAND_SELECTION_RESTORE_PATH + model +"_"+ technique +"_"+ mode +"_"+ str(n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)
    #
    # for model in ["hu"]:
    #     for n_components in range(10, 191, 10):  # IndianPines has 200 bands
    #         for mode in ["supervised"]:
    #             for technique in ["NMF"]:
    #                 my_command = startCommand + " --model " + model + " --band_selection " + technique + " --mode " + mode + " --n_components " + str(
    #                     n_components) + " --autoencoder_epochs " + str(
    #                     10) + " 1> " + BAND_SELECTION_RESTORE_PATH + model + "_" + technique + "_" + mode + "_" + str(
    #                     n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)

    # for model in ["hu"]:
    #     for n_components in range(10, 191, 10):  # IndianPines has 200 bands
    #         for mode in ["supervised"]:
    #             for technique in ["LLE"]:
    #                 my_command = startCommand + " --model " + model + " --band_selection " + technique + " --mode " + mode + " --n_components " + str(
    #                     n_components) + " --autoencoder_epochs " + str(
    #                     10) + " 1> " + BAND_SELECTION_RESTORE_PATH + model + "_" + technique + "_" + mode + "_" + str(
    #                     n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)
    #
    # for model in ["luo_cnn"]:
    #     for n_components in range(10, 191, 10):  # IndianPines has 200 bands
    #         for mode in ["supervised"]:
    #             for technique in ["PCA"]:
    #                 my_command = startCommand + " --model " + model + " --band_selection " + technique + " --mode " + mode + " --n_components " + str(
    #                     n_components) + " --autoencoder_epochs " + str(
    #                     10) + " 1> " + BAND_SELECTION_RESTORE_PATH + model + "_" + technique + "_" + mode + "_" + str(
    #                     n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)

    # for model in ["luo_cnn"]:
    #     for n_components in range(10, 191, 10):  # IndianPines has 200 bands
    #         for mode in ["supervised"]:
    #             for technique in ["NMF"]:
    #                 my_command = startCommand + " --model " + model + " --band_selection " + technique + " --mode " + mode + " --n_components " + str(
    #                     n_components) + " --autoencoder_epochs " + str(
    #                     10) + " 1> " + BAND_SELECTION_RESTORE_PATH + model + "_" + technique + "_" + mode + "_" + str(
    #                     n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)
    # for model in ["luo_cnn"]:
    #     for n_components in range(10, 191, 10):  # IndianPines has 200 bands
    #         for mode in ["supervised"]:
    #             for technique in ["LLE"]:
    #                 my_command = startCommand + " --model " + model + " --band_selection " + technique + " --mode " + mode + " --n_components " + str(
    #                     n_components) + " --autoencoder_epochs " + str(
    #                     10) + " 1> " + BAND_SELECTION_RESTORE_PATH + model + "_" + technique + "_" + mode + "_" + str(
    #                     n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)

    # for model in ["santara"]:
    #     for n_components in range(10, 191, 10):  # IndianPines has 200 bands
    #         for mode in ["supervised"]:
    #             for technique in ["PCA"]:
    #                 my_command = startCommand + " --model " + model + " --band_selection " + technique + " --mode " + mode + " --n_components " + str(
    #                     n_components) + " --autoencoder_epochs " + str(
    #                     10) + " 1> " + BAND_SELECTION_RESTORE_PATH + model + "_" + technique + "_" + mode + "_" + str(
    #                     n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)

    # for model in ["santara"]:
    #     for n_components in range(10, 191, 10):  # IndianPines has 200 bands
    #         for mode in ["supervised"]:
    #             for technique in ["NMF"]:
    #                 my_command = startCommand + " --model " + model + " --band_selection " + technique + " --mode " + mode + " --n_components " + str(
    #                     n_components) + " --autoencoder_epochs " + str(
    #                     10) + " 1> " + BAND_SELECTION_RESTORE_PATH + model + "_" + technique + "_" + mode + "_" + str(
    #                     n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)

    # for model in ["santara"]:
    #     for n_components in range(10, 191, 10):  # IndianPines has 200 bands
    #         for mode in ["supervised"]:
    #             for technique in ["LLE"]:
    #                 my_command = startCommand + " --model " + model + " --band_selection " + technique + " --mode " + mode + " --n_components " + str(
    #                     n_components) + " --autoencoder_epochs " + str(
    #                     10) + " 1> " + BAND_SELECTION_RESTORE_PATH + model + "_" + technique + "_" + mode + "_" + str(
    #                     n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)

    # for model in ["cao"]:
    #     for n_components in range(10, 191, 10):  # IndianPines has 200 bands
    #         for mode in ["supervised"]:
    #             for technique in ["PCA"]:
    #                 my_command = startCommand + " --model " + model + " --band_selection " + technique + " --mode " + mode + " --n_components " + str(
    #                     n_components) + " --autoencoder_epochs " + str(
    #                     10) + " 1> " + BAND_SELECTION_RESTORE_PATH + model + "_" + technique + "_" + mode + "_" + str(
    #                     n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)

    # for model in ["cao"]:
    #     for n_components in range(10, 191, 10):  # IndianPines has 200 bands
    #         for mode in ["supervised"]:
    #             for technique in ["NMF"]:
    #                 my_command = startCommand + " --model " + model + " --band_selection " + technique + " --mode " + mode + " --n_components " + str(
    #                     n_components) + " --autoencoder_epochs " + str(
    #                     10) + " 1> " + BAND_SELECTION_RESTORE_PATH + model + "_" + technique + "_" + mode + "_" + str(
    #                     n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)

    # for model in ["cao"]:
    #     for n_components in range(10, 191, 10):  # IndianPines has 200 bands
    #         for mode in ["supervised"]:
    #             for technique in ["LLE"]:
    #                 my_command = startCommand + " --model " + model + " --band_selection " + technique + " --mode " + mode + " --n_components " + str(
    #                     n_components) + " --autoencoder_epochs " + str(
    #                     10) + " 1> " + BAND_SELECTION_RESTORE_PATH + model + "_" + technique + "_" + mode + "_" + str(
    #                     n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)


    # for model in ["he", "hu", "luo_cnn", "santara", "cao"]:
    #     for n_components in [3]:  # IndianPines has 200 bands
    #         for mode in ["supervised"]:
    #             for technique in ["Autoencoder"]:
    #                 my_command = startCommand + " --model " + model + " --band_selection "+technique+" --mode "+mode+" --n_components "+str(n_components)+" --autoencoder_epochs "+str(10)+" 1> " + BAND_SELECTION_RESTORE_PATH + model +"_"+ technique +"_"+ mode +"_"+ str(n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)

    # for model in ["he", "hu", "luo_cnn", "santara", "cao"]:
    #     for n_components in range(10, 191, 10):  # IndianPines has 200 bands
    #         for mode in ["supervised"]:
    #             for technique in ["UMAP"]:
    #                 my_command = startCommand + " --model " + model + " --band_selection "+technique+" --mode "+mode+" --n_components "+str(n_components)+" --autoencoder_epochs "+str(10)+" 1> " + BAND_SELECTION_RESTORE_PATH + model +"_"+ technique +"_"+ mode +"_"+ str(n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)

    # for model in ["he", "hu", "luo_cnn", "santara", "cao"]:
    #     for n_components in range(10, 191, 10):  # IndianPines has 200 bands
    #         for mode in ["backward_threshold"]:
    #             for technique in ["RandomForest"]:
    #                 my_command = startCommand + " --model " + model + " --band_selection "+technique+" --mode "+mode+" --n_components "+str(n_components)+" --autoencoder_epochs "+str(10)+" 1> " + BAND_SELECTION_RESTORE_PATH + model +"_"+ technique +"_"+ mode +"_"+ str(n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)

    # for model in ["he", "hu", "luo_cnn", "santara", "cao"]:
    #     for n_components in range(10, 191, 10):  # IndianPines has 200 bands
    #         for mode in ["backward_threshold"]:
    #             for technique in ["LogisticRegression"]:
    #                 my_command = startCommand + " --model " + model + " --band_selection "+technique+" --mode "+mode+" --n_components "+str(n_components)+" --autoencoder_epochs "+str(10)+" 1> " + BAND_SELECTION_RESTORE_PATH + model +"_"+ technique +"_"+ mode +"_"+ str(n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)

    # for model in ["he", "hu", "luo_cnn", "santara", "cao"]:
    #     for n_components in range(10, 191, 10):  # IndianPines has 200 bands
    #         for mode in ["backward_threshold"]:
    #             for technique in ["LinearRegression"]:
    #                 my_command = startCommand + " --model " + model + " --band_selection "+technique+" --mode "+mode+" --n_components "+str(n_components)+" --autoencoder_epochs "+str(10)+" 1> " + BAND_SELECTION_RESTORE_PATH + model +"_"+ technique +"_"+ mode +"_"+ str(n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)

    # for model in ["he"]:
    #     for n_components in [90,140,160,190]:  # IndianPines has 200 bands
    #         for mode in ["mixed"]:
    #             for technique in ["LinearRegression"]:
    #                 my_command = startCommand + " --cumulated_band_selection True --model " + model + " --band_selection " + technique + " --mode " + mode + " --n_components " + str(
    #                     n_components) + " --autoencoder_epochs " + str(
    #                     10) + " 1> " + BAND_SELECTION_RESTORE_PATH + model + "_" + technique + "_" + mode + "_" + str(
    #                     n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)

    # for model in ["he"]:
    #     for n_components in range(70,191,10):  # IndianPines has 200 bands
    #         for mode in ["mixed"]:
    #             for technique in ["RandomForest"]:
    #                 my_command = startCommand + " --cumulated_band_selection True --model " + model + " --band_selection " + technique + " --mode " + mode + " --n_components " + str(
    #                     n_components) + " --autoencoder_epochs " + str(
    #                     10) + " 1> " + BAND_SELECTION_RESTORE_PATH + model + "_" + technique + "_" + mode + "_" + str(
    #                     n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)

    # for model in ["luo_cnn"]:
    #     for n_components in [10,20,30,40,80,90,100,110,120,130,160,170,190]:  # IndianPines has 200 bands
    #         for mode in ["mixed"]:
    #             for technique in ["LinearRegression"]:
    #                 my_command = startCommand + " --cumulated_band_selection True --model " + model + " --band_selection " + technique + " --mode " + mode + " --n_components " + str(
    #                     n_components) + " --autoencoder_epochs " + str(
    #                     10) + " 1> " + BAND_SELECTION_RESTORE_PATH + model + "_" + technique + "_" + mode + "_" + str(
    #                     n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)

    # for model in ["luo_cnn"]:
    #     for n_components in [10,20,30,40,70,80,90,100,150,170]:  # IndianPines has 200 bands
    #         for mode in ["mixed"]:
    #             for technique in ["RandomForest"]:
    #                 my_command = startCommand + " --cumulated_band_selection True --model " + model + " --band_selection " + technique + " --mode " + mode + " --n_components " + str(
    #                     n_components) + " --autoencoder_epochs " + str(
    #                     10) + " 1> " + BAND_SELECTION_RESTORE_PATH + model + "_" + technique + "_" + mode + "_" + str(
    #                     n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)

    # for model in ["cao"]:
    #     for n_components in range(17,20,1):  # IndianPines has 200 bands
    #         for mode in ["supervised"]:
    #             for technique in ["PCA"]:
    #                 my_command = startCommand + " --model " + model + " --band_selection " + technique + " --mode " + mode + " --n_components " + str(
    #                     n_components) + " --autoencoder_epochs " + str(
    #                     10) + " 1> " + BAND_SELECTION_RESTORE_PATH + model + "_" + technique + "_" + mode + "_" + str(
    #                     n_components) + ".txt 2>&1"
    #                 # print(my_command)
    #                 os.system(my_command)
    print("test")

def model_distiller_quantize():
    """Perform quantization in Distiller - no model size change"""
    for activations_bits in [4,8,16]:
         for weights_bits in [4,8,16]:
             for accumulator_bits in [4,8,16]:
                 for model in ["he", "hu", "luo_cnn", "santara", "cao"]:
                     startCommand = PYTHON_INTERPRETER_LOCATION + " " + COMPRESS_CLASSIFIER_PATH_LINUX + " --arch " + model + " --dataset IndianPines --epochs 10 -p 1 --lr=0.001 --cuda 0 ../DeepHyperX/Datasets/ --resume-from '" + DEEPHYPERX_PATH_LINUX + "outputs/Experiments/quantizeMe/" + model + "_distiller.pth' --evaluate --quantize-eval --qe-bits-acts " + str(activations_bits) + " --qe-bits-wts " + str(weights_bits) + " --qe-bits-accum " + str(accumulator_bits) + " --name cao_IndianPines_ptquantize --out-dir ../distiller/outputs/post-training-quantization/ --confusion --vs 0.8"
                     os.system(startCommand)
    # for activations_bits in [32]:
    #     for weights_bits in [4,8,16]:
    #         for accumulator_bits in [4,8,16]:
    #             for model in ["he", "hu", "luo_cnn", "santara", "cao"]:
    #                 startCommand = PYTHON_INTERPRETER_LOCATION + " " + COMPRESS_CLASSIFIER_PATH_LINUX + " --arch "+model+" --dataset IndianPines --epochs 10 -p 1 --lr=0.001 --cuda 0 ../DeepHyperX/Datasets/ --resume-from '"+DEEPHYPERX_PATH_LINUX+"outputs/Experiments/quantizeMe/"+model+"_distiller.pth' --evaluate --quantize-eval --qe-bits-acts "+str(activations_bits)+" --qe-bits-wts "+str(weights_bits)+" --qe-bits-accum "+str(accumulator_bits)+" --name cao_IndianPines_ptquantize --out-dir ../distiller/outputs/post-training-quantization/ --confusion --vs 0.8"
    #                 os.system(startCommand)
    # for activations_bits in [4,8,16]:
    #     for weights_bits in [32]:
    #         for accumulator_bits in [4,8,16]:
    #             for model in ["he", "hu", "luo_cnn", "santara", "cao"]:
    #                 startCommand = PYTHON_INTERPRETER_LOCATION + " " + COMPRESS_CLASSIFIER_PATH_LINUX + " --arch "+model+" --dataset IndianPines --epochs 10 -p 1 --lr=0.001 --cuda 0 ../DeepHyperX/Datasets/ --resume-from '"+DEEPHYPERX_PATH_LINUX+"outputs/Experiments/quantizeMe/"+model+"_distiller.pth' --evaluate --quantize-eval --qe-bits-acts "+str(activations_bits)+" --qe-bits-wts "+str(weights_bits)+" --qe-bits-accum "+str(accumulator_bits)+" --name cao_IndianPines_ptquantize --out-dir ../distiller/outputs/post-training-quantization/ --confusion --vs 0.8"
    #                 os.system(startCommand)
    # for activations_bits in [4,8,16]:
    #     for weights_bits in [4,8,16]:
    #         for accumulator_bits in [32]:
    #             for model in ["he", "hu", "luo_cnn", "santara", "cao"]:
    #                 startCommand = PYTHON_INTERPRETER_LOCATION + " " + COMPRESS_CLASSIFIER_PATH_LINUX + " --arch "+model+" --dataset IndianPines --epochs 10 -p 1 --lr=0.001 --cuda 0 ../DeepHyperX/Datasets/ --resume-from '"+DEEPHYPERX_PATH_LINUX+"outputs/Experiments/quantizeMe/"+model+"_distiller.pth' --evaluate --quantize-eval --qe-bits-acts "+str(activations_bits)+" --qe-bits-wts "+str(weights_bits)+" --qe-bits-accum "+str(accumulator_bits)+" --name cao_IndianPines_ptquantize --out-dir ../distiller/outputs/post-training-quantization/ --confusion --vs 0.8"
    #                 os.system(startCommand)

def winmltools_quantize(model, model_path, quantize_bits):
    """find out reduced model sizes of quantized models by quantizing onnx with winmltools. first, pth to onnx, then winmltools for onnx quantization, 8 and 16 bit"""
    from conversion_testfile import pth_to_onnx
    pth_to_onnx(model, model_path)

    model_name = model
    # https://docs.microsoft.com/en-us/windows/ai/windows-ml/convert-model-winmltools#quantize-onnx-model
    # quantize with winmltools
    if quantize_bits == 8: # 8bit INTEGER
        import winmltools
        model = winmltools.load_model(model_name+'.onnx')
        packed_model = winmltools.quantize(model, per_channel=True, nbits=8, use_dequantize_linear=True)
        winmltools.save_model(packed_model, model_name+'_8bit_quantized.onnx')
    elif quantize_bits == 16: # 16bit FLOATING POINT
        from winmltools.utils import convert_float_to_float16
        from winmltools.utils import load_model, save_model
        onnx_model = load_model(model_name+'.onnx')
        new_onnx_model = convert_float_to_float16(onnx_model)
        save_model(new_onnx_model, model_name+'_16bit_quantized.onnx')

    else:
        print("Quantize bits "+str(quantize_bits)+" not supported by winmltools")

def allWinmltoolsQuantize():
    for model in ["he", "hu", "luo_cnn", "santara", "cao"]:
        for bits in [8,16]:
            winmltools_quantize(model, "./outputs/Experiments/pruneMe/"+model+".pth", bits)
            # winmltools_quantize("he", "D:/Experiments/quantizeMe/cao_IndianPines_ptquantize___2019.08.14-231952/he_quantized_checkpoint.pth.tar", bits) # distiller example

def onnx_inference():
    import onnxruntime as rt
    import numpy

    import winmltools
    for model_name in ["he", "hu", "luo_cnn", "santara", "cao"]:
        for sess in [
            rt.InferenceSession("./outputs/Experiments/winmltoolsQuantized/"+model_name+'.onnx'),
            # rt.InferenceSession("./outputs/Experiments/winmltoolsQuantized/"+model_name+'_8bit_quantized.onnx'),
            # rt.InferenceSession("./outputs/Experiments/winmltoolsQuantized/"+model_name+'_16bit_quantized.onnx')
        ]:
            input_name = sess.get_inputs()[0].name
            label_name = sess.get_outputs()[0].name
            # pred = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]


if __name__ == '__main__':
    # onnx_inference()
    # for model in ["he", "hu", "luo_cnn", "santara", "cao"]:
    #     pth_to_onnx(model, "./outputs/Experiments/pruneMe/" + model + ".pth")
    # pth_to_onnx("he", "D:/Experiments/tensorflowQuantize/he.pth")
    # pth_to_onnx("santara", "D:/Experiments/tensorflowQuantize/santara.pth")
    band_prune()