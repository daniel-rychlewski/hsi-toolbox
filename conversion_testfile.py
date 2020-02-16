import time

import onnx
import torch
from datetime import timedelta
from tqdm import tqdm

from DeepHyperX.batch import DEEPHYPERX_PATH_LINUX
from DeepHyperX.custom_models import Cao17


# cao model as example
from DeepHyperX.utils import print_memory_metrics, start_mem_measurement, stop_mem_measurement
from keras_models import cao

expected_input_shapes = {"he":[40,1,200,7,7],
                         "hu":[100, 200],
                         "luo_cnn":[100, 1, 200, 3, 3],
                         "santara":[200,1,200,3,3],
                         "cao": [100,1,200,9,9]}
expected_output_shapes = {"he":[40,17], "cao":[100,17], "santara":[200,17], "luo_cnn":[100,17]}
output_tensors = {"cao":'add_7:0', "hu":'mul_5:0', "he":'add_17:0', "santara":'add_65:0', "luo_cnn":'add_5:0'}

def pth_to_onnx(model, model_path):
    import torch
    # one onnx for each of ["he", "hu", "luo_cnn", "santara", "cao"]
    model_object = None
    input_var = None
    if model == "he":
        from DeepHyperX.models import HeEtAl
        model_object = HeEtAl(200, 17, 7)
        # input_var = torch.FloatTensor(torch.randn((2, 1, 200, 7, 7), requires_grad=True))
        # input_var = torch.FloatTensor(torch.randn((100, 1, 200, 7, 7), requires_grad=True))
    elif model == "hu":
        from DeepHyperX.models import HuEtAl
        model_object = HuEtAl(200, 17)
        # input_var = torch.FloatTensor(torch.randn((2, 200), requires_grad=True))
    elif model == "luo_cnn":
        from DeepHyperX.models import LuoEtAl
        model_object = LuoEtAl(200, 17, patch_size=3)
        # input_var = torch.FloatTensor(torch.randn((2, 1, 200, 3, 3), requires_grad=True))
    elif model == "santara":
        from DeepHyperX.custom_models import Santara16
        model_object = Santara16(n_channels=200, block1_conv1=3330, n_bands=10, patch_size=3, n_classes=17)
        # input_var = torch.FloatTensor(torch.randn((2, 1, 200, 3, 3), requires_grad=True))
        # input_var = torch.FloatTensor(torch.randn((100, 1, 200, 3, 3), requires_grad=True))
    elif model == "cao":
        from DeepHyperX.custom_models import Cao17
        model_object = Cao17(patch_size=9, num_band=200, num_classes=17)
        # input_var = torch.FloatTensor(torch.randn((2, 1, 200, 9, 9), requires_grad=True)) # 100
    input_var = torch.FloatTensor(torch.randn(tuple(expected_input_shapes[model]), requires_grad=True))
    model_object.load_state_dict(torch.load(model_path))
    # model_object.load_state_dict(torch.load(model_path)['state_dict']) # distiller checkpoints
    model_object.train(False)

    from torch.onnx import export
    torch.onnx.export(model_object,  # model being run
                                  (input_var),  # model input (or a tuple for multiple inputs)
                                  model+".onnx",  # where to save the model (can be a file or file-like object)
                                  # input_names=['input'],
                                  # output_names=['output'],
                                  export_params=True)  # store the trained parameter weights inside the model file

def onnx_to_pb(path):
    # path = "D:/Experiments/winmltoolsQuantized/"
    # path = "D:/OneDrive/Dokumente/GitHub/hsi-toolbox/DeepHyperX/"
    import onnx

    # for model_name in ["cao", "luo_cnn", "he", "santara", "hu"]:
    for model_name in ["cao"]:
        model = onnx.load(path+model_name+".onnx")
        # from onnx2keras import onnx_to_keras
        # k_model = onnx_to_keras(model, ['input'])

        from onnx_tf.backend import prepare
        tf_rep = prepare(model)  # prepare tf representation
        tf_rep.export_graph(path+model_name+".pb")  # export the model
        # read pb in keras

# source https://github.com/keras-team/keras/issues/6464
# does not work
def pb_to_hdf5():
    import tensorflow as tf
    from tensorflow.python.platform import gfile
    from keras.applications.resnet50 import ResNet50
    from keras.layers import Dense, GlobalAveragePooling2D, Convolution2D, BatchNormalization
    from keras.models import Model
    from tensorflow.python.framework import tensor_util

    path = "D:/Experiments/winmltoolsQuantized/"

    GRAPH_PB_PATH = path+"cao.pb" #path to your .pb file
    with tf.Session() as sess:
        print("load graph")
        with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            graph_nodes=[n for n in graph_def.node]

            wts = [n for n in graph_nodes if n.op=='Const']

            weight_dict = {}

            for i, n in enumerate(wts):
                weight_dict[n.name] = i


                model = cao(input_shape=(145, 145, 200))
                model.summary()

                for layer in model.layers:
                    layer_weight = layer.get_weights()
                    name = layer.name
                    if len(layer_weight) == 0:
                        continue
                    if isinstance(layer, Convolution2D):
                        kname = name + '/kernel'
                        bname = name + '/bias'
                        if kname not in weight_dict or bname not in weight_dict:
                            print(kname, bname)
                        else:
                            weights = []
                        idx = weight_dict[kname]
                        wtensor = wts[idx].attr['value'].tensor
                        weight = tensor_util.MakeNdarray(wtensor)
                        weights.append(weight)

                        layer.set_weights(weights)
                        continue
                    if isinstance(layer, BatchNormalization):
                        beta_name = name + '/beta'
                        gamma_name = name + '/gamma'
                        mmean_name = name + '/moving_mean'
                        mvar_name = name + '/moving_variance'

                        if beta_name not in weight_dict or gamma_name not in weight_dict or \
                                mmean_name not in weight_dict or mvar_name not in weight_dict:
                            print(beta_name, gamma_name, mmean_name, mvar_name)
                        else:
                            weights = []
                            idx = weight_dict[gamma_name]
                            wtensor = wts[idx].attr['value'].tensor
                            weight = tensor_util.MakeNdarray(wtensor)
                            weights.append(weight)

                            idx = weight_dict[beta_name]
                            wtensor = wts[idx].attr['value'].tensor
                            weight = tensor_util.MakeNdarray(wtensor)
                            weights.append(weight)

                            idx = weight_dict[mmean_name]
                            wtensor = wts[idx].attr['value'].tensor
                            weight = tensor_util.MakeNdarray(wtensor)
                            weights.append(weight)

                            idx = weight_dict[mvar_name]
                            wtensor = wts[idx].attr['value'].tensor
                            weight = tensor_util.MakeNdarray(wtensor)
                            weights.append(weight)
                            layer.set_weights(weights)
                            continue
                    if isinstance(layer, Dense):
                        kname = name + '/kernel'
                        bname = name + '/bias'
                        if kname not in weight_dict or bname not in weight_dict:
                            print(kname, bname)
                        else:
                            weights = []
                            idx = weight_dict[kname]
                            wtensor = wts[idx].attr['value'].tensor
                            weight = tensor_util.MakeNdarray(wtensor)
                            weights.append(weight)

                            idx = weight_dict[bname]
                            wtensor = wts[idx].attr['value'].tensor
                            weight = tensor_util.MakeNdarray(wtensor)
                            weights.append(weight)
                            layer.set_weights(weights)
                            continue

# https://github.com/MarvinTeichmann/KittiSeg/issues/113
def pb_inference(MODEL, path, quantize_afterwards=False):
    import tensorflow as tf  # Default graph is initialized when the library is imported
    import os
    from tensorflow.python.platform import gfile
    from PIL import Image
    import numpy as np
    import scipy
    from scipy import misc
    import matplotlib.pyplot as plt

    DATASET = "IndianPines"
    print("Doing PB inference for model "+MODEL+"...")

    # path = "D:/Experiments/winmltoolsQuantized/"
    GRAPH_PB_PATH = path+MODEL+".pb" #path to your .pb file

    with tf.Graph().as_default() as graph:  # Set default graph as graph

        with tf.Session() as sess:
            # Load the graph in graph_def
            print("load graph")

            # We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
            with gfile.FastGFile(GRAPH_PB_PATH, 'rb') as f:

                # Load IndianPines dataset
                from DeepHyperX.datasets import get_dataset
                img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET,
                                                                                        "./DeepHyperX/Datasets/")

                from DeepHyperX.utils import sample_gt
                _, test_gt = sample_gt(gt, 0.8, mode='random')

                hyperparams = {}
                from DeepHyperX.utils import get_device
                hyperparams.update(
                    {'n_classes': 17, 'n_bands': 200, 'ignored_labels': IGNORED_LABELS, 'device': torch.device("cpu"),#get_device(0),
                     'dataset': "IndianPines"})
                hyperparams['supervision'] = 'full'
                hyperparams['flip_augmentation'] = False
                hyperparams['radiation_augmentation'] = False
                hyperparams['mixture_augmentation'] = False
                hyperparams['center_pixel'] = True

                # model-specific params
                if MODEL == "cao":
                    hyperparams['patch_size'] = 9  # patch_size
                    hyperparams['batch_size'] = 100

                elif MODEL == "hu":
                    hyperparams['patch_size'] = 1  # patch_size
                    hyperparams['batch_size'] = 100
                    # output_tensor = 'mul_5:0'
                elif MODEL == "he":
                    hyperparams['patch_size'] = 7  # patch_size
                    hyperparams['batch_size'] = 40
                    # output_tensor = 'add_17:0'#bs
                    # output_tensor = 'MatMul:0'#bs
                    # output_tensor = 'mul:0'#bs
                elif MODEL == "santara":
                    hyperparams['patch_size'] = 3  # patch_size
                    hyperparams['batch_size'] = 200
                    # output_tensor = 'add_65:0'#bs
                    # output_tensor = 'LogSoftmax:0'
                    # output_tensor = 'MatMul_1:0'#bs
                    # output_tensor = 'transpose_124:0'
                    # output_tensor = 'mul_2:0'#bs
                elif MODEL == "luo_cnn":
                    hyperparams['patch_size'] = 3  # patch_size
                    hyperparams['batch_size'] = 100
                    # output_tensor = 'add_5:0'

                output_tensor = output_tensors[MODEL]

                hyperparams['test_stride'] = 1 # default is 1

                from DeepHyperX.datasets import HyperX
                img_dataset = HyperX(data=img, gt=gt, hyperparams=hyperparams)

                # Set FCN graph to the default graph
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()

                # Import a graph_def into the current default Graph (In this case, the weights are (typically) embedded in the graph)

                tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    return_elements=None,
                    name="",
                    op_dict=None,
                    producer_op_list=None
                )

                # Print the name of operations in the session
                # for op in graph.get_operations():
                #     print("Operation Name :", op.name)  # Operation name
                #     print("Tensor Stats :", str(op.values()))  # Tensor name

                # INFERENCE Here
                l_input = graph.get_tensor_by_name('0:0')  # Input Tensor
                l_output = graph.get_tensor_by_name(output_tensor)  # Output Tensor

                # print("Shape of input : ", tf.shape(l_input))
                # initialize_all_variables
                tf.global_variables_initializer()

                df_column_entry_dict = {}
                print_memory_metrics("before PB Inference", df_column_entry_dict)
                start_mem_measurement()
                start = time.time()

                # get the right input data shape and run model
                probs = test(img_dataset.data, hyperparams, sess, l_output, l_input, MODEL)

                time_elapse = time.time() - start
                event = 'PB Inference'
                formatted_time = str(timedelta(seconds=time_elapse))
                df_column_entry_dict['Time measurement at ' + event + ' [s]'] = time_elapse
                print("\n" + event + " took " + formatted_time + " seconds\n")
                event = "after PB Inference"
                stop_mem_measurement(event, df_column_entry_dict)
                print_memory_metrics(event, df_column_entry_dict)

                prediction = np.argmax(probs, axis=-1)

                # goal: display accuracy metrics, incl. confusion matrix
                from DeepHyperX.utils import metrics
                run_results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'],
                                      n_classes=hyperparams['n_classes'])

                mask = np.zeros(gt.shape, dtype='bool')
                for l in IGNORED_LABELS:
                    mask[gt == l] = True
                prediction[mask] = 0

                results = []
                results.append(run_results)
                from DeepHyperX.utils import show_results
                import visdom
                viz = visdom.Visdom(env=MODEL + "_"+ DATASET)

                dataframe_grid = []
                show_results(run_results, viz, label_values=LABEL_VALUES, df_column_entry_dict=df_column_entry_dict)

                dataframe_grid.append(list(df_column_entry_dict.values()))
                import pandas as pd
                frame = pd.DataFrame(dataframe_grid, columns=list(df_column_entry_dict.keys()))
                means = frame.mean()
                frame = frame.append(means, ignore_index=True)

                from DeepHyperX.batch import STORE_EXPERIMENT_LOCATION
                frame.to_excel(path + MODEL + "_" + DATASET + ".xlsx", index=False)

            if quantize_afterwards:
                print("Quantizing model "+MODEL+" after inference...\n")

                img = tf.identity(tf.get_variable(name="0", dtype=tf.float32, shape=tuple(expected_input_shapes[MODEL])), name="0")
                # img = tf.identity(tf.get_variable(name="Const_53", dtype=tf.float32, shape=tuple(expected_input_shapes[MODEL])), name="Const_53")
                # img = tf.identity(tf.get_variable(name="foo", dtype=tf.float32, shape=tuple(expected_input_shapes[MODEL])), name="0")
                out = tf.identity(tf.get_variable(name=output_tensors[MODEL][:len(output_tensors[MODEL])-2], dtype=tf.float32, shape=tuple(expected_output_shapes[MODEL])), name=output_tensors[MODEL][:len(output_tensors[MODEL])-2]) # cut out ":0" for valid tensor name
                # out = tf.identity(tf.get_variable(name="bar", dtype=tf.float32, shape=tuple(expected_output_shapes[MODEL])), name=output_tensors[MODEL][:len(output_tensors[MODEL])-2]) # cut out ":0" for valid tensor name

                sess.run(tf.global_variables_initializer())
                converter = tf.lite.TFLiteConverter.from_session(sess, [img], [out])
                tflite_model = converter.convert()
                open("converted_model.tflite", "wb").write(tflite_model)

                """
                2019-08-20 08:23:33.350952: F tensorflow/lite/toco/tooling_util.cc:897] Check failed: GetOpWithInput(model, input_array.name()) Specified input array "0_2" is not consumed by any op in this graph. Is it a typo? To silence this message, pass this flag:  allow_nonexistent_arrays
                """

def test(img, hyperparams, sess, l_output, l_input, model):
        """
        Test a model on a specific image
        """
        import numpy as np

        patch_size = hyperparams['patch_size']
        center_pixel = hyperparams['center_pixel']
        batch_size, device = hyperparams['batch_size'], hyperparams['device']
        n_classes = hyperparams['n_classes']

        kwargs = {'step': hyperparams['test_stride'], 'window_size': (patch_size, patch_size)}
        probs = np.zeros(img.shape[:2] + (n_classes,))

        from DeepHyperX.utils import count_sliding_window
        iterations = count_sliding_window(img, **kwargs) // batch_size
        from DeepHyperX.utils import grouper
        from DeepHyperX.utils import sliding_window
        for batch in tqdm(grouper(batch_size, sliding_window(img, **kwargs)),
                          total=(iterations),
                          desc="Inference on the image"
                          ):
            with torch.no_grad():
                if patch_size == 1:
                    data = [b[0][0, 0] for b in batch]
                    data = np.copy(data)
                    data = torch.from_numpy(data)
                else:
                    data = [b[0] for b in batch]
                    data = np.copy(data)
                    data = data.transpose(0, 3, 1, 2)
                    data = torch.from_numpy(data)
                    data = data.unsqueeze(1)

                indices = [b[1:] for b in batch]
                data = data.to(device).float()

                data = data.cpu() # otherwise "TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first."
                # data = data.reshape(shape=(-1, hyperparams['n_bands'], patch_size, patch_size))
                if data.shape == torch.Size(expected_input_shapes[model]): # NOT e.g. 96,1,200,9,9 for last batch
                    output = sess.run(l_output, feed_dict={l_input: data})
                    if isinstance(output, tuple):
                        output = output[0]

                    if not (patch_size == 1 or center_pixel):
                        output = np.transpose(output.numpy(), (0, 2, 3, 1))
                    for (x, y, w, h), out in zip(indices, output):
                        if center_pixel:
                            probs[x + w // 2, y + h // 2] += out
                        else:
                            probs[x:x + w, y:y + h] += out
        return probs

def quantize_pb(model, path):
    import tensorflow as tf
    MODEL=model

    # converter = tf.lite.TFLiteConverter.from_saved_model("cao.pb")

    img = tf.identity(tf.get_variable(name="0", dtype=tf.float32, shape=tuple(expected_input_shapes[MODEL])), name="0")
    out = tf.identity(tf.get_variable(name=output_tensors[MODEL][:len(output_tensors[MODEL])-2], dtype=tf.float32, shape=tuple(expected_output_shapes[MODEL])), name=output_tensors[MODEL][:len(output_tensors[MODEL])-2]) # cut out ":0" for valid tensor name

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     saver = tf.train.Saver()
    #     saver.restore(sess=sess, save_path=path+model+".pb")
    #     save_path= saver.save(sess=sess, save_path=path+"eval_checkpoint_"+model+".pb")

    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        path+model+".pb", ["0"], [output_tensors[MODEL][:len(output_tensors[MODEL])-2]])
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)


if __name__ == "__main__":
    # pth_to_onnx("hu", "D:/Experiments/pruneMe/hu.pth") # produces onnx warning
    pth_to_onnx("cao", "./DeepHyperX/cao_test.pth")
    # pth_to_onnx("he", "D:/OneDrive/Dokumente/GitHub/hsi-toolbox/DeepHyperX/he.pth")
    # pth_to_onnx("hu", "D:/OneDrive/Dokumente/GitHub/hsi-toolbox/DeepHyperX/hu.pth")
    # pth_to_onnx("santara", "D:/OneDrive/Dokumente/GitHub/hsi-toolbox/DeepHyperX/santara.pth")
    # pth_to_onnx("luo_cnn", "D:/OneDrive/Dokumente/GitHub/hsi-toolbox/DeepHyperX/luo_cnn.pth")
    # onnx_to_pb("./")
    # onnx_to_pb(DEEPHYPERX_PATH_LINUX+"outputs/Experiments/winmltoolsQuantized/")
    pb_inference("cao","./",quantize_afterwards=True)

    for model in ["cao", "santara", "luo_cnn", "he"]:
    # for model in ["cao"]:
        pb_inference(MODEL=model, path=DEEPHYPERX_PATH_LINUX + "outputs/Experiments/winmltoolsQuantized/", quantize_afterwards=False)
    # quantize_pb("cao",path="D:/Experiments/tensorflowQuantize/")
    # for model in ["he", "santara"]:
    # path = "D:/Experiments/winmltoolsQuantized/"
    # model1 = onnx.load(path + "cao_8bit_quantized" + ".onnx")
    # model2 = onnx.load(path + "cao_16bit_quantized" + ".onnx")
    # model3 = onnx.load(path + "hu" + ".onnx")
    # model4 = onnx.load(path + "santara" + ".onnx")
    # model5 = onnx.load(path + "luo_cnn_8bit_quantized" + ".onnx")
    # model7 = onnx.load(path + "luo_cnn_16bit_quantized" + ".onnx")
    # print("bla")