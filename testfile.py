from torch.autograd import Variable

import DeepHyperX.models
from DeepHyperX.datasets import get_dataset
from DeepHyperX.models import *
from conversion_testfile import expected_input_shapes


def compareModels():
    m1 = HamidaEtAl(input_channels=200, n_classes=17)
    print(m1.state_dict())
    print("model2begin\n")
    m2 = HamidaEtAl(input_channels=200, n_classes=17)
    print(m2.state_dict())
    exit()

x=torch.rand((256,18,198,3,3))

import torch

import hiddenlayer as h1
for model in ["hu"]:#,"hu","luo_cnn","santara","cao"
    # one onnx for each of ["he", "hu", "luo_cnn", "santara", "cao"]
    model_object = None
    input_var = None
    if model == "he":
        from DeepHyperX.models import HeEtAl

        model_object = HeEtAl(200, 17, 7)
        # input_var = torch.FloatTensor(torch.randn((2, 1, 200, 7, 7), requires_grad=True))
        # input_var = torch.FloatTensor(torch.randn((100, 1, 200, 7, 7), requires_grad=True))
        transforms=[
            # h1.transforms.Fold("Addx3 > Relu", "AddRelu"),
            h1.transforms.Prune("Squeeze"),
            h1.transforms.Prune("Unsqueeze"),
            h1.transforms.Prune("Reshape"),
        ]
    elif model == "hu":
        from DeepHyperX.models import HuEtAl

        model_object = HuEtAl(200, 17)
        transforms=[
            h1.transforms.Fold("Squeeze > Unsqueeze > Conv23", "SqueezeUnsqueezeConv23")
        ]
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

    h1_graph = h1.build_graph(model_object, torch.zeros(expected_input_shapes[model]), transforms=transforms)
    h1_graph.theme = h1.graph.THEMES["blue"].copy()
    # model_object.load_state_dict(torch.load(model_path))
    # model_object.train(False)
    h1_graph.save(path="./"+model+"_with_transformations", format="png")

print("ok")
# read pth file

# compareModels()
# exporting from PyTorch into ONNX, and then load the ONNX proto representation of the model into Glow





# read pb in keras

# what can i do with pb file

# quantize:
import tensorflow as tf
# converter = tf.lite.TFLiteConverter.from_saved_model("cao.pb")
converter = tf.lite.TFLiteConverter.from_frozen_graph(
      "cao.pb", ['input'], ['output'])
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()

# print name of nodes
import tensorflow as tf
from tensorflow.python.platform import gfile
GRAPH_PB_PATH = './output_path.pb'
with tf.Session() as sess:
   print("load graph")
   with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
       graph_def = tf.GraphDef()
   graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   tf.import_graph_def(graph_def, name='')
   graph_nodes=[n for n in graph_def.node]
   names = []
   for t in graph_nodes:
      names.append(t.name)
   print(names)

img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset("IndianPines",
                                                               "./Datasets/")

#inference
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

name = "output_path.pb"

with tf.Session() as persisted_sess:
    print("load graph")
    with gfile.FastGFile(name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    persisted_sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    test = np.random.rand(1, 1, 28, 28).astype(np.float32)

    inp = persisted_sess.graph.get_tensor_by_name('input')
    out = persisted_sess.graph.get_tensor_by_name('output')
    feed_dict = {inp: test}

    classification = persisted_sess.run(out, feed_dict)

file0 = torch.load("./DeepHyperX/outputs/Experiments/pruneMe/cao.pth")

#quantize with winmltools for 8 bit only
import winmltools
model = winmltools.load_model('cao.onnx')
packed_model = winmltools.quantize(model, per_channel=True, nbits=8, use_dequantize_linear=True)
winmltools.save_model(packed_model, 'cao_quantized.onnx')

file1 = torch.load("./DeepHyperX/outputs/Experiments/pruneMe/he.pth")
file2 = torch.load("./DeepHyperX/outputs/Experiments/pruneMe/hu.pth")
file3 = torch.load("./DeepHyperX/outputs/Experiments/pruneMe/luo_cnn.pth")
file4 = torch.load("./DeepHyperX/outputs/Experiments/pruneMe/santara.pth")
file5 = torch.load("./DeepHyperX/outputs/Experiments/allDatasetsAllModels6Runs/hamida_et_al/IndianPines/2019-07-01 17-09-24.344425_epoch100_0.92.pth", map_location='cpu')
file1 = torch.load("/mnt/hgfs/hsi-toolbox/distiller/outputs/post-training-quantization/hamida_IndianPines_ptquantize___2019.07.06-113108/hamida_quantized_checkpoint.pth.tar", map_location='cpu')
file2 = torch.load("/mnt/hgfs/hsi-toolbox/examples/ssl/checkpoints/checkpoint_trained_dense.pth.tar", map_location='cpu')
file3 = torch.load("/mnt/hgfs/hsi-toolbox/distiller/outputs/post-training-quantization/hamida_IndianPines_ptquantize___2019.07.06-134632/hamida_quantized_checkpoint.pth.tar", map_location='cpu')
mouorig_gpu02 = torch.load("/mnt/hgfs/hsi-toolbox/DeepHyperX/outputs/Experiments/allDatasetsAllModels6Runs/mou_et_al/IndianPines/2019-05-28 01-59-22.761879_epoch100_0.83.pth", map_location='cpu')
mouquantizedtochecknnGRU = torch.load("/mnt/hgfs/hsi-toolbox/distiller/outputs/post-training-quantization/hamida_IndianPines_ptquantize___2019.07.06-143542/mou_quantized_checkpoint.pth.tar", map_location='cpu')

pruning1 = torch.load("/mnt/hgfs/hsi-toolbox/distiller/outputs/post-training-quantization/hamida_IndianPines_prune___2019.07.07-151115/greedy__001__100.0__93.7_checkpoint.pth.tar", map_location='cpu')
pruning2 = torch.load("/mnt/hgfs/hsi-toolbox/distiller/outputs/post-training-quantization/hamida_IndianPines_prune___2019.07.07-151115/greedy__003__100.0__93.7_checkpoint.pth.tar", map_location='cpu')
pruning3 = torch.load("/mnt/hgfs/hsi-toolbox/distiller/outputs/post-training-quantization/hamida_IndianPines_prune___2019.07.07-151115/greedy__017__100.0__94.2_checkpoint.pth.tar", map_location='cpu')
print("file read")
model1 = DeepHyperX.models.HamidaEtAl(input_channels=200, n_classes=17)
model1.load_state_dict(file1['state_dict'])
model1.eval()
print("parameters\n")
for child in model1.children():
    for parameter in child.parameters():

        model1.state_dict()['features.0.weight'].copy_(torch.zeros(size=parameter.shape))

        print("debug me")
torch.save(model1, './DeepHyperX/checkpoints/hamida_et_al/IndianPines/2019-05-04 00-48-21.727493_epoch100_0.87_edited.pth')

model2 = torch.load('./DeepHyperX/checkpoints/hamida_et_al/IndianPines/2019-05-04 00-48-21.727493_epoch100_0.87_edited.pth', map_location='cpu')
# device = torch.device('cpu')
print("parameters\n")
for child in model2.children():
    for parameter in child.parameters():
        print("debug me")
# for parameter in model.parameters():
#     print(parameter+"\n")
# print("modules\n")
# for module in model.modules():
#     print(module+"\n")

