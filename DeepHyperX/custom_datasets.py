from deprecated import deprecated

from DeepHyperX.utils import open_file
import numpy as np

CUSTOM_DATASETS_CONFIG = {
    # DFC 2018 HSI dataset is nowhere to be found for download, so skip this entirely
     'DFC2018_HSI': {
        'img': '2018_IEEE_GRSS_DFC_HSI_TR.HDR',
        'gt': '2018_IEEE_GRSS_DFC_GT_TR.tif',
        'download': False,
        'loader': lambda folder: dfc2018_loader(folder)
    },
    'Salinas': {
        'img': 'Salinas_corrected.mat',
        'gt': 'Salinas_gt.mat',
        'download': False,
        'loader': lambda folder: salinas_loader(folder)
    },
    'SalinasA': {
        'img': 'SalinasA_corrected.mat',
        'gt': 'SalinasA_gt.mat',
        'download': False,
        'loader': lambda folder: salinas_a_loader(folder)
    },
    # https://rslab.ut.ac.ir/data
    # Cuprite: After removing the noisy channels (1-2 and 221-224) and water absorption channels (104-113 and 148-167), 188 channels remain.
    'Cuprite-224': {
        'img': 'CupriteS1_F224.mat',
        'gt': 'groundTruth_Cuprite_nEnd12.mat',
        'download': False,
        'loader': lambda folder: cuprite_224_loader(folder)
    },
    'Cuprite-188': {
        'img': 'CupriteS1_R188.mat',
        'gt': 'groundTruth_Cuprite_nEnd12.mat',
        'download': False,
        'loader': lambda folder: cuprite_188_loader(folder)
    },
    'Samson': {
        'img': 'samson_1.mat',
        'gt': 'end3_gt.mat',
        'download': False,
        'loader': lambda folder: samson_loader(folder)
    },
    'JasperRidge-198': {
        'img': 'jasperRidge2_R198.mat',
        'gt': 'end4.mat',
        'download': False,
        'loader': lambda folder: jasper_ridge_198_loader(folder)
    },
    'JasperRidge-224': {
        'img': 'jasperRidge2_F224_2.mat',
        'gt': 'end4.mat',
        'download': False,
        'loader': lambda folder: jasper_ridge_224_loader(folder)
    },
    'Urban-162': {
        'img': 'Urban_R162.mat',
        'gt': 'end6_groundTruth.mat',
        'download': False,
        'loader': lambda folder: urban_162_loader(folder)
    },
    'Urban-210': {
        'img': 'Urban_F210.mat',
        'gt': 'end6_groundTruth.mat',
        'download': False,
        'loader': lambda folder: urban_210_loader(folder)
    },
    'China': {
        'img': 'China_Change_Dataset.mat',
        'download': False,
        'loader': lambda folder: china_loader(folder)
    },
    'USA': {
        'img': 'USA_Change_Dataset.mat',
        'download': False,
        'loader': lambda folder: usa_loader(folder)
    },
    'Washington': {
        'img': 'DC.tif',
        'gt': 'GT.tif',
        'download': False,
        'loader': lambda folder: washington_loader(folder)
    }
}

@deprecated(reason="the dataset is nowhere to be found / downloaded, please use a different dataset")
def dfc2018_loader(folder):
        img = open_file(folder + CUSTOM_DATASETS_CONFIG['DFC2018_HSI']['img'])[:,:,:-2]
        gt = open_file(folder + CUSTOM_DATASETS_CONFIG['DFC2018_HSI']['gt'])
        gt = gt.astype('uint8')

        rgb_bands = (47, 31, 15)

        label_values = ["Unclassified",
                        "Healthy grass",
                        "Stressed grass",
                        "Artificial turf",
                        "Evergreen trees",
                        "Deciduous trees",
                        "Bare earth",
                        "Water",
                        "Residential buildings",
                        "Non-residential buildings",
                        "Roads",
                        "Sidewalks",
                        "Crosswalks",
                        "Major thoroughfares",
                        "Highways",
                        "Railways",
                        "Paved parking lots",
                        "Unpaved parking lots",
                        "Cars",
                        "Trains",
                        "Stadium seats"]
        ignored_labels = [0]
        palette = None

        return img, gt, rgb_bands, ignored_labels, label_values, palette

def salinas_loader(folder):
    img = open_file(folder + CUSTOM_DATASETS_CONFIG['Salinas']['img'])['salinas_corrected']
    gt = open_file(folder + CUSTOM_DATASETS_CONFIG['Salinas']['gt'])['salinas_gt']
    gt = gt.astype('uint8')

    rgb_bands = (47, 27, 13)

    label_values = ["Unclassified",
                    "Brocoli_green_weeds_1",
                    "Brocoli_green_weeds_2",
                    "Fallow",
                    "Fallow_rough_plow",
                    "Fallow_smooth",
                    "Stubble",
                    "Celery",
                    "Grapes_untrained",
                    "Soil_vinyard_develop",
                    "Corn_senesced_green_weeds",
                    "Lettuce_romaine_4wk",
                    "Lettuce_romaine_5wk",
                    "Lettuce_romaine_6wk",
                    "Lettuce_romaine_7wk",
                    "Vinyard_untrained",
                    "Vinyard_vertical_trellis"]

    ignored_labels = []
    palette = None

    return img, gt, rgb_bands, ignored_labels, label_values, palette

def salinas_a_loader(folder):
    img = open_file(folder + CUSTOM_DATASETS_CONFIG['SalinasA']['img'])['salinasA_corrected']
    gt = open_file(folder + CUSTOM_DATASETS_CONFIG['SalinasA']['gt'])['salinasA_gt']
    gt = gt.astype('uint8')

    # remap for contiguous integers to avoid index out of bounds
    salinas_a_remap = {0: 0, 1: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}
    for k, v in salinas_a_remap.items():
        gt = np.where(gt == k, v, gt)

    rgb_bands = (47, 27, 13)

    label_values = ["Unclassified",
                    "Brocoli_green_weeds_1",
                    "Corn_senesced_green_weeds",
                    "Lettuce_romaine_4wk",
                    "Lettuce_romaine_5wk",
                    "Lettuce_romaine_6wk",
                    "Lettuce_romaine_7wk"]

    ignored_labels = []
    palette = None

    return img, gt, rgb_bands, ignored_labels, label_values, palette

@deprecated("gt is incomplete")
def cuprite_224_loader(folder):
    img = open_file(folder + CUSTOM_DATASETS_CONFIG['Cuprite-224']['img'])['Y']
    img = np.reshape(img, (250, 190, 224)) # only includes GT: endmembers.
    gt = np.asarray(np.matrix(open_file(folder + CUSTOM_DATASETS_CONFIG['Cuprite-224']['gt'])['M'].argmax(1)))
    gt = np.transpose(gt)
    gt = np.reshape(gt, (224))

    rgb_bands = (183, 193, 203) # not sure but does not matter

    label_values = ["Alunite",
                    "Andradite",
                    "Buddingtonite",
                    "Dumortierite",
                    "Kaolinite1",
                    "Kaolinite2",
                    "Muscovite",
                    "Montmorillonite",
                    "Nontronite",
                    "Pyrope",
                    "Sphene",
                    "Chalcedony"]

    ignored_labels = []
    palette = None

    return img, gt, rgb_bands, ignored_labels, label_values, palette

@deprecated("gt is incomplete")
def cuprite_188_loader(folder):
    img = open_file(folder + CUSTOM_DATASETS_CONFIG['Cuprite-188']['img'])['Y']
    img = np.reshape(img, (250, 190, 188))  # only includes GT: endmembers.
    gt = np.asarray(np.matrix(open_file(folder + CUSTOM_DATASETS_CONFIG['Cuprite-188']['gt'])['M'].argmax(1)))
    gt = np.transpose(gt)
    gt = np.reshape(gt, (188))

    rgb_bands = (183, 193, 203) # not sure but does not matter

    label_values = ["Alunite",
                    "Andradite",
                    "Buddingtonite",
                    "Dumortierite",
                    "Kaolinite1",
                    "Kaolinite2",
                    "Muscovite",
                    "Montmorillonite",
                    "Nontronite",
                    "Pyrope",
                    "Sphene",
                    "Chalcedony"]

    ignored_labels = []
    palette = None

    return img, gt, rgb_bands, ignored_labels, label_values, palette

def samson_loader(folder):
    img = open_file(folder + CUSTOM_DATASETS_CONFIG['Samson']['img'])['V']
    img = np.reshape(img, (95,95,156))
    gt = np.asarray(np.matrix(open_file(folder + CUSTOM_DATASETS_CONFIG['Samson']['gt'])['A']).argmax(0))
    gt = np.reshape(gt, (95, 95))

    rgb_bands = (9, 44, 54) # manually calculated, assuming linear distribution of bands among wavelengths

    label_values = ["Rock",
                    "Tree",
                    "Water"]

    ignored_labels = []
    palette = None

    return img, gt, rgb_bands, ignored_labels, label_values, palette

def jasper_ridge_198_loader(folder):
    img = open_file(folder + CUSTOM_DATASETS_CONFIG['JasperRidge-198']['img'])['Y']
    img = np.reshape(img, (100, 100, 198))
    gt = np.asarray(np.matrix(open_file(folder + CUSTOM_DATASETS_CONFIG['JasperRidge-198']['gt'])['A'].argmax(0)))
    gt = np.reshape(gt, (100, 100))

    rgb_bands = (5, 15, 18) # manually calculated, assuming linear distribution of bands among wavelengths

    label_values = ["Road",
                    "Soil",
                    "Water",
                    "Tree"]

    ignored_labels = []
    palette = None

    return img, gt, rgb_bands, ignored_labels, label_values, palette

def jasper_ridge_224_loader(folder):
    img = open_file(folder + CUSTOM_DATASETS_CONFIG['JasperRidge-224']['img'])['Y']
    img = np.reshape(img, (100, 100, 224))
    gt = np.asarray(np.matrix(open_file(folder + CUSTOM_DATASETS_CONFIG['JasperRidge-224']['gt'])['A'].argmax(0)))
    gt = np.reshape(gt, (100, 100))

    rgb_bands = (5, 16, 20) # manually calculated, assuming linear distribution of bands among wavelengths

    label_values = ["Road",
                    "Soil",
                    "Water",
                    "Tree"]

    ignored_labels = []
    palette = None

    return img, gt, rgb_bands, ignored_labels, label_values, palette

def urban_162_loader(folder):
    img = open_file(folder + CUSTOM_DATASETS_CONFIG['Urban-162']['img'])['Y']
    img = np.reshape(img, (307, 307, 162))
    gt = np.asarray(np.matrix(open_file(folder + CUSTOM_DATASETS_CONFIG['Urban-162']['gt'])['A'].argmax(0)))
    gt = np.reshape(gt, (307, 307))

    rgb_bands = (13, 11, 2) # manually calculated, assuming linear distribution of bands among wavelengths

    label_values = ["Asphalt",
                    "Grass",
                    "Tree",
                    "Roof",
                    "Metal",
                    "Dirt"]

    ignored_labels = []
    palette = None

    return img, gt, rgb_bands, ignored_labels, label_values, palette

def urban_210_loader(folder):
    img = open_file(folder + CUSTOM_DATASETS_CONFIG['Urban-210']['img'])['Y']
    img = np.reshape(img, (307, 307, 210))
    gt = np.asarray(np.matrix(open_file(folder + CUSTOM_DATASETS_CONFIG['Urban-210']['gt'])['A'].argmax(0)))
    gt = np.reshape(gt, (307, 307))

    rgb_bands = (17, 14, 3) # manually calculated, assuming linear distribution of bands among wavelengths

    label_values = ["Asphalt",
                    "Grass",
                    "Tree",
                    "Roof",
                    "Metal",
                    "Dirt"]

    ignored_labels = []
    palette = None

    return img, gt, rgb_bands, ignored_labels, label_values, palette

""" Please cite using these datasets as below
Published in: International Journal of Remote Sensing, vol. ?, no. ?, p. ?, April. 2018.
Title: "Hyperspectral Change Detection: An Experimental Comparative Study
https://doi.org/10.1080/01431161.2018.1466079
Authors: M. Hasanlou and S. T. Seyedi """
def china_loader(folder):
    img = open_file(folder + CUSTOM_DATASETS_CONFIG['China']['img'])['T2']
    gt_preprocess_me = open_file(folder + CUSTOM_DATASETS_CONFIG['China']['img'])['Multiple']

    # need to preprocess gt
    gt = np.zeros(shape=(gt_preprocess_me.shape[0], gt_preprocess_me.shape[1]))

    for i in range(gt_preprocess_me.shape[0]):
        for j in range(gt_preprocess_me.shape[1]):
            if (gt_preprocess_me[i][j] == [254,0,0]).all():
                gt[i][j] = 0
            elif (gt_preprocess_me[i][j] == [0,254,0]).all():
                gt[i][j] = 1
            elif (gt_preprocess_me[i][j] == [0,0,254]).all():
                gt[i][j] = 2
            elif (gt_preprocess_me[i][j] == [254,254,0]).all():
                gt[i][j] = 3
            # from here on, unused elifs
            elif (gt_preprocess_me[i][j] == [254,0,254]).all():
                gt[i][j] = 4
            elif (gt_preprocess_me[i][j] == [0,254,254]).all():
                gt[i][j] = 5
            elif (gt_preprocess_me[i][j] == [0, 0, 0]).all():
                gt[i][j] = 6
            elif (gt_preprocess_me[i][j] == [254, 254, 254]).all():
                gt[i][j] = 7

    gt = gt.astype(dtype='uint8')
    rgb_bands = (0,0,0) # not given

    label_values = ["soil",
                    "river",
                    "tree",
                    "building",
                    "road",
                    "agricultural field"]

    ignored_labels = []
    palette = None

    return img, gt, rgb_bands, ignored_labels, label_values, palette

""" Please cite using these datasets as below
Published in: International Journal of Remote Sensing, vol. ?, no. ?, p. ?, April. 2018.
Title: "Hyperspectral Change Detection: An Experimental Comparative Study
https://doi.org/10.1080/01431161.2018.1466079
Authors: M. Hasanlou and S. T. Seyedi """
def usa_loader(folder):
    img = open_file(folder + CUSTOM_DATASETS_CONFIG['USA']['img'])['T2']
    gt_preprocess_me = open_file(folder + CUSTOM_DATASETS_CONFIG['USA']['img'])['Multiple']

    # need to preprocess gt
    gt = np.zeros(shape=(gt_preprocess_me.shape[0], gt_preprocess_me.shape[1]))

    for i in range(gt_preprocess_me.shape[0]):
        for j in range(gt_preprocess_me.shape[1]):
            if (gt_preprocess_me[i][j] == [255,0,0]).all():
                gt[i][j] = 0
            elif (gt_preprocess_me[i][j] == [0,255,0]).all():
                gt[i][j] = 1
            elif (gt_preprocess_me[i][j] == [0,0,255]).all():
                gt[i][j] = 2
            elif (gt_preprocess_me[i][j] == [255,255,0]).all():
                gt[i][j] = 3
            elif (gt_preprocess_me[i][j] == [255,0,255]).all():
                gt[i][j] = 4
            elif (gt_preprocess_me[i][j] == [0,255,255]).all():
                gt[i][j] = 5
            # from here on, unused elifs
            elif (gt_preprocess_me[i][j] == [0, 0, 0]).all():
                gt[i][j] = 6
            elif (gt_preprocess_me[i][j] == [255, 255, 255]).all():
                gt[i][j] = 7

    gt = gt.astype(dtype='uint8')
    rgb_bands = (0,0,0) # not given

    label_values = ["soil",
                    "irrigated fields",
                    "river",
                    "building",
                    "type of cultivated land",
                    "grassland"]

    ignored_labels = []
    palette = None

    return img, gt, rgb_bands, ignored_labels, label_values, palette

@deprecated("cannot identify image file")
def washington_loader(folder):
    img = open_file(folder + CUSTOM_DATASETS_CONFIG['Washington']['img'])
    gt = open_file(folder + CUSTOM_DATASETS_CONFIG['Washington']['gt'])

    rgb_bands = (60,27,17)

    # http://sugs.u-strasbg.fr/omiv/imagemining/documents/IMAGEMINING-DallaMurra-practicals.pdf
    label_values = ["Roofs",
                    "Street",
                    "Path",
                    "Grass",
                    "Trees",
                    "Water",
                    "Shadow"]

    ignored_labels = []
    palette = None

    return img, gt, rgb_bands, ignored_labels, label_values, palette