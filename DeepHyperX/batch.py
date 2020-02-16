# Finetune a parameter by trying out different values (one at a time)
import json
import os

import math

USER_DIRECTORY_LINUX = "/insert/your/user/dir/"

PYTHON_INTERPRETER_LOCATION_GPU02 = "/home/gpu02/anaconda3/envs/hsi36/bin/python"
PYTHON_INTERPRETER_LOCATION_WINDOWS = "C:/Users/danie/.conda/envs/hsi36/python.exe"
PYTHON_INTERPRETER_LOCATION = PYTHON_INTERPRETER_LOCATION_WINDOWS

HSI_TOOLBOX_PATH_LINUX = USER_DIRECTORY_LINUX + "PycharmProjects/hsi-toolbox/"
COMPRESS_CLASSIFIER_PATH_LINUX = HSI_TOOLBOX_PATH_LINUX + "examples/classifier_compression/compress_classifier.py"

DEEPHYPERX_PATH_LINUX_GPU02 = "/home/gpu02/PycharmProjects/hsi-toolbox/DeepHyperX/"
DEEPHYPERX_PATH_LINUX = HSI_TOOLBOX_PATH_LINUX + "DeepHyperX/"
DEEPHYPERX_PATH_WINDOWS = "D:/hsi-toolbox/DeepHyperX/"
DEEPHYPERX_PATH = DEEPHYPERX_PATH_WINDOWS

DISTILLER_PATH_LINUX_GPU02 = "/home/gpu02/PycharmProjects/hsi-toolbox/distiller/"
DISTILLER_PATH_LINUX_VM = "/mnt/hgfs/hsi-toolbox/distiller/"
DISTILLER_PATH = DISTILLER_PATH_LINUX_VM

PARAMETER_JSON = DEEPHYPERX_PATH + "Parameters.json"
MAIN_PY_LOCATION = DEEPHYPERX_PATH + "main.py"
VIS_PY_LOCATION = DEEPHYPERX_PATH + "vis.py"

STORE_EXPERIMENT_SUFFIX = "outputs/Experiments/allDatasetsAllModels6Runs/"
STORE_EXPERIMENT_LOCATION = DEEPHYPERX_PATH + STORE_EXPERIMENT_SUFFIX
DISTILLER_STORE_LOCATION = DISTILLER_PATH + "outputs/"
BAND_SELECTION_CHECKPOINTS = DISTILLER_STORE_LOCATION + "bandSelectionCheckpoints/"

VISDOM_PATH_WINDOWS = "C:/Users/danie/.conda/envs/hsi36/Scripts/visdom.exe"
VISDOM_PATH_LINUX = USER_DIRECTORY_LINUX+".conda/envs/hsi36/bin/visdom"
VISDOM_PATH = VISDOM_PATH_WINDOWS

# https://stackoverflow.com/questions/28425705/python-rounding-a-floating-point-number-to-nearest-0-05
def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))

def main():
    import subprocess
    subprocess.Popen([VISDOM_PATH], close_fds=True)

    try_all_datasets = True
    try_all_models = True
    redirect_to_output = True
    do_vary = False
    vary = "training_sample"
    output_path = STORE_EXPERIMENT_LOCATION

    parameterFile = open(PARAMETER_JSON, "r")
    data = json.load(parameterFile)
    parameterFile.close()

    startCommand = PYTHON_INTERPRETER_LOCATION+" "+MAIN_PY_LOCATION+" --training_sample 0.8 --runs 6 --with_exploration --cuda 0"

    datasets = data["datasets"]
    models = data["models"]
    limit = float(data[vary]["max"])
    increment = float(data[vary]["step"])

    if not do_vary:
        if try_all_datasets:
            if try_all_models:
                # Both variations
                for dataset in datasets:
                    for model in models:
                        # Adjust the command
                        adjustedCommand = startCommand + " --dataset " + dataset + " --model " + model + " 1> " + output_path + model + "_" + dataset + ".txt 2>&1"
                        # Execute the command
                        os.system(adjustedCommand)
                        scp_checkpoint_and_log_backup()
            else:
                # Dataset variation
                for dataset in datasets:
                    # Adjust the command
                    adjustedCommand = startCommand + " --dataset " + dataset + " 1> " + output_path + dataset + ".txt 2>&1"
                    # Execute the command
                    os.system(adjustedCommand)
                    scp_checkpoint_and_log_backup()
        else:
            if try_all_models:
                # Model variation
                for model in models:
                    # Adjust the command
                    adjustedCommand = startCommand + " --model " + model + " 1> " + output_path + model + ".txt 2>&1"
                    # Execute the command
                    os.system(adjustedCommand)
                    scp_checkpoint_and_log_backup()
            else:
                # No variation
                exit("just run the normal main.py and you'll be fine")

    # Incremental variation is implemented currently (not multiplicative as might make sense, e.g., for finding the right batch size)
    if try_all_datasets:
        if try_all_models:
            # Both variations
            for dataset in datasets:
                for model in models:
                    current = float(data[vary]["min"])
                    while current <= limit:
                        # Adjust the command
                        adjustedCommand = startCommand + " --" + vary + " " + str(
                            current) + " --dataset " + dataset + " --model " + model + " 1> " + output_path + vary + "_" + str(
                            current) + "_" + model + "_" + dataset + ".txt 2>&1"
                        # Execute the command
                        os.system(adjustedCommand)
                        scp_checkpoint_and_log_backup()
                        current = round_nearest(current + increment, increment)
        else:
            # Dataset variation
            for dataset in datasets:
                current = float(data[vary]["min"])
                while current <= limit:
                    # Adjust the command
                    adjustedCommand = startCommand + " --" + vary + " " + str(current) + " --dataset " + dataset + " 1> " + output_path + vary + "_" + str(current) + "_" + dataset + ".txt 2>&1"
                    # Execute the command
                    os.system(adjustedCommand)
                    scp_checkpoint_and_log_backup()
                    current = round_nearest(current + increment, increment)
    else:
        if try_all_models:
            # Model variation
            for model in models:
                current = float(data[vary]["min"])
                while current <= limit:
                    # Adjust the command
                    adjustedCommand = startCommand + " --" + vary + " " + str(current) + " --model " + model + " 1> " + output_path + vary + "_" + str(current) + "_" + model + ".txt 2>&1"
                    # Execute the command
                    os.system(adjustedCommand)
                    scp_checkpoint_and_log_backup()
                    current = round_nearest(current + increment, increment)
        else:
            # No variation
            current = float(data[vary]["min"])
            while current <= limit:
                # Adjust the command
                adjustedCommand = startCommand + " --" + vary + " " + str(
                    current) + " 1> " + output_path + vary + "_" + str(
                    current) + ".txt 2>&1"
                # Execute the command
                os.system(adjustedCommand)
                scp_checkpoint_and_log_backup()
                current = round_nearest(current + increment, increment)

def scp_checkpoint_and_log_backup():
    os.system("sshpass -p 'PASSWORD' rsync -e \"ssh -p PORT_NUMBER -o StrictHostKeyChecking=no\" -a --ignore-existing "+STORE_EXPERIMENT_LOCATION+" username@ipaddress:"+DEEPHYPERX_PATH_LINUX_GPU02+STORE_EXPERIMENT_SUFFIX+" -v --stats --progress")
    os.system("rm -r "+STORE_EXPERIMENT_LOCATION+"*")

if __name__ == "__main__":
    main()