# import the necessary packages
import data_utils as du
import argparse
import numpy as np
import torch
from network import ConvNet

MODEL_PATH = "model.ckpt"


#########################################################################
# TODO:                                                                 #
# This is used to input our test dataset to your model in order to      #
# calculate your accuracy                                               #
# Note: The input to the function is similar to the output of the method#
# "get_CIFAR10_data" found in the notebooks.                            #
#########################################################################

def predict_usingCNN(X):
    n_channels = 3
    n_classes = 10
    net = ConvNet(n_input_channels=n_channels, n_output=n_classes)

    if torch.cuda.is_available():
        checkpoint = torch.load(MODEL_PATH)
    else:
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

    net.load_state_dict(checkpoint)

    X = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        net.eval()
        predicted = net.predict(X)
    y_pred = np.array(predicted)
    return y_pred


def main(filename, group_number):
    X, Y = du.load_CIFAR_batch(filename)
    mean_pytorch = np.array([0.4914, 0.4822, 0.4465])
    std_pytorch = np.array([0.2023, 0.1994, 0.2010])
    X_pytorch = np.divide(np.subtract(X / 255, mean_pytorch[:, np.newaxis, np.newaxis]),
                          std_pytorch[:, np.newaxis, np.newaxis])
    prediction_cnn = predict_usingCNN(X_pytorch)
    acc_cnn = sum(prediction_cnn == Y) / len(X_pytorch)
    print("Group %s ... CNN= %f" % (group_number, acc_cnn))


if __name__ == "__main__":
    # # For easy test while implementing
    # group_n = 1
    # for i in range(1, 6):
    #     file = "data/cifar-10-batches-py/data_batch_" + str(i)
    #     main(file, group_n)

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--test", required=True, help="path to test file")
    ap.add_argument("-g", "--group", required=True, help="group number")
    args = vars(ap.parse_args())

    main(args["test"], args["group"])
