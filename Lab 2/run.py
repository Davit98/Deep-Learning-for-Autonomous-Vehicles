# import the necessary packages
import Softmax.data_utils as du
import argparse
import numpy as np
import torch
from Softmax.linear_classifier import Softmax
from Pytorch.network import Net


#########################################################################
# TODO:                                                                 #
# This is used to input our test dataset to your model in order to      #
# calculate your accuracy                                               #
# Note: The input to the function is similar to the output of the method#
# "get_CIFAR10_data" found in the notebooks.                            #
#########################################################################

def predict_usingPytorch(X):
    #########################################################################
    # TODO:                                                                 #
    # - Load your saved model                                               #
    # - Do the operation required to get the predictions                    #
    # - Return predictions in a numpy array                                 #
    #########################################################################

    # - Load your saved model
    # Create model
    n_feature = 32 * 32 * 3
    n_hidden = 300
    n_classes = 10
    n_hidden_layers = 3
    net = Net(n_feature=n_feature, n_hidden=n_hidden, n_output=n_classes, n_hidden_layers=n_hidden_layers)

    # Load model weights
    path_model = "Pytorch/model.ckpt"
    checkpoint = torch.load(path_model)
    print(net.load_state_dict(checkpoint))

    # - Do the operation required to get the predictions
    images = torch.tensor(X, dtype=torch.float32)
    predicted = net.predict(images)

    # - Return predictions in a numpy array
    y_pred = np.array(predicted)

    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return y_pred


def predict_usingSoftmax(X):
    #########################################################################
    # TODO:                                                                 #
    # - Load your saved model                                               #
    # - Do the operation required to get the predictions                    #
    # - Return predictions in a numpy array                                 #
    #########################################################################
    WEIGHTS_PATH = 'Softmax/softmax_weights.pkl'
    with open(WEIGHTS_PATH, 'rb') as f:
        weights = du.load_pickle(f)

    model = Softmax()
    model.W = weights.copy()

    y_pred = model.predict(X)
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return y_pred  # y_pred

def main(filename, group_number):
    X, Y = du.load_CIFAR_batch(filename)
    X = np.reshape(X, (X.shape[0], -1))
    mean_image = np.mean(X, axis=0)
    print("Mean", mean_image)
    # X -= mean_image
    prediction_pytorch = predict_usingPytorch(X)
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    prediction_softmax = predict_usingSoftmax(X)
    acc_softmax = sum(prediction_softmax == Y) / len(X)
    acc_pytorch = sum(prediction_pytorch == Y) / len(X)
    print("Group %s ... Softmax= %f ... Pytorch= %f" % (group_number, acc_softmax, acc_pytorch))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--test", required=True, help="path to test file")
    ap.add_argument("-g", "--group", required=True, help="group number")
    args = vars(ap.parse_args())
    main(args["test"], args["group"])
