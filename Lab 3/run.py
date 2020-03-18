# import the necessary packages
import data_utils as du
import argparse
import numpy as np

#########################################################################
# TODO:                                                                 #
# This is used to input our test dataset to your model in order to      #
# calculate your accuracy                                               #
# Note: The input to the function is similar to the output of the method#
# "get_CIFAR10_data" found in the notebooks.                            #
#########################################################################

def predict_usingCNN(X):
    #########################################################################
    # TODO:                                                                 #
    # - Load your saved model                                               #
    # - Do the operation required to get the predictions                    #
    # - Return predictions in a numpy array                                 #
    # Note: For the predictions, you have to return the index of the max    #
    # value                                                                 #
    #########################################################################
    pass
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return y_pred
   

def main(filename, group_number):
    X,Y = du.load_CIFAR_batch(filename)
    mean_pytorch = np.array([0.4914, 0.4822, 0.4465])
    std_pytorch = np.array([0.2023, 0.1994, 0.2010])
    X_pytorch = np.divide(np.subtract( X/255 , mean_pytorch[:,np.newaxis,np.newaxis]), std_pytorch[:,np.newaxis,np.newaxis])
    prediction_cnn = predict_usingCNN(X)
    acc_cnn = sum(prediction_cnn == Y)/len(X)
    print("Group %s ... CNN= %f"%(group_number, acc_cnn))
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--test", required=True, help="path to test file")
    ap.add_argument("-g", "--group", required=True, help="group number")
    args = vars(ap.parse_args())
    main(args["test"],args["group"])