import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def get_logistic_regression_model():
    """Create a simple "logistic regression" model using a single 
    Keras layer.  The layer should use the basic activation, loss
    and optimization methods when compiled that our textbook suggests
    as good defaults for a binary classification problem.

    Returns
    -------
    lr_model - The defined and compiled Keras model with a single output layer,
      effectively a logistic regression classifier.
    """
    # define your model and compile it here
    return None

def get_neural_network_model():
    """Create a neural network model using Keras functional
    API.  The neural network should have an intermediate
    hidden layer with 64 units, and should use a 
    nonlinear output activation function.  Otherwise it is
    similar to the logistic regression model.

    Returns
    -------
    nn_model - A defined and compiled neural network with a single hidden layer of outputs,
       designed for binary classification tasks.
    """
    # define your model and compile it here
    return None

def get_more_powerful_model():
    """Create a neural network model using Keras functional
    API.  Try out some different number of layers, and you can also try
    increasing the number of units in layers.  You should find that
    it is hard to overfit, but you can kind of find a model that
    is powerful enough to show beginning signs of overfitting.

    Returns
    -------
    nn_model - A defined and compiled multi layer neural network
    """
    # define your model and compile it here
    return None

def load_flower_dataset():
    """Generate a "flower" dataset, we generate randomly but set the seed before
    random number generation, so you should always get the same flower dataset.
    The dataset contains 400 samples with 2 features, so can be visualized in
    a plane.  This is a binary calssification problem with two labels, 0 and 1

    Returns
    -------
    X, Y - Returns a (400,2) shaped tensor of the flower dataset, and a (400,) shaped 
      vector of the labels for this dataset.
    """
    # parameters for generation of random flower data
    np.random.seed(1)
    m = 400 # number of samples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower, the range from center to end of flower

    # magic to generate the flower dataset
    for j in range(2):
        sample = range(N * j,N * (j + 1))
        t = np.linspace(j * 3.12,(j + 1) * 3.12, N) + np.random.randn(N) * 0.1 # theta
        r = a*np.sin(4 * t) + np.random.randn(N) * 0.1 # radius
        X[sample] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[sample] = j

    # randomly shuffle data since it is currently sorted by category
    permutation = np.random.permutation(len(Y))
    X = X[permutation]
    Y = Y[permutation]

    return X, Y

def plot_history(ax, history_dict, metric_key):
    """Plot the asked for metrics. Usually we need to plot the metric from the training
    data and its corresponding measurement using the validation data, thus we pass in
    two keys for the training and validation metric to plot.

    Arguments
    ---------
    ax - a matplotlib figure axis to create plot onto
    history_dict - A Python dictionary whose keys should return list like enumerable
      items holding the measured metrics over some number of epochs of training.
    metric_key - The string key for the metric, validation data is assumed to be
      accessible as "val_" + metric_key


    """
    # setup epochs and keys/labels for the plot
    train_key = metric_key
    train_label = "Training " + metric_key
    val_key = "val_" + metric_key
    val_label = "Validation " + metric_key
    epochs = np.arange(1, len(history_dict[train_key]) + 1)
    
    # create the plot of the train and test metric
    ax.plot(epochs, history_dict[train_key], 'r-', label=train_label)
    ax.plot(epochs, history_dict[val_key], 'b-', label=val_label)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric_key)
    #ax.set_xticks(epochs)
    ax.grid()
    ax.legend();