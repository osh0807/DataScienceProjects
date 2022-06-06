import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import expit
from sklearn.utils import shuffle
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from os.path import isfile
from tensorflow.keras.preprocessing import image
from IPython.display import Image

# Plotting Support Function
def configure_plots():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for _ in range(2): # needs to run twice for some reason
            sns.set(style="ticks", color_codes=True, font_scale=1.5)
            sns.set_palette(sns.color_palette())
            plt.rcParams['figure.figsize'] = [16, 9]
            plt.rcParams['axes.titlesize'] = 20
            plt.rcParams['axes.labelsize'] = 16
            plt.rcParams['xtick.labelsize'] = 14
            plt.rcParams['ytick.labelsize'] = 14
            plt.rcParams['lines.linewidth'] = 2

        print('Plots configured! 📊')


def load_toy():
    '''load the data set as a two-feature binary classification problem'''
    data = load_breast_cancer() #load_iris()

    # ignore duplicate observations (if any)
    X, idx = np.unique(data.data[:, :2], axis=0, return_index=True)
    y = np.where(data.target[idx] > 0, -1, 1) # make binary, convert to -1, +1

    X, y = shuffle(X, y, random_state=2)

    return X[:117,:], y[:117]


def optimize(gradient_fn, X, y, theta, eta=1e-2, iterations=5e4, eps=1e-3):
    '''
    computes weights W* that optimize the given the derivative of the loss function
    DFN given starting weights W
    '''

    for _ in range(int(iterations)):
        grad = gradient_fn(X, y, theta)

        if np.linalg.norm(grad) < eps:
            break

        theta += eta * grad

    return theta

def sigmoid(x):
    return expit(x)

def logistic_gradient(X, y, theta):
    '''Computes the gradient of the logistic loss function with respect to theta'''

    N, _ = X.shape

    return (X * y).T.dot(sigmoid(-y * X.dot(theta)))

def optimize_logistic(X, y, theta=None, **kwargs):
    if theta is None:
        _, d = X.shape
        theta = np.zeros((d, 1))

    y = y.reshape(-1, 1)

    return optimize(logistic_gradient, X, y, theta, **kwargs).squeeze()

def plot_confusion_matrix(confusion_matrix, labels=None, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    adapted from sklearn
    """
    if not labels:
        labels = ['+1', '-1']

    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    confusion_matrix = np.rot90(confusion_matrix, 2)

    fig = plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = '.2f' if (normalize or np.sum(confusion_matrix[1, :]) == 1) else 'd'

    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], fmt), FontSize='15', horizontalalignment="center", color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return fig.axes

def preprocess_image(preprocess_fn, path, target_size=(224, 224)):
    '''
    Takes an image path and returns a preprocessed version of it.
    '''
    img = image.load_img(path, target_size=target_size)
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    return preprocess_fn(img_data)


def extract_features(extractor_model, preprocess_fn, paths):
    '''
    Extracts features using the extractor model and returns a matrix of
    extracted feature vectors.
    '''
    if type(paths) is not list:
        paths = [path]

    features = []

    print('Extracting Features from Images:')

    for i, path in enumerate(paths):
        if not isfile(path):
            raise FileNotFoundError(f'Path is invalide (given {path})')

        print('.', end='')
        if not (i + 1) % 50:
            print(i + 1)

        # passed `preprocess_fn` to preprocessor
        processed_image = preprocess_image(preprocess_fn, path)

        feature = np.array(extractor_model.predict(processed_image)).flatten()
        features.append(feature)

    return np.array(features)


def show_image(path):
    '''
    Loads and displays the image at PATH
    '''

    image = Image(path)
    display(image)