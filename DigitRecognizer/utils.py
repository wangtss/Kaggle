import numpy as np
import pandas as pd
import os.path as osp
import matplotlib.pyplot as plt

def load_data(path='data', norm=True):
    """load train test data from local path

    Parameters
    ----------
    path : string, data path (default='data')

    norm : boolean, (default=True)
           whether to normalize the data,
           if True, divide all train and test pixel values by 255.0

    Returns
    -------
    train_label : ndarray, shape (train data size,)

    train_value : ndarray, shape (train data size, 784)}

    test_value : ndarray shape (test data size, 784)
    """

    raw_train = pd.read_csv(osp.join(path, 'train.csv')).values
    raw_test = pd.read_csv(osp.join(path, 'test.csv')).values

    train_value, test_value = raw_train[:, 1:], raw_test[:, 0:]
    if norm:
        train_value = train_value / 255.0
        test_value = test_value / 255.0

    train_label = raw_train[:, 0]

    return train_label, train_value, test_value

def write_result(predictions, filename='result.csv'):
    """write the predictions into a csv file for submission

    Parameters
    ----------
    predictions : ndarray, shape (test data size,)
                  model predictions of test data

    filename : string, output file name (default='result.csv')
    """
    predictions = np.squeeze(predictions)
    result = pd.DataFrame({
        'ImageId': np.arange(1, predictions.shape[0] + 1, 1),
        'Label': predictions
    })
    result.to_csv(filename, index=False)

def make_one_hot_label(label, cls_num=10):
    """convert a 1-D array into a 2-D one-hot array

    Parameters
    ----------
    label : ndarray, should be a 1-D array

    cls_num : class number, integer (default=10)

    Returns
    -------
    one_hot_label : ndarray, shape (label.shape[0], cls_num)
                    2-D one-hot array from label
    """
    label = np.squeeze(label)
    if len(label.shape) > 1:
        raise ValueError('label is not a 1-D array')

    data_size = label.shape[0]
    one_hot_label = np.zeros(shape=[data_size, cls_num], dtype=np.int8)
    one_hot_label[np.arange(data_size), label] = 1

    return one_hot_label

def visualize_data(raw_data):
    """draw the data

    Parameters
    ----------
    raw_data : ndarray, shape (None, 784)
               raw pixel values
    """

    index = np.arange(25, 50)
    imgs = np.zeros(shape=[28 * 5, 28 * 5])
    for i in range(5):
        for j in range(5):
            imgs[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = np.reshape(
                raw_data[index[i * 5 + j]], [28, 28]
            )

    plt.imshow(imgs, cmap='gray')
    plt.axis('off')
    plt.show()

def calculate_acc(prediction, label):
    """calculate accuracy of the predictions

    Parameters
    ----------
    predictions : ndarray, shape (None, 10)
                  model predictions, shuould be the probability of softmax

    label : ndarray, shape (None,)
            ground-truth label

    Returns
    -------
    the accuracy, also a ndarray
    """
    prediction = np.argmax(prediction, 1)
    return np.sum(prediction == label) / label.shape[0]


