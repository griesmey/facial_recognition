import glob
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import multiprocessing
import numpy as np
import os
from scipy import misc

datasets_dir = '/media/datasets/'
gender_label_file = 'LFW-gender-folds.dat'
lfw_image_samples = 'lfw_X_bin'
lfw_image_labels = 'lfw_Y_bin' 

def read_gender_labels_file(path):
    labels = {}
    for sample in open(path):
        toks = sample.split('\t')
	label = toks[2].rstrip()
	name = toks[0].split('/')[-1]
	if label == 'M':
            labels[name] = 1
	else:
            labels[name] = 0
    return labels

def display_greyscale_image(img):
    plt.imshow(img, cmap=cm.Greys_r)
    plt.show()    


def process_image(f):
    img = misc.imread(f)
    grey = image2greyscale(img)
    return grey.flatten()

def generate_dataset_from_images(just_Y=False):
    image_data_arr = []
    Y = []
    data_dir = os.path.join(datasets_dir, 'lfw/')
    labels = read_gender_labels_file(os.path.join(datasets_dir, gender_label_file))

    image_paths = []
    for dir in glob.glob(os.path.join(data_dir, '*')):
	for f in glob.glob(os.path.join(dir, '*')):
            image_paths.append(f)

    for path in image_paths:
        Y.append(labels[path.split('/')[-1]])  

    num_cores = multiprocessing.cpu_count()
    print 'number of cores {0}'.format(num_cores)
    results = Parallel(n_jobs=num_cores)(delayed(process_image)(image) for image in image_paths)

    for result in results:
        image_data_arr.extend(result)

    # save
    Y = np.array(Y)
    np.save(os.path.join(datasets_dir, lfw_image_labels), Y)

    X = np.array(image_data_arr).reshape((Y.shape[0], 250*250)).astype(float)
    np.save(os.path.join(datasets_dir, lfw_image_samples), X)

def image2greyscale(image):
    grey = np.zeros((image.shape[0], image.shape[1]))
    for rownum in xrange(len(image)):
        for colnum in xrange(len(image[rownum])):
            grey[rownum][colnum] = rgb2gray(image[rownum][colnum])

    return grey

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def one_hot(x,n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x),n))
    o_h[np.arange(len(x)),x] = 1
    return o_h

def lfw(num_train=10000, onehot=True):
    X = np.load(os.path.join(datasets_dir, lfw_image_samples + '.npy'))
    Y = np.load(os.path.join(datasets_dir, lfw_image_labels + '.npy'))

    X = X/255
    
    trX = X[:num_train]
    teX = X[num_train:]
    trY = Y[:num_train]
    teY = Y[num_train:]

    if onehot:
        trY = one_hot(trY, 2)
        teY = one_hot(teY, 2)
    else:
        trY = np.asarray(trY)
        teY = np.asarray(teY)

    return trX, teX, trY, teY


if __name__ == "__main__":
    generate_dataset_from_images()
