import theano
from theano import tensor as T
import numpy as np
from load import lfw, display_greyscale_image

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def model(X, w):
    return softmax(T.dot(X, w))

x = 100
y = 100
trX, teX, trY, teY = lfw(x, y, onehot=True, rescale=True)

X = T.fmatrix()
Y = T.fmatrix()

w = init_weights((x*y, 2))

py_x = model(X, w)
y_pred = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
gradient = T.grad(cost=cost, wrt=w)
update = [[w, w - gradient * 0.05]]

train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print i, np.mean(np.argmax(teY, axis=1) == predict(teX))

fp = 0
fn = 0
tp = 0
tn = 0


for y, prediction in zip(np.argmax(teY, axis=1), predict(teX)):
    if y == 1 and prediction == 1:
        tp += 1
    elif y == 0 and prediction == 1:
        fp += 1
    elif y == 0 and prediction == 0:
        tn += 1
    else:
        fn += 1

print 'true positives {0}\tfalse positives {1}\ntrue negatives {2}\tfalse nagatives {3}\n'.format(tp, fp, tn, fn)
    
