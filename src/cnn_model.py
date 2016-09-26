import numpy as np
import pickle
from data_helpers import load_data
from w2v import train_word2vec

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization

'''
store files at
'''
X_FILE = '../data/matrices/x.npy'
Y_FILE = '../data/matrices/y.npy'
VOCABULARY_FILE = '../data/matrices/vocabulary.pickle'
VOCABULARY_INV_FILE = '../data/matrices/vocabulary_inv.pickle'

'''
training hyperparameters
'''
BATCH_SIZE = 256
NUM_EPOCHS = 5

MODEL_FILE = '../data/models/cnn_model.h5'
VALIDATION_SPLIT = 0.1

TRAIN_WORD2VEC = True
EMBEDDING_SIZE = 20
'''
model hyperparameters
'''
ACTIVATION = 'relu'

DROPOUT_0 = 0.25

NUM_FILTERS_1 = 150
FILTER_LENGTH_1 = 4
POOL_LENGTH_1 = 2

HIDDEN_LAYER_2 = 150
DROPOUT_2 = 0.25

OPTIMIZER = 'adam'

'''
word2vec parameters
'''
MIN_WORD_COUNT = 1
CONTEXT_WINDOW_SIZE = 10

'''
build the CNN model with
1 - convolutional layer
2 - hidden dense layer
3 - output softmax layer
'''
def cnn_model(input_shape, num_classes, vocabulary_size, embedding_weights):
    # get model
    model = Sequential()
    
    # embedding layer
    model.add(Embedding(vocabulary_size, input_shape[1], input_length=input_shape[0], weights=embedding_weights))
    model.add(Dropout(DROPOUT_0, input_shape=input_shape))
    
    # conv layer 1
    #model.add(Input(shape=input_shape))
    model.add(Convolution1D(nb_filter=NUM_FILTERS_1, filter_length=FILTER_LENGTH_1, activation=ACTIVATION, border_mode='valid', subsample_length=1))
    model.add(MaxPooling1D(pool_length=POOL_LENGTH_1))
    
    # flatten
    model.add(Flatten())
    
    # dense layer 2
    model.add(Dense(HIDDEN_LAYER_2))
    model.add(Dropout(DROPOUT_2))
    model.add(Activation(ACTIVATION))
    
    # output layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
    
    return model

'''
train, save and return the model
'''
def train_cnn(model, x, y, num_epochs, model_file, save_model):
    model.fit(x, y, batch_size=BATCH_SIZE,
          nb_epoch=num_epochs, validation_split=VALIDATION_SPLIT, verbose=2)
    
    if save_model:
        model.save_weights(model_file)
    
    return model
    
def get_data(load_from_file):

    if load_from_file:
        # load data from npy files
        print "reading data from disk"
        x = np.load(X_FILE)
        y = np.load(Y_FILE)
        vocabulary = pickle.load(open(VOCABULARY_FILE, "rb"))
        vocabulary_inv = pickle.load(open(VOCABULARY_INV_FILE, "rb"))
        
    else:
        # load data from text
        print "loading data"
        x, y, vocabulary, vocabulary_inv = load_data()
        
        # save to file
        print "saving data to disk"
        np.save(X_FILE, x)
        np.save(Y_FILE, y)
        pickle.dump(vocabulary, open(VOCABULARY_FILE, "wb"))
        pickle.dump(vocabulary_inv, open(VOCABULARY_INV_FILE, "wb"))
        
    return x, y, vocabulary, vocabulary_inv

'''
training pipeline
'''
def main(load_from_file):

    # obtain the data to train and validate on
    x, y, vocabulary, vocabulary_inv = get_data(load_from_file)
           
    print "randomly shuffling loaded data"
    new_indices = np.random.permutation(np.arange(len(x)))
    x = x[new_indices]
    y = y[new_indices]
        
    # find shape of input
    input_shape = (len(x[0]), EMBEDDING_SIZE)

    # if chosen, train the word2vec model
    if TRAIN_WORD2VEC:
        embedding_weights = train_word2vec(x, vocabulary_inv, EMBEDDING_SIZE, MIN_WORD_COUNT, CONTEXT_WINDOW_SIZE)
    else:
        embedding_weights = None
        
    # get the cnn model
    model = cnn_model(input_shape, np.shape(y[0])[0], len(vocabulary), embedding_weights)
    
    # begin training model
    print "training cnn model"
    model = train_cnn(model, x, y, NUM_EPOCHS, MODEL_FILE, False)
    
    return 0

if __name__=='__main__':
    main(False)
