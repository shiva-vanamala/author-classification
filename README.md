# author-classification
The project performs the tasks of identifying the author of a book

## Results
Model run only for 5 epochs,
Train on 58995 samples, validate on 6555 samples
Epoch 1/5
209s - loss: 0.0850 - acc: 0.9728 - val_loss: 0.0480 - val_acc: 0.9834
Epoch 2/5
204s - loss: 0.0412 - acc: 0.9861 - val_loss: 0.0298 - val_acc: 0.9899
Epoch 3/5
203s - loss: 0.0265 - acc: 0.9910 - val_loss: 0.0233 - val_acc: 0.9919
Epoch 4/5
203s - loss: 0.0183 - acc: 0.9938 - val_loss: 0.0188 - val_acc: 0.9936
Epoch 5/5
204s - loss: 0.0129 - acc: 0.9956 - val_loss: 0.0172 - val_acc: 0.9940

notes:
1. validation set does not consist of samples from unique book not seen in train set
2. due to speed limitations did not run on too many models
3. only used relu, could try tanh + batch normalization
4. a smaller model would suffice for this case
5. small word2vec vectors used for this case, could be expanded to use a pretrained word2vec model, with transfer learning on the training set. But due to the usage of short vectors, no pretrained models available

## Requirements
1. python2
2. numpy
3. tensorflow
4. keras
5. gensim

## Files
1. TrainCNN.sh: to train and find validation score
2. ProcessData.sh: run to filter out the license agreements from all the books (no raw data included in the git, so will not run)
3. src:
  a. cnn_model.py
  b. w2v.py
  c. data_helpers.py
  d. TrainCNN.sh: script to 
4. data:
  a. processed: filtered and combined text files, each file contains all works by an author
  b. marices: contains the data as word indexes and proper  labels
  c. word2vec_models: all the word2vec models stored here
  d. models: all cnn models stored here
  
## Run pipeline
1. run to filter out the license agreements from all the books (no raw data included in the git, so will not run)
    ./ProcessData.sh
2. to train and find validation score
    ./TrainCNN.sh
