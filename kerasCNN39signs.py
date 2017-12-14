# USAGE
# python simple_neural_network.py --dataset kaggle_dogs_vs_cats

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
#from keras.layers import Activation
from keras.optimizers import SGD
#from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import cv2
import os
from os import listdir
import keras
# Import libraries
#import os,cv2
#import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split

from keras import backend as K
from keras.regularizers import Regularizer
from keras import regularizers
# K.set_image_dim_ordering('th')
K.set_image_dim_ordering('tf')

#from keras.utils import np_utils
#from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
#####################################################################################################################
#%%
num_classes = 39
num_epoch=30
num_channel=3

def image_to_feature_vector(image, size=(64,64),flatten=True):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
    if flatten == False : return cv2.resize(image,size);
    return cv2.resize(image, size).flatten()

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required=True,
#	help="path to input dataset")
#args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] describing images...")
#imagePaths = list(paths.list_images(args["dataset"]))
#imagePaths = list(paths.list_images('train'))
#imagePathsa = list(paths.list_images('F:\sem 6\capstone\Sign-Language-Interpreter-latest work\a\output3Ds'))

projectFolder =r"F:\sem 6\capstone\NovemberTraining Shivam"
foldera = projectFolder+r'/Sign-Language-Interpreter-latest work/a/output3Ds/'
folderb = projectFolder+r'/Sign-Language-Interpreter-latest work/b/output3Ds/'
folderc = projectFolder+r'/Sign-Language-Interpreter-latest work/c/output3Ds/'
folderd = projectFolder+r'/Sign-Language-Interpreter-latest work/d/output3Ds/'
foldere = projectFolder+r'/Sign-Language-Interpreter-latest work/e/output3Ds/'
folderf = projectFolder+r'/Sign-Language-Interpreter-latest work/f/output3Ds/'
folderg = projectFolder+r'/Sign-Language-Interpreter-latest work/g/output3Ds/'
folderh = projectFolder+r'/Sign-Language-Interpreter-latest work/h/output3Ds/'
folderi = projectFolder+r'/Sign-Language-Interpreter-latest work/i/output3Ds/'
folderj = projectFolder+r'/Sign-Language-Interpreter-latest work/j/output3Ds/'
folderk = projectFolder+r'/Sign-Language-Interpreter-latest work/k/output3Ds/'
folderl = projectFolder+r'/Sign-Language-Interpreter-latest work/l/output3Ds/'
folderm = projectFolder+r'/Sign-Language-Interpreter-latest work/m/output3Ds/'
foldern = projectFolder+r'/Sign-Language-Interpreter-latest work/n/output3Ds/'
foldero = projectFolder+r'/Sign-Language-Interpreter-latest work/o/output3Ds/'
folderp = projectFolder+r'/Sign-Language-Interpreter-latest work/p/output3Ds/'
folderq = projectFolder+r'/Sign-Language-Interpreter-latest work/q/output3Ds/'
folderr = projectFolder+r'/Sign-Language-Interpreter-latest work/r/output3Ds/'
folders = projectFolder+r'/Sign-Language-Interpreter-latest work/s/output3Ds/'
foldert = projectFolder+r'/Sign-Language-Interpreter-latest work/t/output3Ds/'
folderu = projectFolder+r'/Sign-Language-Interpreter-latest work/u/output3Ds/'
folderv = projectFolder+r'/Sign-Language-Interpreter-latest work/v/output3Ds/'
folderw = projectFolder+r'/Sign-Language-Interpreter-latest work/w/output3Ds/'
folderx = projectFolder+r'/Sign-Language-Interpreter-latest work/x/output3Ds/'
foldery = projectFolder+r'/Sign-Language-Interpreter-latest work/y/output3Ds/'
folderz = projectFolder+r'/Sign-Language-Interpreter-latest work/z/output3Ds/'
folderaboard=projectFolder+r'/Sign-Language-Interpreter-latest work/aboard/output3Ds/'
folderallgone=projectFolder+r'/Sign-Language-Interpreter-latest work/allgone/output3Ds/'
folderarrest=projectFolder+r'/Sign-Language-Interpreter-latest work/arrest/output3Ds/'
folderbeside=projectFolder+r'/Sign-Language-Interpreter-latest work/beside/output3Ds/'
folderhouse=projectFolder+r'/Sign-Language-Interpreter-latest work/house/output3Ds/'
folderhungry=projectFolder+r'/Sign-Language-Interpreter-latest work/hungry/output3Ds/'
folderhunt=projectFolder+r'/Sign-Language-Interpreter-latest work/hunt/output3Ds/'
folderlisten=projectFolder+r'/Sign-Language-Interpreter-latest work/listen/output3Ds/'
folderman=projectFolder+r'/Sign-Language-Interpreter-latest work/man/output3Ds/'
folderme=projectFolder+r'/Sign-Language-Interpreter-latest work/me/output3Ds/'
folderoath=projectFolder+r'/Sign-Language-Interpreter-latest work/oath/output3Ds/'
folderprisoner=projectFolder+r'/Sign-Language-Interpreter-latest work/prisoner/output3Ds/'
folderdelete = projectFolder+r'/Sign-Language-Interpreter-latest work/delete/output3Ds/'





imagePathsa = list(os.listdir(foldera))
imagePathsb = list(os.listdir(folderb))
imagePathsc = list(os.listdir(folderc))
imagePathsd = list(os.listdir(folderd))
imagePathse = list(os.listdir(foldere))
imagePathsf = list(os.listdir(folderf))
imagePathsg = list(os.listdir(folderg))
imagePathsh = list(os.listdir(folderh))
imagePathsi = list(os.listdir(folderi))
imagePathsj = list(os.listdir(folderj))
imagePathsk = list(os.listdir(folderk))
imagePathsl = list(os.listdir(folderl))
imagePathsm = list(os.listdir(folderm))
imagePathsn = list(os.listdir(foldern))
imagePathso = list(os.listdir(foldero))
imagePathsp = list(os.listdir(folderp))
imagePathsq = list(os.listdir(folderq))
imagePathsr = list(os.listdir(folderr))
imagePathss = list(os.listdir(folders))
imagePathst = list(os.listdir(foldert))
imagePathsu = list(os.listdir(folderu))
imagePathsv = list(os.listdir(folderv))
imagePathsw = list(os.listdir(folderw))
imagePathsx = list(os.listdir(folderx))
imagePathsy = list(os.listdir(foldery))
imagePathsz = list(os.listdir(folderz))

imagePathsaboard = list(os.listdir(folderaboard))
imagePathsallgone = list(os.listdir(folderallgone))
imagePathsarrest = list(os.listdir(folderarrest))
imagePathsbeside = list(os.listdir(folderbeside))
imagePathshouse = list(os.listdir(folderhouse))
imagePathshungry = list(os.listdir(folderhungry))
imagePathshunt = list(os.listdir(folderhunt))
imagePathslisten = list(os.listdir(folderlisten))
imagePathsman = list(os.listdir(folderman))
imagePathsme = list(os.listdir(folderme))
imagePathsoath = list(os.listdir(folderoath))
imagePathsprisoner = list(os.listdir(folderprisoner))
imagePathsdelete =  list(os.listdir(folderdelete))

folder=     [foldera,       folderb,    folderc,    folderd,    foldere,    folderf,    folderg,    folderh,    folderi,    folderj,    folderk,    folderl,    folderm,    foldern,    foldero,    folderp,    folderq,    folderr,    folders,    foldert,    folderu,    folderv,    folderw,    folderx,    foldery,    folderz,    folderaboard,       folderallgone,      folderarrest,       folderbeside,       folderhouse,        folderhungry,       folderhunt,     folderlisten,       folderman,      folderme,       folderoath,     folderprisoner,     folderdelete]
imagePaths= [imagePathsa,   imagePathsb,imagePathsc,imagePathsd,imagePathse,imagePathsf,imagePathsg,imagePathsh,imagePathsi,imagePathsj,imagePathsk,imagePathsl,imagePathsm,imagePathsn,imagePathso,imagePathsp,imagePathsq,imagePathsr,imagePathss,imagePathst,imagePathsu,imagePathsv,imagePathsw,imagePathsx,imagePathsy,imagePathsz,imagePathsaboard,   imagePathsallgone,  imagePathsarrest,   imagePathsbeside,   imagePathshouse,    imagePathshungry,   imagePathshunt, imagePathslisten,   imagePathsman,  imagePathsme,   imagePathsoath, imagePathsprisoner, imagePathsdelete]



# initialize the data matrix and labels list
data = []
labels = []
imgSize = (128,128)

######################################
for k in range(num_classes):
    for(j,imagePath) in enumerate(imagePaths[k]):
        image = cv2.imread(folder[k]+imagePath)
        imageName=imagePath.split(os.path.sep)[-1]
        label = imageName.split(".")[0]

    	# construct a feature vector raw pixel intensities, then update
    	# the data matrix and labels list
        features = image_to_feature_vector(image,size=imgSize,flatten=False)
        data.append(features)
        labels.append(label)

    	# show an update every 1,000 images
        if j > 0 and j % 500 == 0:
            print("[INFO] processed {}/{}/{}".format(j, len(imagePaths[k]),label))

####################################################

#%%
# encode the labels, converting them from strings to integers
le = LabelEncoder()
print (labels)
labels = le.fit_transform(labels)
print (labels)
# scale the input image pixels to the range [0, 1], then transform
# the labels into vectors in the range [0, num_classes] -- this
# generates a vector for each label where the index of the label
# is set to `1` and all other entries to `0`
data = np.array(data) / 255.0
# labels = np_utils.to_categorical(labels, 2)
labels = np_utils.to_categorical(labels, num_classes)

#%%

#Shuffle the dataset
data,labels = shuffle(data,labels, random_state=2)
# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data, labels, test_size=0.25, random_state=42)
#%%
X_train     = trainData
X_test      = testData
y_train     = trainLabels
y_test      = testLabels;

# define the architecture of the network


#%%
# Defining the model
input_shape=data[0].shape
#num_classes = 26
num_of_samples = data.shape[0]
print (num_of_samples)
#%%
model = Sequential()

model.add(Convolution2D(32, 7,7,border_mode='same',input_shape=input_shape))
model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Activation('relu'))

model.add(Convolution2D(64, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 5,5,border_mode='same',input_shape=input_shape))
model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

#model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Activation('relu'))
model.add(Convolution2D(128, 3,3,border_mode='same',input_shape=input_shape))
#model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Activation('relu'))
model.add(Convolution2D(256, 3,3,border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Convolution2D(64, 3, 3))
model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

model.add(Activation('relu'))
#model.add(Convolution2D(64, 3, 3))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
#model.add(Dense(512))
model.add(Dense(256,kernel_regularizer=regularizers.l2(0.000000000001),activity_regularizer=regularizers.l1(0.000000000000001)))
model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Dropout(0.5))
#model.add(Dense(512))
model.add(Dense(768,kernel_regularizer=regularizers.l2(0.000000000001),activity_regularizer=regularizers.l1(0.000000000000001)))
model.add(Activation('relu'))
#model.add(Dropout(0.25))
#model.add(Dense(768,init='uniform',activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(384,init='uniform',activation='relu'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
#%%
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
nadam = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])

# Viewing model_configuration

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

#%%
# Training
#hist = model.fit(trainData, trainLabels, batch_size=200, nb_epoch=num_epoch, verbose=1, validation_data=(testData, testLabels))


# %%
#hist = model.fit(X_train, y_train, batch_size=32, nb_epoch=20,verbose=1, validation_split=0.2)

# Training with callbacks
from keras import callbacks

filename='model_train_new.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)

early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')

filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [csv_log,early_stopping,checkpoint]

hist1 = model.fit(X_train, y_train, batch_size=50, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test),callbacks=callbacks_list,shuffle=True)

#%%
hist = hist1
# %%


# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc = range(14)
#xc=range(num_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['ggplot'])

#%%

# Evaluating the model

#score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
score = model.evaluate(X_test, y_test, verbose=1)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[0:1]
print (test_image.shape)

print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])


# %%

#%%

# Evaluating the model

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[0:1]
print (test_image.shape)

print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])
#%%
# Testing a new image
test_image = cv2.imread(projectFolder+'/Sign-Language-Interpreter-latest work/m/output3Ds/m.500.3ds.png')
#test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,imgSize)
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)

if num_channel==1:
	if K.image_dim_ordering()=='th':
		test_image= np.expand_dims(test_image, axis=0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=3)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)

else:
	if K.image_dim_ordering()=='th':
		test_image=np.rollaxis(test_image,2,0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)

# Predicting the test image
#print((model.predict(test_image)))
print(model.predict_classes(test_image))

#%%

# Visualizing the intermediate layer

#
def get_featuremaps(model, layer_idx, X_batch):
	get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
	activations = get_activations([X_batch,0])
	return activations

layer_num=4
filter_num=0

activations = get_featuremaps(model, int(layer_num),test_image)

print (np.shape(activations))
feature_maps = activations[0][0]
print (np.shape(feature_maps))

if K.image_dim_ordering()=='th':
	feature_maps=np.rollaxis((np.rollaxis(feature_maps,2,0)),2,0)
print (feature_maps.shape)

fig=plt.figure(figsize=(16,16))
plt.imshow(feature_maps[:,:,filter_num],cmap='gray')
plt.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num)+'.jpg')

num_of_featuremaps=feature_maps.shape[2]
fig=plt.figure(figsize=(16,16))
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
for i in range(int(num_of_featuremaps)):
	ax = fig.add_subplot(subplot_num, subplot_num, i+1)
	#ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
	ax.imshow(feature_maps[:,:,i],cmap='gray')
	plt.xticks([])
	plt.yticks([])
	plt.tight_layout()
plt.show()
fig.savefig("featuremaps-layer-{}".format(layer_num) + '.jpg')

#%%
# Printing the confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
#y_pred = model.predict_classes(X_test)
#print(y_pred)
target_names = ['class 0 (a)','class 1 (aboard)','class 2 (allgone)','class 3 (arrest)', 'class 4 (b)'
                ,'class 5 (beside)','class 6 (c)','class 7 (d)','class 8 (delete)' ,'class 9 (e)','class 10 (f)'
                ,'class 11 (g)','class 12 (h)','class 13 (house)','class 14 (hungry)','class 15 (hunt)'
                ,'class 16 (i)','class 17 (j)','class 18 (k)', 'class 19 (l)','class 20 (listen)'
                ,'class 21 (m)','class 22 (man)','class 23 (me)'
                ,'class 24 (n)','class 25 (o)','class 26 (oath)','class 27 (p)','class 28 (prisoner)'
                ,'class 29 (q)', 'class 30 (r)'
                ,'class 31 (s)','class 32 (t)','class 33 (u)','class 34 (v)','class 35 (w)','class 36 (x)'
                ,'class 37 (y)','class 38 (z)'
                ]

print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))


# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))

np.set_printoptions(precision=4)

plt.figure(figsize=(16, 16), dpi=150)

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
#plt.figure()
# Plot normalized confusion matrix
#plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
#                      title='Normalized confusion matrix')
#plt.figure()
plt.show()

#%%
# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model
import h5py as h5py

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
#model.load_weights("Best-weights-my_model-010-0.1810-0.9500.hdf5")
print("Loaded model from disk")

model.save('model.hdf5')
loaded_model=load_model('model.hdf5')


# %%

# model = Sequential()
# model.add(Dense(768, input_dim=3072, init="uniform",
# 	activation="relu"))
# model.add(Dense(384, init="uniform", activation="relu"))
# model.add(Dense(2))
# model.add(Activation("softmax"))
#
# # train the model using SGD
# print("[INFO] compiling model...")
# sgd = SGD(lr=0.01)
# model.compile(loss="binary_crossentropy", optimizer=sgd,
# 	metrics=["accuracy"])
# model.fit(trainData, trainLabels, nb_epoch=50, batch_size=128,
# 	verbose=1)
#
# # show the accuracy on the testing set
# print("[INFO] evaluating on testing set...")
# (loss, accuracy) = model.evaluate(testData, testLabels,
# 	batch_size=128, verbose=1)
# print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
# 	accuracy * 100))
