

import numpy as np
import pandas as pd
from keras import backend as K

K.set_image_dim_ordering('th')
from keras.optimizers import SGD
import os.path
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import img_to_array, load_img


metadata = pd.read_csv('/mnt/metadata_sf.csv')
metadata.rename(columns={'id': 'image_id', 'datetaken': 'date_taken'}, inplace=True)


def prepare_image(image_path):
    if os.path.exists(image_path):
        img = load_img(image_path, target_size=(224, 224))  # this is a PIL image
        array = img_to_array(img)
        return array


metadata['img_path'] = '/mnt/images/' + metadata['image_id'].astype(str) + '.jpg'
metadata['img_array'] = metadata['img_path'].apply(lambda row: prepare_image(row))
metadata = metadata[pd.notnull(metadata['img_array'])]

x = np.asarray(metadata['img_array'].tolist()).reshape(len(metadata), 3, 224, 224)


def creating_labels(x):
    if ('topic' in x):
        return x['topic']
    elif ("nature" in str(x['tags_clean'])) or ("lake" in str(x['tags_clean'])) or (
            "river" in str(x['tags_clean'])) or ("view" in str(x['tags_clean'])) or (
            "beach" in str(x['tags_clean'])) or ("flowers" in str(x['tags_clean'])) or (
            "landscape" in str(x['tags_clean'])) or ("waterfall" in str(x['tags_clean'])) or (
            "sunrise" in str(x['tags_clean'])) or ("sunset" in str(x['tags_clean'])) or (
            "water" in str(x['tags_clean'])) or ("nationalpark" in str(x['tags_clean'])) or (
            "alaska" in str(x['tags_clean'])) or ("sky" in str(x['tags_clean'])) or (
            "yosemite" in str(x['tags_clean'])) or ("mountains" in str(x['tags_clean'])):
        return 'Natural Landscape'
    elif ("birds" in str(x['tags_clean'])) or ("wild" in str(x['tags_clean'])) or (
            "wildlife" in str(x['tags_clean'])) or ("forest" in str(x['tags_clean'])) or (
            "animals" in str(x['tags_clean'])) or ("zoo" in str(x['tags_clean'])):
        return 'Animals & Birds'
    elif ("food" in str(x['tags_clean'])) or ("brunch" in str(x['tags_clean'])) or (
            "dinner" in str(x['tags_clean'])) or ("lunch" in str(x['tags_clean'])) or (
            "bar" in str(x['tags_clean'])) or ("restaurant" in str(x['tags_clean'])) or (
            "drinking" in str(x['tags_clean'])) or ("eating" in str(x['tags_clean'])):
        return 'Food'
    elif ("urban" in str(x['tags_clean'])) or ("shop" in str(x['tags_clean'])) or (
            "market" in str(x['tags_clean'])) or ("square" in str(x['tags_clean'])) or (
            "building" in str(x['tags_clean'])) or ("citylights" in str(x['tags_clean'])) or (
            "cars" in str(x['tags_clean'])) or ("traffic" in str(x['tags_clean'])) or (
            "city" in str(x['tags_clean'])) or ("downtown" in str(x['tags_clean'])) or (
            "sanfrancisco" in str(x['tags_clean'])) or ("newyork" in str(x['tags_clean'])) or (
            "newyork" in str(x['tags_clean'])) or ("seattle" in str(x['tags_clean'])) or (
            "sandiego" in str(x['tags_clean'])) or ("washington" in str(x['tags_clean'])):
        return 'Urban Scenes'
    elif ("hotel" in str(x['tags_clean'])) or ("home" in str(x['tags_clean'])) or ("interior" in str(x['tags_clean'])):
        return 'Interiors'
    elif ("us" in str(x['tags_clean'])) or ("people" in str(x['tags_clean'])) or ("group" in str(x['tags_clean'])) or (
            "friends" in str(x['tags_clean'])):
        return 'people'
    else:
        return "Others"


metadata['tags_clean'] = metadata['tags'].str.split()
metadata = metadata.replace(np.nan, '', regex=True)

metadata['labels'] = metadata.apply(creating_labels, axis=1)
topics = metadata['labels'].unique().tolist()
topics = list(set(topics) - set(['Others']))

metadata = metadata.loc[metadata['labels'].isin(topics)]
metadata['labels'].value_counts()


label_map = d = {x: i for i, x in enumerate(topics)}
y = metadata['labels'].apply(lambda row: label_map[row])
y.value_counts()


from sklearn.model_selection import train_test_split
from keras.utils import np_utils

num_classes = 6
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=13)
# Transform targets to keras compatible format
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(6, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit(x_train, y_train, nb_epoch=2, verbose=1, validation_data=(x_test, y_test))

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:


# for i, layer in enumerate(base_model.layers):
# print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit(x_train, y_train, nb_epoch=30, verbose=1, validation_data=(x_test, y_test))

model.save_weights('/mnt/cnn_2epoch_1210.1')

y_true, y_pred = y_test, model.predict(x_test)
