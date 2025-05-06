# You will need to run the below code.
# pip install tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
from keras.utils import to_categorical

(ds_train, ds_test), ds_info = tfds.load(
    'emnist/balanced',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

def prepare_data(ds):
    images = []
    labels = []
    for image, label in tfds.as_numpy(ds):
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)


train_x, train_y = prepare_data(ds_train)
test_x, test_y = prepare_data(ds_test)


train_x = train_x.astype('float32') / 255.0
test_x = test_x.astype('float32') / 255.0

train_x = train_x.reshape(-1, 28, 28, 1)
test_x = test_x.reshape(-1, 28, 28, 1)

# One-hot encode labels
num_classes = ds_info.features['label'].num_classes
train_y = to_categorical(train_y, num_classes)
test_y = to_categorical(test_y, num_classes)

# Optionally merge datasets
Full_x = np.concatenate((train_x, test_x), axis=0)
Full_y = np.concatenate((train_y, test_y), axis=0)




