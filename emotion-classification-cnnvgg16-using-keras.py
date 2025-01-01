import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
import os
import random
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout,BatchNormalization ,Activation
from tensorflow.keras.models import Model, Sequential
from keras.applications.nasnet import NASNetLarge
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Chemins principaux
train_dir = r'C:\Users\drief\Desktop\Deep-Transfert learning project\data_\FEDS\train'
test_dir = r'C:\Users\drief\Desktop\Deep-Transfert learning project\data_\FEDS\test'

# Chemins pour les sous-dossiers 10 %
train_subset_dir = r'C:\Users\drief\Desktop\Deep-Transfert learning project\data_\FEDS\train_subset'
test_subset_dir = r'C:\Users\drief\Desktop\Deep-Transfert learning project\data_\FEDS\test_subset'

# Fonction pour copier 10 % des données
def create_subset(input_dir, output_dir, fraction=0.1):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    for class_dir in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_dir)
        if os.path.isdir(class_path):
            os.makedirs(os.path.join(output_dir, class_dir))
            files = os.listdir(class_path)
            subset_files = random.sample(files, int(len(files) * fraction))
            for file in subset_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(output_dir, class_dir))

# Créer les sous-dossiers pour 10 %
create_subset(train_dir, train_subset_dir, fraction=0.1)
create_subset(test_dir, test_subset_dir, fraction=0.1)

# Paramètres
image_size = (48, 48)
batch_size = 128
#epochs = 50

# Préparation des générateurs de données
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)
print(train_generator.class_indices)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Charger le modèle VGG16 pré-entraîné
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
#base_model.trainable = False
for layer in base_model.layers[:-4]:
    layer.trainable=False

# Ajouter des couches personnalisées
"""model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])"""

model=Sequential()
model.add(base_model)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(32,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(7,activation='softmax'))

# Model Summary
model.summary()

import tensorflow as tf


def f1_score(y_true, y_pred):
    # Clip les prédictions et les labels pour garantir qu'ils sont dans [0, 1]
    y_pred = tf.clip_by_value(y_pred, 0, 1)
    y_true = tf.cast(y_true, dtype=tf.float32)

    # Calcul des métriques
    true_positives = tf.reduce_sum(tf.round(y_true * y_pred))
    possible_positives = tf.reduce_sum(tf.round(y_true))
    predicted_positives = tf.reduce_sum(tf.round(y_pred))

    # Précision et rappel
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())

    # Calcul du F1-Score
    f1_val = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1_val


METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
        f1_score,
]

lrd = ReduceLROnPlateau(monitor = 'val_loss',patience = 50,verbose = 1,factor = 0.50, min_lr = 1e-10)

mcp = ModelCheckpoint('vgg16model.keras')

es = EarlyStopping(verbose=1, patience=20)

# Compiler le modèle
#model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=METRICS)

# Entraîner le modèle
"""history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)"""

history=model.fit(train_generator,validation_data=test_generator,epochs = 50,verbose = 1,callbacks=[lrd,mcp,es])

# Sauvegarder le modèle
model.save('vgg16_model.h5')

from tensorflow.keras.models import load_model

# Charger le modèle
#model = load_model('vgg16_model.h5', custom_objects={'f1_score': f1_score})

# Afficher l'historique
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


def Train_Val_Plot(acc, val_acc, loss, val_loss, auc, val_auc, precision, val_precision, f1, val_f1):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle(" MODEL'S METRICS VISUALIZATION ")

    ax1.plot(range(1, len(acc) + 1), acc)
    ax1.plot(range(1, len(val_acc) + 1), val_acc)
    ax1.set_title('History of Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(['training', 'validation'])

    ax2.plot(range(1, len(loss) + 1), loss)
    ax2.plot(range(1, len(val_loss) + 1), val_loss)
    ax2.set_title('History of Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(['training', 'validation'])

    ax3.plot(range(1, len(auc) + 1), auc)
    ax3.plot(range(1, len(val_auc) + 1), val_auc)
    ax3.set_title('History of AUC')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('AUC')
    ax3.legend(['training', 'validation'])

    ax4.plot(range(1, len(precision) + 1), precision)
    ax4.plot(range(1, len(val_precision) + 1), val_precision)
    ax4.set_title('History of Precision')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Precision')
    ax4.legend(['training', 'validation'])

    ax5.plot(range(1, len(f1) + 1), f1)
    ax5.plot(range(1, len(val_f1) + 1), val_f1)
    ax5.set_title('History of F1-score')
    ax5.set_xlabel('Epochs')
    ax5.set_ylabel('F1 score')
    ax5.legend(['training', 'validation'])

    plt.show()


Train_Val_Plot(history.history['accuracy'], history.history['val_accuracy'],
               history.history['loss'], history.history['val_loss'],
               history.history['auc'], history.history['val_auc'],
               history.history['precision'], history.history['val_precision'],
               history.history['f1_score'], history.history['val_f1_score']
               )