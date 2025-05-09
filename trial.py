import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pathlib import Path
from PIL import Image
import cv2
from tensorflow.keras import layers
from sklearn import preprocessing

image_link = list(Path(r'dataset').glob(r'**/*.jpg'))
image_name = [x.parents[0].stem for x in image_link]
image_label = preprocessing.LabelEncoder().fit_transform(image_name)

import numpy as np
df = pd.DataFrame()
df['link'] = np.array(image_link, dtype = np.str)
df['name'] = image_name
df['label'] = image_label

df


df.name.value_counts().plot(kind = 'bar', figsize = (12, 8), grid = True, color = 'teal')

fig = plt.figure(1, figsize = (15,15))
grid = ImageGrid(fig, 121, nrows_ncols = (5,4), axes_pad = 0.10)
i = 0
for category_id, category in enumerate(df.name.unique()):
    for filepath in df[df['name'] == category]['link'].values[:4]:
        ax = grid[i]
        img = Image.open(filepath)
        ax.imshow(img)
        ax.axis('off')
        if i % 4 == 4-1:
            ax.text(300,100, category, verticalalignment = 'center', fontsize = 20, color ='red')
        i+=1
        
plt.show()

import splitfolders
splitfolders.ratio(r'dataset', output = './', seed = 101,ratio = (.8,.1,.1))

train_df, test_df = train_test_split(df, test_size=0.3, random_state = 1)

train_images = ImageDataGenerator().flow_from_dataframe(
    dataframe = train_df,
    x_col = 'link',
    y_col = 'name',
    color_mode = 'rgb',
    batch_siz = 32,
    target_size=(28,28),
    clas_mode = 'categorical',
    subset = 'training'

)

test_images = ImageDataGenerator().flow_from_dataframe(
    dataframe = test_df,
    x_col = 'link',
    y_col = 'name',
    color_mode = 'rgb',
    batch_siz = 32,
    target_size=(28,28),
    clas_mode = 'categorical'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), input_shape = (28, 28, 3),activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),tf.keras.layers.Conv2D(32, (3,3),activation = 'relu'),tf.keras.layers.MaxPool2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(5, activation = 'softmax')
])



model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(train_images, epochs = 3)

model.evaluate(test_images)