#%%
#import packages
import tensorflow as tf
from tensorflow import keras
from IPython.display import clear_output
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import layers,optimizers,losses,callbacks,applications
import numpy as np
import datetime
import os
import cv2
import matplotlib.pyplot as plt 

#%%
#empty list
inputs_train = []
masks_train = []
inputs_test = []
masks_test = []
#%%
#Path
inputs_train_path = os.path.join(os.getcwd(),'dataset','train','inputs')
masks_train_path = os.path.join(os.getcwd(),'dataset','train','masks')
inputs_test_path = os.path.join(os.getcwd(),'dataset','test','inputs')
masks_test_path = os.path.join(os.getcwd(),'dataset','test','masks')

#%%
# Use os.listdir() method to list down all the inputs file (train)
inputs_files = os.listdir(inputs_train_path)
for img in inputs_files:
    images_dir = os.path.join(inputs_train_path,img)
    converted_inputs_train = cv2.imread(images_dir)
    converted_inputs_train = cv2.cvtColor( converted_inputs_train,cv2.COLOR_BGR2RGB)
    converted_inputs_train = cv2.resize( converted_inputs_train,(128,128))
    inputs_train.append( converted_inputs_train)
# %%
# Use os.listdir() method to list down all the masks file (train)
mask_files = os.listdir(masks_train_path)
for mask in mask_files:
    mask_dir = os.path.join(masks_train_path,mask)
    converted_masks_train = cv2.imread(mask_dir,cv2.IMREAD_GRAYSCALE)
    converted_masks_train = cv2.resize(converted_masks_train,(128,128))
    masks_train.append(converted_masks_train)
#%%
# Use os.listdir() method to list down all the inputs file (test)
inputs_files_test = os.listdir(inputs_test_path)
for imgs in inputs_files_test:
    images_dir_test = os.path.join(inputs_test_path,imgs)
    converted_inputs_test = cv2.imread(images_dir_test)
    converted_inputs_test = cv2.cvtColor(converted_inputs_test,cv2.COLOR_BGR2RGB)
    converted_inputs_test = cv2.resize(converted_inputs_test,(128,128))
    inputs_test.append( converted_inputs_test)


#%%
# Use os.listdir() method to list down all the masks file (test)
mask_files_test = os.listdir(masks_test_path)
for masks in mask_files_test:
    mask_dir_test = os.path.join(masks_test_path,masks)
    converted_masks_test = cv2.imread(mask_dir_test,cv2.IMREAD_GRAYSCALE)
    converted_masks_test = cv2.resize(converted_masks_test,(128,128))
    masks_test.append(converted_masks_test)



# %%
#Convert to fully numpy array
converted_inputs_train = np.array(inputs_train)
converted_masks_train = np.array(masks_train)
converted_inputs_test = np.array(inputs_test)
converted_masks_test = np.array(masks_test)
# %%
#Check some examples
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(converted_inputs_train[i])
    plt.axis('off')
    
plt.show()
#%%

plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(converted_masks_train[i])
    plt.axis('off')
    
plt.show()
#%%
# Data preprocessing

#Expand the mask dimension

masks_np_exp = np.expand_dims(converted_masks_train,axis=-1)
masks_test_np_exp = np.expand_dims(converted_masks_test,axis=-1)

#Check the mask output

print(np.unique(masks_train[0]))
print(np.unique(masks_test[0]))

#%%
#Convert the mask values into class labels

converted_masks_train = np.round(masks_np_exp /255.0).astype(np.int64)
converted_masks_test = np.round(masks_test_np_exp /255.0).astype(np.int64)
#Check the mask output

print(np.unique(converted_masks_train[0]))
print(np.unique(converted_masks_test[0]))

#%%
#Normalize image pixels value
converted_inputs_train = converted_inputs_train / 255.0
converted_inputs_test = converted_inputs_test / 255.0
sample = converted_inputs_train[0]
sample = converted_inputs_test[0]
# %%
# Convert the numpy array into tensorflow tensors

X_train_tensor = tf.data.Dataset.from_tensor_slices(converted_inputs_train)
X_test_tensor = tf.data.Dataset.from_tensor_slices(converted_inputs_test)
y_train_tensor = tf.data.Dataset.from_tensor_slices(converted_masks_train)
y_test_tensor = tf.data.Dataset.from_tensor_slices(converted_masks_test)
# %%
# Combine features and labels together to form a zip dataset

train_dataset = tf.data.Dataset.zip((X_train_tensor,y_train_tensor))
test_dataset = tf.data.Dataset.zip((X_test_tensor,y_test_tensor))
# %%
#[EXTRA] Create a subclass layer for data augmentation
class Augment(layers.Layer):
    def __init__(self,seed=42):
        super().__init__()
        self.augment_inputs = layers.RandomFlip(mode='horizontal',seed=seed)
        self.augment_labels = layers.RandomFlip(mode='horizontal',seed=seed)
        
    def call(self,inputs,labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs,labels

#%%
#Convert into prefetch dataset
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
TRAIN_SIZE = len(train_dataset)
STEPS_PER_EPOCH = TRAIN_SIZE//BATCH_SIZE

train_batches = (train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat().map(Augment()).prefetch(buffer_size=tf.data.AUTOTUNE))
test_batches = test_dataset.batch(BATCH_SIZE)

# %%
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image','True Mask','Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
        
    plt.show()

#%%

for images, masks in train_batches.take(2):
    sample_image,sample_mask = images[0],masks[0]
    display([sample_image,sample_mask])

#%%
# Model development
# Use a pretrained model as the feature extractor
base_model = tf.keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)
# %%
# Use these activation layers as the output from the feature extractor (some of the outputs will be use to perform concatenatuion with the unsampling path)

layer_names = ['block_1_expand_relu','block_3_expand_relu','block_6_expand_relu','block_13_expand_relu','block_16_project']

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
# %%
# Instantiate the feature extractor

down_stack = tf.keras.Model(inputs=base_model.input,outputs=base_model_outputs)
down_stack .trainable = False

#%%
# Define the upsampling path
up_stack = [pix2pix.upsample(512,3),pix2pix.upsample(256,3),pix2pix.upsample(128,3),pix2pix.upsample(64,3)]
# %%
# Functional API to construct the entire unit

def unet(output_channels:int):
    inputs = tf.keras.layers.Input(shape=[128,128,3])

    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    #Build the upsampling path and establishing the skip connections
    for up,skip in zip(up_stack,skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x,skip])

   # Use a transpose convolution layer to perfoem one last upsampling.

    last = tf.keras.layers.Conv2DTranspose(filters=output_channels,kernel_size=3,strides=2,padding='same')

    outputs = last(x)

    model = tf.keras.Model(inputs=inputs,outputs=outputs)

    return model

# %%
# Create the model using the above function

OUTPUT_CLASSES = 2
model = unet(OUTPUT_CLASSES)
model.summary()
tf.keras.utils.plot_model(model)

#%%
# Compile the model
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])
# %%
#Create functions to show predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])

    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])

show_predictions()

# %%
LOGS_PATH = os.path.join(os.getcwd(),'logs', datetime.datetime.now().strftime('%Y&m%d-%H%M%S'))
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH)
# %%
# Create callback function
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample prediction after epoch {}\n'.format(epoch+1))

#%%
#Hyperparameters for the model
EPOCHS = 10
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(test_dataset)//BATCH_SIZE//VAL_SUBSPLITS

history = model.fit(train_batches,validation_data=test_batches,epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,validation_steps=VALIDATION_STEPS,callbacks=[tensorboard_callback,DisplayCallback()])
# %%
#10. Deploy model
show_predictions(test_batches,3)
# %%
