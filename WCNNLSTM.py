import numpy as np
import struct
import pywt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, ConvLSTM3D, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
import matplotlib.pyplot as plt


#data loading-----------------------------------------------------------------------------------------------------------------------------------------------

#prepare vorticity fields = omega1, omega2, omega3
#input data are 5 vorticity fields and output data is vorticity fields 
#these size are (128, 128, 128)

#data preprocessing 
# 3D wavelet transformation
# pywt.dwtn(data, wavelet function, boundary condition)-> {'aaa':, 'aad':, ...}
wa1 = pywt.dwtn(omega1, 'coif2', mode='periodization')
aaa1 = np.array(wa1['aaa']).astype(np.float32)
wa2 = pywt.dwtn(omega2, 'coif2', mode='periodization')
aaa2 = np.array(wa2['aaa']).astype(np.float32)
wa3 = pywt.dwtn(omega3, 'coif2', mode='periodization')
aaa3 = np.array(wa3['aaa']).astype(np.float32)
#only use aaa 

#Input size is (, 5, 64, 64, 64, 3)
#Output size is (, 64, 64, 64, 3)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------
#Machine Learning Model

def create_model(model_ip):
   y1 = ConvLSTM3D(128, kernel_size=(3, 3, 3), activation='tanh', recurrent_activation='sigmoid', use_bias=False, padding='same', return_sequences=False, data_format='channels_last')(model_ip)
   y2 = tf.keras.layers.LayerNormalization(axis=[1, 2, 3], center=False, scale=True)(y1)
   y3 = Conv3D(48, kernel_size=(3, 3, 3), activation='linear', use_bias=False, padding='same')(y2)
   y4 = tf.keras.layers.LayerNormalization(axis=[1, 2, 3], center=False, scale=True)(y3)
   xout = Conv3D(n_com, kernel_size=(3, 3, 3), activation='linear', use_bias=False, padding='same')(y4)
   return Model(inputs=model_ip, outputs=xout)

#128, number of filters(channels) first middle layer
#48, number of filters(channels) third middle layer

n_reb = 64     #coefficients of wavelet
n_com = 3      #component of vorticity
n_in = 5       #number of input

lr_ip = Input(shape=(n_in, n_reb, n_reb, n_reb, n_com))
model = create_model(lr_ip)


#model check
model.summary()


#optimizer setting
#loss function, optimizer
gpu_count = 4
one_n_batch = 1

n_batch = one_n_batch * gpu_count
ep=200
multi_model = multi_gpu_model(model, gpus=gpu_count)

multi_model.compile(loss='mae',
              optimizer='Adam'
              )

cp = ModelCheckpoint("multi_weights.hdf5", monitor="val_loss", verbose=1,
                     save_best_only=True, save_weights_only=True)

#machine learning (epochs, batch_size)
history = multi_model.fit(I, O, epochs=ep, batch_size=n_batch, shuffle=True, callbacks=[cp], validation_split=0.2)

#load&save model
multi_model.load_weights("multi_weights.hdf5")
model.save_weights("weights.hdf5")

# Plot training & validation accuracy values
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('loss.png')


#test
test = np.array(test).astype(np.float32)
predict=model.predict(test)
predict=np.array(predict).astype(np.float32)

data=np.array(predict).reshape(n_reb, n_reb, n_reb, n_com)
print(data.shape)

aaa1 = []
aad1 = []
ada1 = []
add1 = []
daa1 = []
dad1 = []
dda1 = []
ddd1 = []
aaa2 = []
aad2 = []
ada2 = []
add2 = []
daa2 = []
dad2 = []
dda2 = []
ddd2 = []
aaa3 = []
aad3 = []
ada3 = []
add3 = []
daa3 = []
dad3 = []
dda3 = []
ddd3 = []

for i in range(len(data)):
   for j in range(len(data[0])):
      for k in range(len(data[0][0])):
         aaa1.append(data[i][j][k][0])
         aad1.append(0.0000)
         ada1.append(0.0000)
         add1.append(0.0000)
         daa1.append(0.0000)
         dad1.append(0.0000)
         dda1.append(0.0000)
         ddd1.append(0.0000)
         aaa2.append(data[i][j][k][1])
         aad2.append(0.0000)
         ada2.append(0.0000)
         add2.append(0.0000)
         daa2.append(0.0000)
         dad2.append(0.0000)
         dda2.append(0.0000)
         ddd2.append(0.0000)
         aaa3.append(data[i][j][k][2])
         aad3.append(0.0000)
         ada3.append(0.0000)
         add3.append(0.0000)
         daa3.append(0.0000)
         dad3.append(0.0000)
         dda3.append(0.0000)
         ddd3.append(0.0000)


aaa1 = np.array(aaa1).reshape(n_reb, n_reb, n_reb)
aad1 = np.array(aad1).reshape(n_reb, n_reb, n_reb)
ada1 = np.array(ada1).reshape(n_reb, n_reb, n_reb)
add1 = np.array(add1).reshape(n_reb, n_reb, n_reb)
daa1 = np.array(daa1).reshape(n_reb, n_reb, n_reb)
dad1 = np.array(dad1).reshape(n_reb, n_reb, n_reb)
dda1 = np.array(dda1).reshape(n_reb, n_reb, n_reb)
ddd1 = np.array(ddd1).reshape(n_reb, n_reb, n_reb)
aaa2 = np.array(aaa2).reshape(n_reb, n_reb, n_reb)
aad2 = np.array(aad2).reshape(n_reb, n_reb, n_reb)
ada2 = np.array(ada2).reshape(n_reb, n_reb, n_reb)
add2 = np.array(add2).reshape(n_reb, n_reb, n_reb)
daa2 = np.array(daa2).reshape(n_reb, n_reb, n_reb)
dad2 = np.array(dad2).reshape(n_reb, n_reb, n_reb)
dda2 = np.array(dda2).reshape(n_reb, n_reb, n_reb)
ddd2 = np.array(ddd2).reshape(n_reb, n_reb, n_reb)
aaa3 = np.array(aaa3).reshape(n_reb, n_reb, n_reb)
aad3 = np.array(aad3).reshape(n_reb, n_reb, n_reb)
ada3 = np.array(ada3).reshape(n_reb, n_reb, n_reb)
add3 = np.array(add3).reshape(n_reb, n_reb, n_reb)
daa3 = np.array(daa3).reshape(n_reb, n_reb, n_reb)
dad3 = np.array(dad3).reshape(n_reb, n_reb, n_reb)
dda3 = np.array(dda3).reshape(n_reb, n_reb, n_reb)
ddd3 = np.array(ddd3).reshape(n_reb, n_reb, n_reb)

y1 = {}
y1['aaa'] = aaa1
y1['aad'] = aad1
y1['ada'] = ada1
y1['add'] = add1
y1['daa'] = daa1
y1['dad'] = dad1
y1['dda'] = dda1
y1['ddd'] = ddd1
y2 = {}
y2['aaa'] = aaa2
y2['aad'] = aad2
y2['ada'] = ada2
y2['add'] = add2
y2['daa'] = daa2
y2['dad'] = dad2
y2['dda'] = dda2
y2['ddd'] = ddd2
y3 = {}
y3['aaa'] = aaa3
y3['aad'] = aad3
y3['ada'] = ada3
y3['add'] = add3
y3['daa'] = daa3
y3['dad'] = dad3
y3['dda'] = dda3
y3['ddd'] = ddd3

#invert wavelet transformation
predict1 = pywt.idwtn(y1, 'coif2', mode='periodization')
predict2 = pywt.idwtn(y2, 'coif2', mode='periodization')
predict3 = pywt.idwtn(y3, 'coif2', mode='periodization')



