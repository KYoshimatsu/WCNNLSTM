import pywt
import numpy as np
import struct
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D
import matplotlib.pyplot as plt

#data loading-----------------------------------------------------------------------------------------------------------------------------------------------

#prepare vorticity fields = omega1, omega2, omega3
#input data are 1 vorticity fields and output data is vorticity fields
#these size are (128, 128, 128)


#data preprocessing
# subcube division-------------------------------------
n_div = 4        #number of division
IN=[]
OUT=[]
for i in range(len(O)):
   oa = np.split(O[i], n_div,  0)
   oa = np.array(oa)
   ia = np.split(I[i], n_div,  0)
   ia = np.array(ia)
   ob=[]
   ib=[]
   for j in range(n_div):
      ob.append(np.split(oa[j], n_div, 1))
      ib.append(np.split(ia[j], n_div, 1))
   ob = np.array(ob)
   ib = np.array(ib)
   oc=[]
   ic=[]
   for j in range(n_div):
      for k in range(n_div):
         oc.append(np.split(ob[j][k], n_div, 2))
         ic.append(np.split(ib[j][k], n_div, 2))
   oc=np.array(oc)
   ic=np.array(ic)
   for j in range(n_div*n_div):
      for k in range(n_div):
         OUT.append(oc[j][k])
         IN.append(ic[j][k])

#wavelet transformation------------------------------------------------------------
n_gp = 128 #grid points
n_reb = int(n_gp/n_div/2+5)     #coefficients of wavelet
n_wav = 8        #component of wavelet function & scaling function

Winput=[]
Woutput=[]
for x in range(len(IN)):
   wa = pywt.dwtn(IN[x], 'coif2', mode='symmetric')
   aaa = np.array(wa['aaa'])
   aad = np.array(wa['aad'])
   ada = np.array(wa['ada'])
   add = np.array(wa['add'])
   daa = np.array(wa['daa'])
   dad = np.array(wa['dad'])
   dda = np.array(wa['dda'])
   ddd = np.array(wa['ddd'])
   WInput=[]
   for k in range(len(aaa)):
      for j in range(len(aaa[0])):
         for i in range(len(aaa[0][0])):
            WInput.append(aaa[k][j][i])
            WInput.append(aad[k][j][i])
            WInput.append(ada[k][j][i])
            WInput.append(add[k][j][i])
            WInput.append(daa[k][j][i])
            WInput.append(dad[k][j][i])
            WInput.append(dda[k][j][i])
            WInput.append(ddd[k][j][i])
   WInput = np.array(WInput).reshape(n_reb, n_reb, n_reb, n_wav)
   Winput.append(WInput)
   wa = pywt.dwtn(OUT[x], 'coif2', mode='symmetric')
   aaa = np.array(wa['aaa'])
   aad = np.array(wa['aad'])
   ada = np.array(wa['ada'])
   add = np.array(wa['add'])
   daa = np.array(wa['daa'])
   dad = np.array(wa['dad'])
   dda = np.array(wa['dda'])
   ddd = np.array(wa['ddd'])
   WOutput=[]
   for k in range(len(aaa)):
      for j in range(len(aaa[0])):
         for i in range(len(aaa[0][0])):
            WOutput.append(aaa[k][j][i])
            WOutput.append(aad[k][j][i])
            WOutput.append(ada[k][j][i])
            WOutput.append(add[k][j][i])
            WOutput.append(daa[k][j][i])
            WOutput.append(dad[k][j][i])
            WOutput.append(dda[k][j][i])
            WOutput.append(ddd[k][j][i])
   WOutput = np.array(WOutput).reshape(n_reb, n_reb, n_reb, n_wav)
   Woutput.append(WOutput)

#machine learning model-----------------------------------------------------------------------------------------------------

weight_decay= 1e-4

model = tf.keras.Sequential([
Conv3D(32, kernel_size=(3, 3, 3), activation=tf.keras.layers.LeakyReLU(0.2), use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), padding='same', input_shape=(n_reb, n_reb, n_reb, n_wav)),
   Conv3D(32, kernel_size=(3, 3, 3), activation=tf.keras.layers.LeakyReLU(0.2), use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), padding='same'),
   Conv3D(32, kernel_size=(3, 3, 3), activation=tf.keras.layers.LeakyReLU(0.2), use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), padding='same'),
   Conv3D(32, kernel_size=(3, 3, 3), activation=tf.keras.layers.LeakyReLU(0.2), use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), padding='same'),
   Conv3D(8, kernel_size=(3, 3, 3), activation=tf.keras.layers.LeakyReLU(0.2), use_bias=True, padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
])

#optimizer & loss function
model.compile(loss='mean_squared_error',
              optimizer='Adam'
              )

n_batch=16
ep=30

#machine learning (epochs, batch_size)
cp = ModelCheckpoint("WCNN3d_weights.hdf5", monitor="val_loss", verbose=1,
                     save_best_only=True, save_weights_only=True)

history = model.fit(I1, O1, epochs=ep, batch_size=n_batch, shuffle=True,
callbacks=[cp], validation_split=0.2)

#load&save model
model.save_weights("WCNN3d_weights.hdf5")

# Plot training & validation accuracy values
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('loss_WCNN3d.png')

#predict-------------------------------------------------------


# inverse wavelet transformation
TEST=[]
for l in range(len(predict)):
   aaa = []
   aad = []
   ada = []
   add = []
   daa = []
   dad = []
   dda = []
   ddd = []
   for k in range(len(predict[0])):
      for j in range(len(predict[0][0])):
         for i in range(len(predict[0][0][0])):
            aaa.append(predict[l][k][j][i][0])
            aad.append(predict[l][k][j][i][1])
            ada.append(predict[l][k][j][i][2])
            add.append(predict[l][k][j][i][3])
            daa.append(predict[l][k][j][i][4])
            dad.append(predict[l][k][j][i][5])
            dda.append(predict[l][k][j][i][6])
            ddd.append(predict[l][k][j][i][7])
   aaa = np.array(aaa).reshape(n_reb, n_reb, n_reb)
   aad = np.array(aad).reshape(n_reb, n_reb, n_reb)
   ada = np.array(ada).reshape(n_reb, n_reb, n_reb)
   add = np.array(add).reshape(n_reb, n_reb, n_reb)
   daa = np.array(daa).reshape(n_reb, n_reb, n_reb)
   dad = np.array(dad).reshape(n_reb, n_reb, n_reb)
   dda = np.array(dda).reshape(n_reb, n_reb, n_reb)
   ddd = np.array(ddd).reshape(n_reb, n_reb, n_reb)
   y = {}
   y['aaa'] = aaa
   y['aad'] = aad
   y['ada'] = ada
   y['daa'] = daa
   y['add'] = add
   y['dad'] = dad
   y['dda'] = dda
   y['ddd'] = ddd
   x = pywt.idwtn(y, 'coif2', mode='symmetric')
   x = np.array(x).astype(np.float32)
   TEST.append(x)

TEST = np.array(TEST).reshape(n_div*n_div*n_div,n_com,int(n_gp/n_div),int(n_gp/n_div),int(n_gp/n_div)).transpose(0, 2, 3, 4, 1)

#merging subcubes
data0 = TEST
data1=[]
for j in range(n_div*n_div):
   data1.append(np.concatenate([data0[0+j*n_div], data0[1+j*n_div], data0[2+j*n_div], data0[3+j*n_div]], 2))

data1 = np.array(data1)
data2=[]
for j in range(n_div):
   data2.append(np.concatenate([data1[0+j*n_div], data1[1+j*n_div], data1[2+j*n_div], data1[3+j*n_div]], 1))

data2=np.array(data2)
print(data2.shape)

data=np.concatenate([data2[0], data2[1], data2[2], data2[3]], 0)
