import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow import keras
import pdb
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

""" Params """
n_hidden_layers = 6
w_hidden_layers = 80
epochs = 2000
batch_size = 512


""" Load data """
# with open(r'./data/training_data_complete.pkl', 'rb') as f:
#     data = pickle.load(f)
# X_raw = data['X']
# U_raw = data['U']
# P_raw = data['P']
# H = data['H']
#
# P_hvac = [np.reshape(u_raw[0, 0] - u_raw[1, 0], (1, 1)) for u_raw in U_raw]
# P_bat =  [np.reshape(u_raw[-1, 0], (1, 1)) for u_raw in U_raw]
# U = [np.hstack([p_hvac, p_bat]) for (p_hvac, p_bat) in zip(P_hvac, P_bat)]
#
# """ Scaling """
# x_lb = np.array([[20.0, 15.0,  0.0,     0.0]])
# x_ub = np.array([[23.0, 25.0, 50.0, 20000.0]])
# X_s = [(x.T - x_lb)/(x_ub - x_lb) for x in X_raw]
#
# u_lb = np.array([[-1000, -1000]])
# u_ub = np.array([[ 1000,  1000]])
# U_s = [(u - u_lb)/(u_ub - u_lb) for u in U]
#
# p_lb = np.array([[-10.0,    0.0]])
# p_ub = np.array([[ 30.0, 1200.0]])
# P_s = [(p - p_lb)/(p_ub - p_lb) for p in P_raw]
# T_s = [np.reshape(p[:, 0], (1, -1)) for p in P_s]
# SR_s = [np.reshape(p[:,1], (1, -1)) for p in P_s]
#
# data_in = []
# for x_s, t_s, sr_s in zip(X_s, T_s, SR_s):
#     data_in.append(np.hstack([x_s, t_s, sr_s]).reshape(1, -1))

X = np.load('data/input_complete.npy')
scaler = StandardScaler()
X = scaler.fit_transform(X)
pca = PCA(n_components=6)
X = pca.fit_transform(X)
np.save('data/input_pca', X)
y = np.load('data/output_complete.npy')

joblib.dump(pca, 'models/pca.gz')
joblib.dump(scaler, 'models/scaler.gz')

""" Build NN model """
inputs = keras.Input(shape=(X.shape[1],))
x = keras.layers.Dense(w_hidden_layers,activation='relu')(inputs)
for _ in range(n_hidden_layers-1):
    x = keras.layers.Dense(w_hidden_layers,activation='relu')(x)
outputs = keras.layers.Dense(y.shape[1],activation='linear')(x)

model = keras.Model([inputs], [outputs])
optimizer = keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-3)
model.compile(optimizer=optimizer, loss='mse')

early_stopping = keras.callbacks.EarlyStopping(patience=100)

# Train model
hist = model.fit(X,
                 y,
                 batch_size = batch_size,
                 epochs= epochs,
                 shuffle=True,
                 validation_split=0.2,
                 callbacks=[early_stopping])

# save model
model.save('./models/nn_controller_pca.h5')
