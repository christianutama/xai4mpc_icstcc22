import numpy as np
import tensorflow as tf
import shap
import joblib


model = tf.keras.models.load_model('models/nn_controller_complete.h5')
X = np.load('data/input_complete.npy')
y = np.load('data/output_complete.npy')

explainer = shap.KernelExplainer(model.predict, shap.kmeans(X, 10))
shap_values = explainer.shap_values(X)
joblib.dump(shap_values, 'results/shap_values')