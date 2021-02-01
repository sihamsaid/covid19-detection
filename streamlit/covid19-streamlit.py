
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import cv2 
import io

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras import utils

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import load_model
# On utilise joblib pour charger les modèles hybrides inception
from joblib import load

from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import preprocess_input


# Encoding des trois classes
ENCODING = {'COVID-19': 0, 'NORMAL': 1, 'Viral Pneumonia': 2}
IMAGES_TYPES = list(ENCODING.keys())

st.title("Prédire la classe d'une radiologie")

def im_resize_256(img):
    """
    Cette fonction lit une image et retourne la même image avec une résolution de 256 X 256
    """
    if img.shape == (256,256):
        img_ret = img
    else :
        img_ret = cv2.resize(img, dsize = (256,256))
    return img_ret

def predict_inception(array):
    # Charger le modèle inception avce keras
    model = tf.keras.models.load_model("./models/InceptionV3")
    intermediate_layer_model = Model(model.input, model.layers[2].output)
    X_feature = intermediate_layer_model.predict(preprocess_input(array))


    # Charger le modèle XGBoost avec joblib
    xgboost = load("./models/InceptionV3/xgb.joblib")

    return xgboost.predict(X_feature)
    
    
df = pd.DataFrame({
  'Nom du Modèle': ["LeNet", "Perso 1", "EfficientNetB5", "InceptionV3"],
  'Description': ["CNN LeNet", "CNN Personnalisé 1", "Transfert Learning EfficientNetB5", "Extraction Features InceptionV3"],
  'Chemin':['LeNet', 'Perso 1', 'EfficientNetB5', 'InceptionV3']
 })



option = st.selectbox('Veuillez selectionner un modèle', df['Nom du Modèle'])

'Vous avez selectionné le modèle: ', option

df_model = df[df['Nom du Modèle'] == option]
#df_model
st.write("Ci-dessous un résumé de l'architecture selectionnée ainsi que les différentes métriques obtenues lors du training")

# Chemin vers l'architecture
chemin = f".\models\{df_model['Chemin'].values[0]}"

# Afficher le summary du modèle
model_summary = open(f"{chemin}\summary.txt").read()
st.text(model_summary)

# Afficher la loss accuracy
loss_image = plt.imread(f"{chemin}\loss_accuracy.png")
st.image(loss_image) 

# Afficher la matrice de confusion
confusion_image = plt.imread(f"{chemin}\confusion_matrix.png")
st.image(confusion_image) 

# Afficher le rapport de classification
classification_report = open(f"{chemin}\classification_report.txt").read()
st.text(classification_report)


# Load le modèle Keras stocké

uploaded_file = st.file_uploader("Télécharger votre radiologie", type=['png','jpeg', 'jpg'])
if uploaded_file is not None:
    image = plt.imread(uploaded_file)
    image = im_resize_256(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    st.image(image)

    # Mettre l'image dans un numpy array.
    array=np.array(([image]))

    if "InceptionV3" in chemin:
        classes = predict_inception(array)
        #st.write(classes[0])
        image_class = classes[0]
        if image_class == 0:
            st.error('Radiologie de COVID')
        elif image_class == 1:
            st.write('Radiologie Normale')
        else:
            st.warning('Radiologie de Pneumo')
    else:
        model = tf.keras.models.load_model(f"{chemin}")
        classes = model.predict(array)
  
        # Récupérer l'index de la classe
        index = np.argmax(classes[0], axis=0)
        # calculer le pourcentage
        valeur = classes[0][index]
        valeur = round(valeur*100, 2)
        #st.write(array)
        df_prediction = pd.DataFrame(data=classes[0], index = IMAGES_TYPES)
        df_prediction = df_prediction.rename({'0': 'Prédiction'})
        st.write(df_prediction)
        if index == 0:
            st.error(f'Radiologie de COVID à {valeur}%')
        elif index == 1:
            st.write(f'Radiologie Normale à {valeur}%')
        else:
            st.warning(f'Radiologie de Pneumo à {valeur}%')
   

