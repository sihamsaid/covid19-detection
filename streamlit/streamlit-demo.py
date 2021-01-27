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

# Encoding des trois classes
ENCODING = {'COVID-19': 0, 'NORMAL': 1, 'Viral Pneumonia': 2}

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


uploaded_file = st.file_uploader("Télécharger votre radiologie", type=['png','jpeg', 'jpg'])
if uploaded_file is not None:
     image = plt.imread(uploaded_file)
     image = im_resize_256(image)
     image = image.reshape(256, 256, 1)
     st.image(image)


df = pd.DataFrame({
  'Nom du Modèle': ["CNN Architecture 1", "CNN Architecture 2"],
  'Description': ["Architecture CNN Personnalisée", "Architecture CNN Personnalisée"],
  'Chemin':['cnn_architecture_1', 'cnn_architecture_2']
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

# Afficher la loss
loss_image = plt.imread(f"{chemin}\loss.png")
st.image(loss_image) 

# Afficher la matrice de confusion
confusion_image = plt.imread(f"{chemin}\matrice_confusion.png")
st.image(confusion_image) 

# Afficher le rapport de classification
classification_report = open(f"{chemin}\classification_report.txt").read()
st.text(classification_report)


# Load le modèle Keras stocké
if st.button('Lancer la Prédiction'):
    model = tf.keras.models.load_model(f"{chemin}")
    #model.summary()
    if image is not None:
        # Mettre l'image dans un numpy array
        array = np.asarray([image])
        # Nous devons aussi normaliser 
        array = array / 255
        classes = model.predict(array)
        # pour rappel on a l'enconding suivant : {'COVID-19': 0, 'NORMAL': 1, 'Viral Pneumonia': 2}
        #classes
        # Récupérer l'index de la classe
        index = np.argmax(classes[0], axis=0)
        # calculer le pourcentage
        valeur = classes[0][index]
        valeur = round(valeur*100, 2)
        if index == 0:
            st.error(f'Radiologie de COVID à {valeur}%')
        elif index == 1:
            st.write(f'Radiologie Normale à {valeur}%')
        else:
            st.warning(f'Radiologie de Pneumo à {valeur}%')
    else:
        st.warning("Aucune image sélectionée")
   

