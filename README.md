# Description 
Le but du projet, est de mettre en place un modèle de classification, permettant de catégoriser les radiographies pulmonaires en trois classes : 
- Normal 
- Pneumologie
- Covid19

Nous avons opté pour la mise en place d'un classificateur par le biais de modèles de Deep Learning pour la classification d'images.

Merci de noter que le dataset est disponible sur [Kaggle](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database). Nous en remercions les auteurs.

Nous remercions les équipes [Datascientest](https://datascientest.com/).

# Modélisation et Architecture des modèles
Nous avons mis en place différentes architectures basées sur les réseaux de neurones convolutifs (CNN), le transfert learning ainsi que les modèles hybrides (combinant l'extraction de features avec les modèles de machine learning standard). Nous les avons évalués avec les différentes métriques.

## Les modèles entraînés 

- Le modèle LeNet
- Un modèle personnalisé
- Un modèle de transfert learning EfficientNetB5
- Un modèle d'extraction de feautures combinant InceptionV3 et XGBoost

## Préprocessing des données

Lors de l'étape de la [data visualisation](https://github.com/sihamsaid/covid19-detection/blob/main/dataviz/datavisualisation.ipynb/), nous avons observé que l'application d'un traitement d'image peut atténuer le biais induit par la présence de bords noirs verticaux. L'idée sous-jacente, est de réaliser un «zoom» sur l'ensemble des radiographies. La solution retenue préconise l'élimination des bords latéraux, inférieurs et supérieurs, afin de se focaliser sur l'information utile, les poumons. 

Lors du traitement des données nous avons appliqué une transformation de 10% pour supprimer les bordures des images, de ce fait, nous avons procédé de deux manères différentes option 1 et option 2. 

### Option 1

Ici nous avons entraîné nos différents modèles sur ces nouvelles données transformées, cela a donné de très bons résultats sur le test et sur la validation. Voir [ici](https://github.com/sihamsaid/covid19-detection/blob/main/modeling/covid19_V1.ipynb) pour plus de détails sur l'implémentation.

### Option 2

L'idée ici est de voir le comportement de nos modèles sur différents types d'images, avec et sans bordures. Nous décidons d'appliquer la transformation de 10% uniquement sur le dataset du train, sans toucher au dataset du test. L'apprentissage des modèles se fera, donc, sur un dataset englobant les images originales(le X_train original) et les images transformées. Cette option a également montré de très bons résultats. 
Voir [ici](https://github.com/sihamsaid/covid19-detection/blob/main/modeling/covid19_V2.ipynb) pour plus de détails sur l'implémentation.



## Streamlit

Nous avons mis en place une application web, en utilisant [Streamlit]( https://www.streamlit.io/). Cette application permettra de choisir un modèle, de télécharger une radiographie et d'afficher le résultat de la classification. Voir [ici](https://github.com/sihamsaid/covid19-detection/blob/main/streamlit/covid19-streamlit.py) pour plus de détails sur l'implémentation.

L'utilisation de cette application, requiert l'exécution des notebooks (voir ci dessous), et de sauvegarder les modèles générés, dans le dossier [models](https://github.com/sihamsaid/covid19-detection/tree/main/streamlit/models) du dossier streamlit :

-  [Option 1](https://github.com/sihamsaid/covid19-detection/blob/main/modeling/covid19_V1.ipynb)  

- [Option 2](https://github.com/sihamsaid/covid19-detection/blob/main/modeling/covid19_V2.ipynb)


La commande qui lance notre application streamlit est : `streamlit run covid19-streamlit.py`
