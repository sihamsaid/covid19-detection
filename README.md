# covid19-detection
Ce projet rentre dans le cadre de la detection du covid19 en se basant sur les radiologies des patients.

 
# Description 
Le but est la mise en place d'un modèle de classification, permettant de catégoriser les radiographies pulmonaires en trois classes : 
- Normal 
- Pneumologie
- Covid19

Nous avons opté pour la mise en place d'un classificateur par le biais de modèles de Deep Learning pour la classification d'images.


# Modélisation et Architecture des modèles
Nous avons mis en place différentes architectures basées sur les réseaux convolutifs (CNN), de transfert learning ainsi que les modèles hybrides (feauture extraction + modèles de machine learning standard). Nous les avons évalués avec les différentes métriques.

## Les modèles entraînés 

- Le modèle LeNet
- Le modèle personnalisé
- Un modèle de transfert learning EfficientNetB5
- Un modèle d'extraction de feautures combinant InceptionV3 et XGBoost

## Préprocessing des données

L'application d'un traitement d'image peut atténuer le biais induit par la présence de bords noirs verticaux. L'idée sous-jacente, est de réaliser un «zoom» sur l'ensemble des radiographies. La solution retenue préconise l'élimination des bords latéraux, inférieurs et supérieurs, afin de se focaliser sur l'information utile, les poumons. 
Lors du traitement des données où on appliqué une transformation de 10% pour supprimer les bordures des images, de ce fait on a procédé ça a été fait sur l'ensemble des données brutes, c'est sur ces nouvelles données transformées qu'on a entraîné nos différents modèles et qui ont donné de très bons résultats sur le test et sur la validation.

Néanmoins, nous avons mis en place pendant ce week end, ce qu'on a appelé une option 2 qui consiste en l'application de la transformation que sur le data set du train, sans y toucher au dataset du test. Nous décidons de faire l'apprentissage des modèles sur un dataset englobant les images originales(le X_train original) et les images transformées.(transformation effectuée sur X_train). L'idée ici est de voir le comportement de notre algo sur de types d' images, sans et avec bordures.


On avons fait tourner l'integralité de nos modèles sur les deux options, et dans les deux cas ça donne de bons résultats
