# Facial recognition
## Présentation du projet
### Exposition de la proposition
Le but de ce projet est de développer un système de connexion par reconnaissance faciale. 

### Périmètre fonctionnel
Tout d’abord, nous avons intégré la possibilité pour un utilisateur de créer un nouveau compte. L’administrateur peut donner un identifiant à une personne, qui peut ensuite créer un profil en entrant cet identifiant. Le nouvel utilisateur est alors photographié avec sa webcam, puis les photos sont sauvegardées afin de pouvoir se connecter par la suite à l’aide de la reconnaissance faciale.

Ensuite, nous avons ajouté un système de connexion sécurisée par reconnaissance faciale. L’utilisateur doit entrer son identifiant, et le système vérifie si la personne derrière la caméra est bien celle qui essaie de se connecter. 


### Choix techniques (matériels et logiciels)
Ci-dessous une liste du matériel et des logiciels utilisés pour mener à bien notre projet.

- Nous avons évolué sous différents environnements au sein de l’équipe, à savoir Windows et Linux
- Nous avons tout développé en python, pour des soucis de compatibilité entre les différentes parties.
- Pour la reconnaissance faciale, nous avons utilisé les bibliothèques Keras et Tensorflow pour créer les différents réseaux de neurones et les entraîner. Nous nous sommes servis de OpenCV pour manipuler les images, et de nombreux autres modules comme Numpy pour les tableaux, ou MatPlotLib pour les différents graphiques.



## Analyse fonctionnelle
### Détail des fonctionnalités
Notre solution finale comprend les fonctionnalités suivantes.

- **Login par reconnaissance faciale.** L’utilisateur entre son identifiant. Le visage est ensuite comparé à l’identifiant afin de détecter s’il correspond ou non. S’il correspond, l’utilisateur est redirigé vers la page d’accueil du site internet.
- **Création de compte.** Un nouvel utilisateur a la possibilité de créer un compte sur le site. Si l’identifiant entré par l’utilisateur est dans une base de données d’identifiants d’élèves, le compte est créé. Sinon, l’utilisateur ne peut pas créer son compte. Le compte ne peut pas être créé deux fois pour des raisons de sécurité.


### Maquettes
Voici les aperçus de la solution que nous avons développé.

<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/Aspose.Words.e148c4a0-0357-478c-95de-9f103658d36e.002.png>

*Page de création de compte*

<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/Aspose.Words.e148c4a0-0357-478c-95de-9f103658d36e.003.png>

*Page de login au site internet*



## Réalisation reconnaissance faciale

L’Intelligence Artificielle, est une science dans laquelle on essaie de permettre à une machine de reproduire des tâches réalisables par l’Homme. L’Intelligence Artificielle se compose de plusieurs domaines, dont un des principaux qui est le Machine Learning.

Le Machine Learning, appelé apprentissage automatique, est défini par un de ses pionniers Arthur Samuel comme une discipline qui vise à développer des algorithmes donnant à un ordinateur la capacité d’apprendre, plutôt que de le programmer de façon explicite.

Le DeepLearning, ou apprentissage profond, est une sous-catégorie du Machine Learning dans laquelle on utilise des réseaux neuronaux. A l’origine inspiré des neurones biologiques, les algorithmes de réseaux de neurones sont capables de réaliser n’importe quelle tâche. 


Ainsi, un réseau de neurones peut être capable de jouer aux échecs, de classifier différentes images, ou encore de reconnaître un chiffre écrit à la main. Les applications sont vastes, et l’une d’entre elles est la Computer Vision (vision par ordinateur), elle consiste à « donner des yeux » à un ordinateur, c’est-à-dire à lui apprendre à voir et interpréter le contenu d’une image. Dans le cadre de notre projet, notre objectif est d’apprendre à un programme à reconnaître une personne sur une image, c’est donc le principe de Computer Vision qu’on met utilise. 

Pour cela, nous avons décidé de créer nous-mêmes l’architecture de nos différents modèles. En Machine Learning, un modèle est un algorithme qui apprend à réaliser une tâche spécifique donnée à partir d’un ensemble de données, appelé dataset. L’objectif est de donner un ensemble d’entraînement au modèle (trainset), et de lui donner la réponse qu’il doit sortir, afin qu’il puisse apprendre à donner la bonne sortie. Ensuite, on teste le modèle sur un testset, pour voir s’il est capable de prédire une sortie en connaissant uniquement l’entrée. 

Le dataset est une partie fondamentale en Machine Learning, car la machine apprend à l’aide de nombreuses données, et au plus les données sont de qualité au mieux elle pourra apprendre. Nous avons donc décidé de créer nous-mêmes notre dataset, en prenant des photos avec la webcam de nombreuses personnes de l’ISEN, et on a également récupéré des datasets sur Internet pour avoir des données en plus grande quantité. On a utilisé le dataset LFW (Labeled Faces in the Wild) et le CelebA.

La complexité de notre projet repose donc sur le fait que nous créons tout nous-mêmes, de l’architecture de nos modèles à l’entraînement de ces derniers, en passant par la création d’une partie du dataset d’entraînement.

*Echantillon des datasets utilisés*


|**LFW**|**CelebA**|
| - | - |
| <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/LFW.png width=200 height=200> | <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/celebA.png width=200 height=200> |


La Reconnaissance Faciale peut se décomposer en deux exercices. Tout d’abord, il faut extraire les visages présents sur une image. C’est la phase de détection. Vient ensuite la phase de reconnaissance, dans laquelle le but est de déterminer à qui le visage extrait appartient. Pour ce faire, nous réalisons ce qu’on appelle un pipeline. Un pipeline est, en deep-learning, une série de transformations auxquelles on va soumettre une image. 

Regardons en détail les modèles de reconnaissance et de détection.



## Modèles de détection
#### Première proposition
Le but du modèle de détection est d’extraire les différentes têtes présentes sur une photo. Pour cela, nous avons testé différents modèles de réseaux de neurones convolutifs (cnn). Les réseaux de neurones convolutifs sont un type de réseau de neurones particulièrement adapté aux images. Il fonctionne en appliquant des filtres successifs sur l’image d’entrée.

La première méthode que nous avons implémentée se base sur le principe de la fenêtre glissante. La tâche de détection est divisée en deux tâches plus simples :

- L’extraction de zones de l’image
- La classification de ces zones en deux catégories : tête ou non

Pour la première partie, nous utilisons une fenêtre glissante : Toutes l’image est parcourue par une plus petite fenêtre afin d’extraire suffisamment de zones à envoyer pour la classification. 

Cependant, avec cette méthode, une seule et même tête était détectée plusieurs fois par notre algorithme, créant des superpositions de propositions comme sur la figure ci-dessous. Il a alors fallu mettre en place un algorithme de non-max-suppression afin de ne garder que les meilleures régions. Si une région chevauche une autre région plus qu’un certain seuil (par exemple 30%), seule la région la mieux notée est conservée.

|**Avant la nonMaxSuppression**|**Après la nonMaxSuppression**|
| - | - |
| <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/nnms1.png width=200 height=200> | <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/nms1.png width=200 height=200> |
| <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/nnms2.png width=200 height=250> | <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/nms2.png width=200 height=250> |
| <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/nnms3.png width=200 height=250> | <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/nms3.png width=200 height=250> |


Nous n’avons cependant pas retenu cette solution de fenêtre glissantes en raison de sa lenteur. Effectivement, il faut compter entre 1.2 et 1.4 secondes pour détecter les images sur la photo. Une telle lenteur s’explique en particulier par les nombreuses sous-images à classifier (il faut plus de 0.5s pour classifier les 324 sous-images ici), et l’algorithme de non-max suppression, qui prend entrer 0.1 et 0.15 secondes.

###### *FACE CLASSIFIER*  
Voici l’architecture utilisée pour le modèle de classification. Le principe est de réduire progressivement la dimension de l’image grâce à des couches de max-Pooling et des couches de convolutions. Ensuite, la décision en tant que telle est prise par les trois couches denses. Les deux perceptrons de la dernière couche valent chacun la probabilité que l’image soit une tête ou non.

<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/model0.png>  

  
###### *SLIDING WINDOWS*  
En résumé, le principe de la sliding windows est expliqué ci-dessous : Une fenêtre parcourt l’image pour la découper en sous-images. Cet ensemble d’images est ensuite donné au classifier, puis au non-max-suppresseur. Il est ainsi possible de déterminer l’emplacement d’une tête.

<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/pika1.png>

#### Deuxième proposition.
La deuxième solution choisie fut de créer un modèle - nommé Region Proposal Network (RPN) – qui détermine en une seule fois l’emplacement des têtes sur une image. 

L’idée première était d’entrainer un modèle à proposer des régions d’intérêt susceptibles de contenir une tête, pour ensuite envoyer ces données au classifier et ainsi passer de 324 sous-images à une petite dizaine. Cependant, le modèle s’est avéré plus efficace que prévu, si bien que nous avons pu enlever le classifier derrière et ne garder que ce RPN. Nous avons entrainé ce modèle quatre jours complets sur la workstation du club Info de l’ISEN, sur beaucoup de données (plus de 25000 images). Cette quantité de données nous a permis d’éviter l’overfitting (c’est-à-dire le surapprentissage). 

Effectivement, lorsqu’un modèle est trop complexe ou n’est pas entrainé sur beaucoup de données, il a tendance à apprendre trop précisément les données d’entraînement, en s’ajustant parfaitement sur celles-ci, et n’est donc plus capable de généraliser sur les données de test qu’il n’a jamais vu auparavant.

Dans notre cas, les données d’entrainement contenaient, en entrée, un dataset de photo de personnes, et en sortie un masque des têtes, c’est-à-dire une image noire avec un carré blanc à l’emplacement de la tête. Notre RPN a donc appris à extraire un tel masque sur une image quelconque.

Un rapide traitement de la sortie du modèle (détection des contours, récupération des coordonnées limites) permet ensuite d’extraire précisément les têtes de l’image.

Cette méthode étant plus précise et plus rapide (il faut de 0.14 à 0.17s pour traiter une image), c’est celle que nous avons décidé de conserver.



|*Image fournie au modèle*|*Sortie du modèle*|*Traitement de la sortie*|
| :-: | :-: | :-: |
| <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/rpn1.png> | <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/rpn2.png> | <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/rpn3.png> |






###### *REGION PROPOSAL NETWORK*
Ce réseau utilise une architecture de type U-net. C’est-à-dire qu’il va progressivement diminuer la taille de l’image grâce à des max-pooling, puis réaugmenter la taille de l’image par paliers. La spécificité de ce réseau est qu’il possède des connexions résiduelles entre les images encodées et les images décodées de la même taille. Ces connexions permettent de ne pas perdre d’informations par la compression de l’image et ainsi d’obtenir des frontières bien délimitées. 

<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/model1.png>
## Modèles de reconnaissance
Concernant la partie de reconnaissance, nous avons également expérimenté plusieurs modèles avant d’obtenir un modèle suffisamment performant.

La reconnaissance faciale repose en grande partie sur un modèle – que nous appellerons l’encoder – qui apprend à transformer une image d’un visage en un vecteur représentatif de ce visage.

Si l’encoder fonctionne bien, il suffit de calculer la distance euclidienne entre notre vecteur et les vecteurs des visages dont on connait l’identité. La distance sera minimisée par la même personne. Autrement dit, une personne a des vecteurs assez similaires.

Le problème majeur avec ce principe de reconnaissance survient dans l’entrainement de l’encoder. Effectivement, ce n’est pas de l’apprentissage supervisé : nous ne connaissons pas les vecteurs à l’avance pour entrainer le modèle dessus

###### AUTOENCODER
La première idée que nous avons eue pour entrainer un tel encoder fut d’entrainer un auto-encoder et d’en extraire l’encoder. Un auto-encoder est un modèle qui apprend à compresser une image en un vecteur de dimension n, puis à décompresser cette image pour en retrouver l’originale. Il est entrainé avec, en entrée, une image, et en sortie la même image. C’est la spécificité du modèle.  

Ainsi, un tel auto-encoder apprend à transformer une image en un vecteur suffisamment significatif pour pouvoir reconstruire l’image avec. Une idée intuitive de ce vecteur est qu’il correspond à une liste de caractéristiques spécifiques (couleur des yeux, forme de la mâchoire, lunettes ou non, etc.).  

Nous utilisons en entrée des images de taille 128\*128\*3 et au centre un vecteur de taille 100. Il y a donc une compression par un facteur 491. Cette architecture permet donc de construire et entrainer un encoder qui transforme une image en un vecteur plein de sens. Il est naturel de penser qu’une même personne a des vecteurs encodés grâce à ce modèle assez similaires  

<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/model2.png>  
Cependant, le vecteur encodé ne sera pas spécifique à la tâche de reconnaissance et prête trop attention aux informations non nécessaires pour la détection mais importantes pour la reconstitution de l’image, comme l’orientation du visage. Effectivement, l’orientation du visage est importante pour pouvoir reconstruire le visage fidèlement, mais pas nécessaire pour la reconnaissance du visage.  

Voyons quelques exemples de sorties de l’autoencoder entrainé. 

*Ici, le vecteur latent (c’est-à-dire le vecteur central) a une dimension de 64. Il y a donc compression de l’image d’entrée par un facteur 768.*

|**Entrée de l’autoencoder**|**Sortie de l’autoencoder**|
| - | - |
| <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/face1.png> | <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/face2.png> |
| <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/face3.png> | <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/face4.png> |
| <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/face5.png> | <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/face6.png> |
| <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/face7.png> | <img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/face8.png> |


La compression affecte fortement la qualité de l’image de sortie, mais l’information générale de position du visage, la forme de la bouche, la place des yeux, etc. est conservée.  




###### SIAMESE NETWORK
Le deuxième modèle que nous avons implémenté est un « siamese network » ou « réseau siamois ». Son nom provient des deux entrées en parallèle qu’il possède. Les deux images en entrée sont encodées avec le même encoder, et on calcule ensuite la distance euclidienne entre les deux vecteurs encodés. 

Lors de la phase d’entraînement, on donne en entrée soit deux images de deux personnes identiques, soit deux différentes, et le but est de minimiser l’erreur, en prédisant des distances petites pour deux images de la même personne, et à l’inverse des distances grandes lorsque ce sont deux personnes différentes. Lors de cette phase, les poids de l’encoder sont ajustés afin de minimiser la fonction coût et donc de répondre au mieux à cette tâche.  

<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/model3.png>  


###### MODEL FACENET

Après avoir testé le siamese network, nous avons décidé d’utiliser une nouvelle architecture afin d’obtenir de meilleurs résultats, le modèle FaceNet. Le principe de ce modèle est d’avoir non plus 2 entrées, mais cette fois 3 entrées. Une entrée anchor, c’est-à-dire une image d’une personne choisie au hasard qui est notre image de référence, puis une image de la même personne (entrée dite « positive »), et une image d’une personne différente (entrée « négative »). 

Comme pour le siamese network, on utilise le même encoder pour encoder les différentes images d’entrée. On calcule ensuite l’erreur de notre modèle en utilisant un « triplet loss ». Cette fonction coût donne 0 si la distance entre l’anchor et le positive est suffisamment inférieure comparée à la distance entre l’anchor et le négative. Ainsi, lors de l’entraînement, le modèle apprend à prédire de grandes distances lorsque les personnes sont différentes, et de petites distances dans le cas contraire. La force de ce modèle réside dans le fait qu’il minimise la distance entre deux personnes identiques en même temps qu’il maximise la distance entre deux personnes différentes.

Ce modèle est entrainé avec trois images en entrée, et 0 en sortie, quelles que soient les images.  

<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/model4.png>  

Afin d’optimiser les performances de notre modèle, nous avons testé de nombreuses architectures d’encoders sur notre modèle FaceNet. L’encoder qu’on a choisi finalement est « l’encoder bêta », qui nous a offert les meilleurs résultats en termes de précision (les résultats de chaque modèle sont indiqués dans le tableau récapitulatif) et de rapidité.

Les architectures que nous avons essayées sont décrites ci-dessous.


###### ALPHA ENCODER
Cet encoder est assez prometteur bien qu’en léger overfitting. La factorisation de la couche conv 5\*5 en deux couches conv 5\*1 puis 1\*5 permet de réduire grandement le nombre de paramètres (10 au lieu de 25), et donc la rapidité d’entrainement.  

<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/model5.png>  


###### XCEPTION ENCODER
Entrainer un modèle complexe sur beaucoup de données prend énormément de temps. C’est pourquoi nous avons essayé d’utiliser un modèle pré-entrainé (le modèle Xception de google). Le modèle a été entrainé sur une tâche de classification d’images. Nous avons donc repris ce modèle et les valeurs de ses paramètres, et avons ensuite « gelé » la moitié du modèle et entrainé l’autre moitié (en partant depuis les poids pré entrainés).

Ce modèle a fortement overfitté le dataset car il y a trop de paramètres (près de 8 millions de paramètres entrainés et 14 millions de paramètres non entrainables). De plus, les performances sont décevantes car la tâche sur laquelle a été entrainé le Xception est trop éloigné de notre tâche d’encodage.   



<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/model6.png>  


###### BÊTA ENCODER
C’est avec cet encoder qu’on obtient les meilleurs résultats. Séparer les données en entrée du bloc bêta permet d’obtenir un point de vue différent de l’entrée. Les trois sorties de convolution sont concaténées. Cela permet de ne pas perdre de données, mais augmente le nombre de paramètres. Pour compenser cet ajout de paramètres, on utilise une couche de Batch normalisation et une couche de Dropout (pour limiter l’overfitting).  

<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/model7.png>  




###### ENCODER + SKLEARN CLASSIFIER
Afin d’améliorer les performances de notre modèle, nous avons essayé d’ajouter en sortie de l’encoder un modèle de classification de la librairie Scikit-Learn (une librairie qui propose de nombreux modèles de machine learning). Effectivement, la distance euclidienne (ainsi que la similarité cosinus) donne autant d’importance à toutes les valeurs du vecteur encodé de l’image. Nous sommes donc partis du principe que faire une distance pondérée pouvait augmenter les résultats de l’encoder. Nous avons donc utilisé un randomForestClassifier (afin de tirer parti de sa non-linéarité) avec 100 estimateurs - valeur pour laquelle nous avions un bon compromis entre le biais et la variance. Nous l’avons ainsi entrainé à déterminer si deux vecteurs correspondent à la même personne, en fonction de leur différence. Avec cette méthode, nous avons obtenu le meilleur score sur l’ensemble de test en atteignant 91% de précision.

Cependant, le modèle était lourd en calculs et donc trop long pour pouvoir l’implémenter sur le site, c’est pourquoi nous avons finalement conservé l’architecture Bêta, afin d’avoir un bon compromis précision-rapidité.


<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/model8.png>




###### RESULTATS

Les modèles présentés ci-dessus donnent les résultats suivants.  
La performance (accuracy) des models est calculée de la façon suivante : __Accuracy = 100*[y = (y'>treshold)]/N__  


y : True value 0 or 1  
y' : Predicted value (float)  
treshold : Chosen value (float)  


|**Models**|**Accuracy (%)**|**Treshold**|
| - | - | - |
|**autoencoder**|63.69|1.58|
|**siamese**|64.60|0.31|
|**faceNet Xception**|68.29|3.43|
|**faceNet alpha**|69.96|1.10|
|**faceNet simple encoder**|71.39|3.93|
|**faceNet bêta**|77.25|2.03	|
|**encoder+sklearn**|91.33|-|

Il est intéressant de noter l’impact de l’architecture d’un modèle sur sa performance. C’est pourquoi nous avons essayé tant de modèles différents, pour finalement aboutir à la performance remarquable du modèle encoder+scikit-learn.


### VISUALISATION

Concrètement, que fait l’encoder ? L’encoder, attribue un vecteur représentatif à chaque visage. On a donc des « groupes » de points correspondants à des personnes différentes. Il est possible de visualiser ces groupes en réduisant la dimension du vecteur encodé à 2, et en affichant ainsi les points sur un plan. Pour réduire la dimension, nous utilisons un PCA (principal component analisis) de scikit-learn.

Chaque point correspond à une image de visage, et chaque couleur à une personne. On remarque ainsi que les vecteurs correspondants à une même personne sont assez proches. L’utilisation de la distance euclidienne pour comparer deux personnes prend alors tout son sens.

<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/viz1.png>


### CONCLUSION

Notre système s’appuie donc, pour conclure, sur une pipeline de deep-learning que voici ci-dessous.

<img src=https://github.com/Prevost-Guillaume/Facial-recognition/blob/main/images/pika2.png>  

2020/2021 – ISEN 3
