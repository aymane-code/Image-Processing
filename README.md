# Interface de Traitement d'Images avec MATLAB

Ce projet est une application GUI basée sur MATLAB, conçue pour le traitement d'images. L’application propose une variété de filtres et de transformations interactifs, offrant une boîte à outils complète pour les étudiants et les professionnels souhaitant explorer et expérimenter les techniques de traitement d’images sous MATLAB.

## Fonctionnalités

### Gestion des Fichiers
- **Chargement et sauvegarde d'images** : Importez des images depuis votre appareil et sauvegardez les résultats directement via l'interface.

### Filtres Passe-Bas
Appliquez des filtres pour lisser les images et réduire le bruit :
- **Filtre Médian**
- **Filtre Moyenneur (3x3 et 5x5)**
- **Filtres Gaussiens** : Filtres de 3x3 et 5x5 pour un lissage plus doux.
- **Filtres Conique et Pyramidale** : Pour différents effets de lissage.

### Filtres Passe-Haut
Mettez en évidence les détails et les contours dans les images :
- **Filtre Laplacien**
- **Filtre Gradient**
- **Opérateurs de Sobel, Prewitt, et Roberts** : Détection des contours par des techniques classiques.
- **Filtre de Marr-Hildreth** : Pour des contours plus précis et détaillés.

### Transformations d'Images
- **Négatif** : Inversion des couleurs de l'image.
- **Ajustement de Contraste et de Luminosité** : Contrôle interactif des niveaux de contraste et de luminosité.
- **Binarisation** : Convertissez l'image en une version binaire.
- **Niveaux de Gris** : Transformation en échelle de gris.
- **Histogramme** : Visualisation de l'histogramme pour une analyse des intensités de l’image.

### Filtres dans le Domaine Fréquentiel
Transformez l'image en fréquence pour des modifications avancées :
- **Filtre Passe-Bas Idéal**
- **Filtre Passe-Bas de Butterworth**
- **Filtre Passe-Haut Idéal**
- **Filtre Passe-Haut de Butterworth**

### Ajout de Bruit
Simulez des bruits courants pour tester la robustesse des filtres :
- **Bruit Gaussien**
- **Bruit Poivre et Sel**

### Détection de Contours
Utilisez des méthodes avancées pour identifier les contours dans les images :
- **Contours Internes** : Extraction des contours internes d'une image.
- **Contours Externes** : Détection des contours externes pour mettre en valeur les objets.
- **Contours Morphologiques** : Techniques basées sur la morphologie mathématique pour identifier les contours.

### Transformée de Hough
Détectez des lignes et des cercles dans les images en utilisant la transformée de Hough :
- **Hough pour les Droites** : Identification et tracé des lignes dans l'image.
- **Hough pour les Cercles** : Détection des cercles en utilisant un accumulateur Hough.

## Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/username/ImageProcessingGUI.git
