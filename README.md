"# Image-Processing" 
Alae Bouchiba Groupe 2/2

Miniprojet :Traitement d’image et vision

Exercice 1:éliminationde bruit :

Dans cet exercice,onest demandé d'éliminer le bruit quiest sous forme de traits noirs :

import numpy as np

import cv2

from matplotlib import pyplot as plt

- lecture de l'image

image = cv2.imread(r'C:\Users\Alae\Downloads\liftingbodybruite.png',0)

- transformation discrète de fourier de l'image

dft = cv2.dft(np.float32(image),flags = cv2.DFT\_COMPLEX\_OUTPUT)

- shift du zero au centre du spectrum dft\_shift = np.fft.fftshift(dft)

  rows, cols = image.shape

crow,ccol = rows//2 , cols//2

mask = np.zeros((rows,cols,2),np.uint8) mask[crow-70:crow+70, ccol-70:ccol+70] = 1

- application du mask et inverse DFT

fshift = dft\_shift\*mask

f\_ishift = np.fft.ifftshift(fshift)

img\_back = cv2.idft(f\_ishift)

img\_back = cv2.magnitude(img\_back[:,:,0],img\_back[:,:,1])

- visualization de l'image plt.subplot(121),plt.imshow(image, cmap = 'gray') plt.title('Input Image'), plt.xticks([]), plt.yticks([]) plt.subplot(122),plt.imshow(img\_back, cmap = 'gray') plt.title('Image sans bruit'), plt.xticks([]), plt.yticks([]) plt.show()

  cv2.waitKey(0)

- Pour ce faire,Nous verrons d'abordcomment trouver la transformée de Fourier enutilisant Numpy.Numpya unpackage FFTpour le faire.np. t. t2()nous fournit la transformée de fréquence quisera untableaucomplexe.Sonpremier argument est l'image d'entrée,quiest en niveauxde gris.Le deuxième argument est facultatifet décide de la taille dutableaude sortie.Maintenant,une fois que nous avons obtenule résultat,le composant de fréquence zéro (composant DC)sera dans le coinsupérieur gauche.Sinous voulons le centrer,nous devons décaler le résultat de \frac{N}{2}dans les deuxsens.quiest fait par la fonction,np. t. tshift().
- Après avoir trouvé la transformée de fréquence.Onpeut e ectuer certaines opérations dans le domaine fréquentiel,comme le filtrage passe-haut et reconstruire l'image,c'est-à-dire trouver la DFTinverse.Pour cela ilnous su t de supprimer les basses fréquences enles masquant avec une fenêtre rectangulaire de taille 70x70(comme utilisé dans notre cas).Onapplique ensuite le décalage inverse à l'aide de np. t.i tshift()afinque la composante DCrevienne dans le coin supérieur gauche.onTrouve ensuite la FFTinverse à l'aide de la fonctionnp. t 2().Ainsi,on obtient après a chage,notre image sans bruit.

![](Aspose.Words.60199179-39ec-417f-b31e-6be323be9e97.001.jpeg)

Remarque :sioncherche à a cher le spectrumde magnitude,ilsu t d’ajouter les lignes de code suivantes :

dft\_shift =np. t. tshift(dft) magnitude\_spectrum=20\*np.log(cv2.magnitude(dft\_shift[:,:,0],dft\_shift[:,:,1]))

plt.subplot(121),plt.imshow(image,cmap='gray')

plt.title('Input Image'),plt.xticks([]),plt.yticks([]) plt.subplot(122),plt.imshow(magnitude\_spectrum,cmap='gray') plt.title('Magnitude Spectrum'),plt.xticks([]),plt.yticks([]) plt.show()

![](Aspose.Words.60199179-39ec-417f-b31e-6be323be9e97.002.jpeg)

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* \*\*\*\*\*\*\*\*\*\*\*

Exercice 2 :éliminationde bruit et sauvegarde résultat

Dans cette partie,onest demandé d'éliminer le bruit et de sauvegarder le résultat dans unfichier

import cv2 as cv

import numpy as np

from matplotlib import pyplot as plt from scipy import ndimage

import matplotlib.image as mpimg from PIL import Image

import PIL

ImageOriginal= cv.imread(r'C:\Users\Alae\Downloads\cartebruitee.png')

- filtre médian

img=cv.medianBlur(ImageOriginal, 5) newImg = cv.GaussianBlur(img, (5, 5), 3) cv.imshow("nouvelleImage",img)

- enregistrer l’image

imwrt = cv.imwrite(r'C:\Users\Alae\Downloads\medianBlur.jpg',newImg) cv.imshow('medianBlur', newImg)

if imwrt:

print('Image enregistrée avec succès.')

cv.waitKey(0)

- Le filtre médianest celuiquiremplace chaque valeur de pixelpar la médiane de sonpixelvoisin.La méthode medianBlur()est idéale lorsqu’ils’agit d’une image avec unbruit poivre et se(comme trouvé dans l’image originale)l.Onyajoute le filtre Guassien,pour supprimer tout autre type de bruit
- ![](Aspose.Words.60199179-39ec-417f-b31e-6be323be9e97.003.jpeg) ![](Aspose.Words.60199179-39ec-417f-b31e-6be323be9e97.004.jpeg)

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* Dans l’autre partie,onveut obtenir des caractères blancs sur fondnoir :

import numpy as np

import cv2

from matplotlib import pyplot as plt

image= cv2.imread(r'C:\Users\Alae\Downloads\medianBlur.jpg') kernel = cv2.getStructuringElement(cv2.MORPH\_OPEN,(3,3))

imagegray = cv2.cvtColor(image, cv2.COLOR\_BGR2GRAY)

(retVal, I2) = cv2.threshold(imagegray, 150, 255, cv2.THRESH\_BINARY) cv2.imshow('image',I2)

#ouverture de l’image

ouverture = cv2.morphologyEx(I2, cv2.MORPH\_OPEN, kernel) cv2.imshow('ouverture',ouverture)

#détermier les contours

SobelC=np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) SobelL=np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) W1=abs(cv2.filter2D(I2,cv2.CV\_64F,SobelC)) W2=abs(cv2.filter2D(I2,cv2.CV\_64F,SobelL)) W=W1+W2

cv2.imshow('Contours ',W )

cv2.waitKey(0)

- closing all open windows cv2.destroyAllWindows()
  - Onutilise notre image medianBlur déjà enregistrée pour rendre ses pixels oubiennoir (0)oublanc (255),étant donné que le fondsera noir.ceciest appelé seuillage,et c’est guarantie grace à la fonction THRESH\_BINARY,les valeurs supérieures ouégales à 150 sont mises à 255,les autres à 0.
  - Onutilise Sobel,quiest unfiltre 3 x3,pour détecter les contours, dans les directions xet y(d'oùle 1 répété 2 fois)pour obtenir une image de type uint8.
- ![](Aspose.Words.60199179-39ec-417f-b31e-6be323be9e97.005.jpeg) ![](Aspose.Words.60199179-39ec-417f-b31e-6be323be9e97.006.jpeg)
  - ![](Aspose.Words.60199179-39ec-417f-b31e-6be323be9e97.007.jpeg)

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

Exercice 3 :Changement de couleur de drapeau:

import numpy as np import cv2 as cv

#obtention path et lecture de l'image originale path=r'C:\Users\Alae\Downloads\DrapeauAllemagne.png' img = cv.imread(path)

#affichage image originale cv.imshow('originale',img)

#determination de l'hauteur h=np.size(img, 0) print("hauteur =",h)

#determiner la longueurlongueur l=np.size(img, 1)

print("longueur =",l)

#determiner l'hauteur d'une bande du drapeau b=h/3

print("bande=",b)

#changer couleur 1ere bande (cx1,cy1)=(96,480) img[0:cx1,0:cy1]=(0,255,255)

#changer couleur 2eme bande (cx2,cy2)=(192,480) img[96:cx2,0:cy2]=(100,190,50)

##changer couleur 3eme bande (cx3,cy3)=(288,480) img[192:cx3,0:cy3]=(50,0,255)

#affichage image sortie cv.imshow('sortie',img)

#savegarde image sortie filename=r'C:\Users\Alae\Downloads\image\_sortie.png' cv.imwrite(filename,img)

cv.waitKey(0); cv.destroyAllWindows(); cv.waitKey(1)

- Oncommence par couper l’image en3 pour obtenir 3 bandes,et identifier les hauteurs et les longueurs de chaque bande (endivisant par 3,car ils sont identiques)puis identifier la couleur de chaque bande,et la convertir enutilisant le code couleur convenable pour chaque bande.

![](Aspose.Words.60199179-39ec-417f-b31e-6be323be9e97.008.png) ![](Aspose.Words.60199179-39ec-417f-b31e-6be323be9e97.009.png)
