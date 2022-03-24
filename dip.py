'''
¡¡ATENCION!! 
  *El programa presenta mejores resultados con imagenes en alta resolución
  *Para obtener siluetas correctas es recomendable recogerse el cabello largo antes de tomarse la foto y así dejar ver la medida de los hombros

1.Utilizar versiones anteriores a Python 3.10 y mayores iguales que Python 3.7
2.Instalar ndimage.whl segun las caracteristicas de su OS del siguiente enlace https://www.lfd.uci.edu/~gohlke/pythonlibs/#ndimage 
3.De la siguiente manera podrá instalar en el terminal el archivo .whl --- $ pip install "nombre del archivo".whl
4.Finalmente, instale las liberías correspondientes numpy,OpenCV,math,mediapipe y scipy
'''

#Importamos las librerías necesarias
import numpy as np
import cv2
import math
import mediapipe as mp
from scipy.ndimage.morphology import binary_erosion

#Ancho y tamaño deseado de la imagen
height = 480
width = 480

#Creamos una función, la cual va a dar a la imagen las dimensiones que queremos y la mostrará
def resize_and_show(name,image):
  #obtenemos solo el alto y el ancho de la imagen, por eso seleccionamos solo los dos primeros 
  #la función shape con ayuda a obtener las dimensiones de una matriz (alto,ancho,dimension de color)
  #en este caso solo necesitamos el alto y ancho de la matriz de la imagen
  h, w = image.shape[:2]
  
  #Si es muy ancho acomodamos la imagen para que cuadre con nuestras preferencias
  if h < w:
    img = cv2.resize(image, (height, math.floor(h/(w/width))))
  #Si es muy alto acomodamos la imagen para que cuadre con nuestras preferencias
  else:
    img = cv2.resize(image, (math.floor(w/(h/height)), width))
  #Mostramos la imagen
  cv2.imshow(name,img)

#Obtenemos la clase selfie_segmentation, la cual nos ayudará a 
mp_selfie_segmentation = mp.solutions.selfie_segmentation

#Incializamos un contexto "with" y utilizamos los valores configurables de SelfieSegmentation
#Tenemos los valores de 0 ó 1 como elección, en este caso escogeremos 1
#ya que nos permite de un modelo de rango completo mejor figuras dentro de los 5 metros
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
  #Obtenemos el nombre de la imagen que queremos segmentar
  image_name="prueba 2.jpg"
  
  #Inicializamos la imagen con OpenCV
  image_o=cv2.imread(image_name)
  
  #Convertimos nuestra imagen en formato BGR a RGB y procesamos con  MediaPipe Selfie Segmentation 
  results = selfie_segmentation.process(cv2.cvtColor(image_o, cv2.COLOR_BGR2RGB))

  #A la imagen resultante obtendremos su UMBRAL, donde si los valores dentro de la imagen son mayores 
  #que 0.75 se visualizaran en blanco (255) y los menores en negro(0), para esto tambien le 
  #agregamos que queremos hacer una binarización binaria invertida, lo que resultará que tengamos
  #un fondo de color blanco
  _,tr_binary=cv2.threshold(results.segmentation_mask,0.75,255,cv2.THRESH_BINARY_INV)
  

  #Una vez obtenida el umbral de la imagen, pasamos a realizarle una erosión donde rellenaremos los espacios faltantes en la imagen
  #El rellenador tendrá un tamaño de 1x1 y se realizaran 20 iteraciones donde buscará las partes por rellenar en la imagen
  eroded = binary_erosion(tr_binary, structure=np.ones((1, 1)), iterations=20)
  
  #Obtendremos una matriz con valores True y False
  #De la siguiente manera haremos el cambio de tipo de dato a float32
  image=np.float32(eroded)
  
  #Realizamos un filtro medianBlur para suavizar la imagen
  image = cv2.medianBlur(image, 5)

  #Reescalamos y mostramos la imagen obtenida y la imagen original
  resize_and_show("imagen original",image_o)
  resize_and_show(image_name,image)
  #Pausaremos las ventanas al terminar de mostrar
  cv2.waitKey(0)