# Internal documentation

This document is a personal reference.

# Train Script

Generar los scripts para entrenar Semantic Segmentation, tomar las siguientes consideraciones:

- Framework PyTorch
- Arquitectura Model LR-ASPP (lraspp_mobilenet_v3_large)
- Precisión del modelo tiene que ser FP16 (idealmente en entrenamiento)
- Input Resolution de 480x640 (vertical)
- Scripts para pruning del modelo
- El codigo tiene que estar en ingles
- Todos los script tienen que estar en la carpeta train/ generar la carpeta si no existe
- Crear la documentacion en train/README.md
- Usar os.path.realpath() para garantizar la compatibilidad entre pataformas e interpretes
- El dataset se encuentra en ../dataset relativo al directorio train/
- Informacion adicional del dataset
  - En el directorio /train/images estan las imagenes de entrenamiento
  - En el directorio /train/masks estan las mascaras de entrenamiento
  - En el directorio /test/images estan las imagenes de test
  - En el directorio /test/masks estan las mascaras de test
  - Solo hay 2 categorias, carta y fondo (card or background)
  - Las imagenes estan en formato jpg
  - Las mascaras estan en formato png los pixeles blancos son la carta, negro el fondo
- Aplicar Aumentación de datos al dataset
- Use Cuda for training 
- Do not use wandb

'''

Agregar a los script en train/ la opcion de usar una aquitectura Lite R-ASPP con un backbone mobilenet_v3_small.

'''

# Webapp

Crear una aplicacion compatible con Huggingface Spaces

- El codigo tiene que estar en ingles
- Usar ONNX Runtime web, el procesamiento se tiene que realizar en el navegador
- La aplicacion muestra una seleccion de las camaras disponibles al inicio
- Tras seleccionar la camara, se reproduce el video de la camara en 480x640 si esta disponible, si no 640x480
- Si la camara es horizontal, rotar la camara 90 grados a contrarreloj para simular que la camara es vertical.
- Realiza inferencia del modelo que se encuentra en train/exported_models/card_segmentation.onnx
- Para mas informacion del modelo ver train/exported_models/README.md
- El modelo es de segmentacion de imagen, en la app se tiene que mostrar la mascara celeste transparente.
- La inferencia se tiene que realizar en el video (ciclo continuo)
- Si la imagen de entrada al modelo es 640x480, rotar la imagen 90 grados a contrarreloj para llevarlo a 480x640
- La app se guarda en /demo

'''
La demo en demo/ da el siguiente error en el navegador

Mejorar la demo en demo/ para que ONNX Runtime web utilice la GPU si esta disponible.
'''