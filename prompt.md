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

En train/dataset.py get_training_transforms implementar RandomSizedCrop, PixelDropout y Erasing

'''

# Webapp

Crear una aplicacion:


Modificar la aplicacion para que cumpla con los siguientes requisitos

- Compatible con Huggingface Spaces
- El codigo tiene que estar en ingles
- Usar ONNX Runtime web, el procesamiento se tiene que realizar en el navegador
- Usar la version 1.22.0 de ONNX Runtime web
- Instentar utilizar WebGPU, si falla, WASM
- La aplicacion muestra una seleccion de las camaras disponibles al inicio
- Realiza inferencia del modelo que se encuentra en train/exported_models/card_segmentation.onnx
- Hay un webserver (demo.py) que pone el modelo en /models/card_segmentation.onnx
- Para mas informacion del modelo ver train/exported_models/README.md
- El modelo es de segmentacion de imagen, en la app se tiene que mostrar la mascara celeste transparente.
- La inferencia se tiene que realizar en el video (ciclo continuo)
- La app se guarda en /demo
- Intentar inicializar la camara en las siguientes resoluciones, en este orden
  - 1280x720, 640x480
  - Si todas las resoluciones fallan, mostrar un mensaje de error
- En la web, mostrar la camara de la siguiente forma si es tiene una horientacion Horizontal
  - Recortar los costados para obtener una imagen en una proporcion 4:3 centrada
- En la web, mostrar la camara de la siguiente forma si es tiene una horientacion Vertical
  - Recortar las partes de arriba y abajo para obtener una imagen en una proporcion 4:3 centrada
- Utilizar la misma logica de recorte antes de realizar la inferencia en el modelo
- En la parte de abajo se muestra 
  - El backend de ONNX (WebGPU/WASM) que se esta utilizando 
  - La cantidad de FPS
  - La resolucion de la camara


Generar una imagen SVG, de 640 de alto por 480 de ancho (portraid), en el medio de la imgen, con un borde del 20% en cada lado, generar un rectangulo fino de esquinas redondeadas, el rectangulo tiene una proporcion de 5:7, la parte exterior del rectangulo tiene un griz con un 80% de transparencia. El rectangulo del centro tiene que ser completamente transparente.

Agregar una funcion a la clase ImageUtils en el archivo image-utils.js que recorte el centro del video en una proporcion 3:4, el recorte tiene que ser lo mas grande posible.


# Train Script -- Pose estimation

- Aplicar Aumentación de datos al dataset, tiene que incluir las siguientes:
  - Zoom In desde un 0 a un 50% de la imagen centrado
  - Blur suave y blur de movimiento
  - Mover la imagen vertical o horizontalmente hasta un 25% de la misma
  - Simular diferentes ISO noise
  - Sumar augmentaciones adicionales
- Usar augmentations para las aumentaciones
- Generar un script para generar 10 ejemplos de la augmentacion de datos
  - La precision del modelo tiene que ser FP16

Generar los scripts para entrenar el modelo:

- Informacion adicional del dataset
  - En el directorio /train/images estan las imagenes de entrenamiento 
  - En el directorio /test/images estan las imagenes de test (validacion de entrenamiento)
  - Las imagenes estan en formato jpg
  - Las imagenes son de 480 (ancho) por 640 (alto) (Portraid)
  - El dataset se encuentra en el directorio dataset
  - La informacion sobre las 4 esquinas se encuentra en el archivo corner_annotations.json
- Sobre el modelo:
  - Usar yolo12n-pose y ultralytics
  - Resolucion de entrada 480 (ancho) por 640 (alto) (Portraid)
  - Retornar los 4 puntos de la carta
  - Modelo exportable a ONNX
  - El modelo tiene que poder ejecutarse en Web
  - El modelo tiene que poder ejecutarse en Mobile (Android/IOS)
- El codigo tiene que estar en ingles
- Todos los script tienen que estar en la carpeta train generar la carpeta si no existe
- Crear la documentacion en train/README.md
- Usar os.path.realpath() para garantizar la compatibilidad entre pataformas e interpretes
- Usar Cuda para el entrenamiento
- Implementar las siguientes tecnicas de entrenamiento
  - Early Stopping 
  - Reduce LR On Plateau 
- Metricas de porcentual de resultados esta a menos de 20px, 10px y 5px de lo esperado 
