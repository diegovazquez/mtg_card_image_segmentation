# Internal documentation

This document is a personal reference.

## 01 - Download references

Utilizar la api publica de scryfall para bajar imagenes de referencias de las cartas de MTG. Poner una pausa de 100 milisegundos entre solicitudes.

Generar 2 sets de imagenes, uno de test y otro de entrenamiento,

El dataset de entrenamiento requiere 2000 imagenes de cartas diferentes.
El dataset de test requiere 500 imagenes de cartas diferentes.

El dataset de entrenamiento descargarlo en el directorio references/train, generarlo si no existe.
El dataset de test descargarlo en el directorio references/test, generarlo si no existe.

En ambos datasets 
- 1/4 de las imagenes tienen que ser full_art true
- el resto con full_art en false.

Todas las imagenes tienen que ser "image_status" igual a "highres_scan". 

Otras consideraciones
- Generar una barra de progreso al descargar
- Usar el ID de la carta, para evitar duplicados.
  - Si es full_art el archivo tiene un prefijo "full_art" 
  - Si no, normal el archivo tiene un prefijo "normal" 
- Si una carta esta en train, no usar la misma en test, usar el "name" de la carta no el ID para esto.
- Si se produce un error, esperar 100 milisegundos y intentar con otra carta.
- Guardar el script en dataset_generator/01_download_references.py
- Generar el codigo en ingles.
- El parametro Q tiene el valor (game:paper) en URL encode

La api https://api.scryfall.com/cards/search retorna un nodo data, donde tendremos una lista de propiedades de cartas.

Ejemplo, GET https://api.scryfall.com/cards/search?format=json&include_extras=false&include_multilingual=false&include_variations=false&order=cmc&page=1&q=%28game%3Apaper%29&unique=prints"

Retorna 

"next_page": "https://api.scryfall.com/cards/search?format=json&include_extras=false&include_multilingual=false&include_variations=false&order=cmc&page=2&q=c&unique=cards",
"data":
[{
      "name": "EXAMPLE",
      "object": "card",      
      "highres_image": true,
      "image_status": "highres_scan",
      "full_art": false,
      "textless": false,
      "image_uris": {"png": "URL"}
}]

El parametro next_page es la URL donde obtener la siguiente lista de cartas, "image_uris" -> "png" es donde obtener la URL de la imagen.

## 02 - Download HDR

Teniendo en consideracion el OpenAPI https://api.polyhaven.com/api-docs/swagger.json 

- Descargar 20 hdris de la categoria indoor en la carpeta dataset_generator/hdri 
- Fijarse los HDRIs que estan disponible y seleccionar la cantidad a descargar de forma aleatoria
- Tener una barra de progreso en la descarga.
- Descargar solo versiones de 8K
- Guardar el script en dataset_generator/02_dowload_hdrs.py
- Generar las carpetas relativas a el archivo del script.
- Generar el codigo en ingles.
- Usar os.path.realpath() para garantizar la compatibilidad entre pataformas e interpretes.

## 03 - Generate Synthetic single

'''
Utilizar BlenderProc2 para generar imagenes una imagen sintetica de cartas Magic The Gatering.
La clase para generar las imagenes tiene que tener como entrada una imagen de referencia de 745x1040 pixeles, con esa imagen
hay que generar una carta de bordes redondeados de 0.3 milimetros y ponerla verticalmente enfrente de la camara,  la camara va ser vertical de proporcion 9:16.
La carta tiene que estar centrada y ocupar de desde un 40% a un 70% de la imagen captada por la camara, para esto, cambiar la distancia de la carta a la camara.

- La cantidad de imagenes generadas por imagen de referencia es un parametro de entrada, por default 4.
- Generar los mapas de segmentacion de la carta y el background
- Las imagenes de salida se tienen que guardar en una carpeta configurable, por defecto synthetic_output/image y synthetic_output/mask 
- La carta puede estar rotada de 0 a 15 grados en cualquiera de los ejes, aleatoreamente.
- Los fondos tienen que ser aleatoreos y variados.
- Se pueden utilizar archivos HDRI, se encuentran en dataset_generator/hdri.
- La resolucion de salida esperada es 720x1280, para simular una camara vertical.
- Generar el codigo en ingles.
- Las imagenes tienen que ser guardadas en formato PNG
- Usar os.path.realpath() para garantizar la compativilidad entre pataformas e interpretes
- Se usa la version Blender 4.2.1 LTS
'''

'''
Las imagenes generadas por el script dataset_generator/generate_synthetic.py tienen un problema, la carta no muestra la textura de la imagen de input correctamente. 
La idea es que la de la imagen PNG del input este en el lado frontal de la carta (solo el frontal), la parte de atras use la imagen back.png que se encuentra en el mismo directorio que el script y los bordes en negro.

Mantener las modificaciones al minimo.

Para ejecutar el script usar el siguiente comando:

python -m blenderproc run dataset_generator/generate_synthetic.py --input dataset_generator/references/test/full_art_0a35bb96-89de-4a0a-a53e-aa97f800e92f.png --count 1 --hdri dataset_generator/hdri --output synthetic_output_test
'''

'''
Las imagenes generadas por el script dataset_generator/generate_synthetic.py tienen un problema, la carta no muestra la textura de la imagen de input correctamente. 
La idea es que la de la imagen PNG del input este en el lado frontal de la carta (solo el frontal), la parte de atras use la imagen dataset_generator/back.png y los bordes sean negros.

- Usar os.path.realpath() para garantizar la compatibilidad entre pataformas e interpretes
- Generar el codigo en ingles.
'''

'''
Modificar dataset_generator/generate_synthetic.py: 

- Rotar la camara para que este de 45 a 135 grados en el eje Y
- Rotar la camara de 0 a 360 sobre el eje X
- Mover la carta MTGCard para que quede en la misma posicion relativa a la camara antes de rotar la misma
- Rotar la carta MTGCard para que quede en el mismo angulo relativo a la camara como antes de rotar
- Generar el codigo en ingles.

Para ejecutar el script usar el siguiente comando:

python -m blenderproc run dataset_generator/generate_synthetic.py --input dataset_generator/references/test/full_art_0a35bb96-89de-4a0a-a53e-aa97f800e92f.png --count 10 --hdri dataset_generator/hdri --output synthetic_output_test
'''

NOTA: La llm no se funciona bien para esto, se modifico bastante a mano.

## 04 - Generate Synthetic

Generar un script para generar un dataset de imagenes utilizando la clase MTGCardSynthetic.

- Usar la clase MTGCardSynthetic del archivo dataset_generator/generate_synthetic.py
- El parametro hdri_dir es la carpeta hdri que se encuentra en el mismo directorio que el script
- Las imagenes de las cartas referencia las encuentran en las carpetas references/train y references/test (reference_image_path)
- Por cada imagen de referencia, generar 4 imagenes sinteticas (images_per_reference)
- Si se usan las imagenes de referencia de references/train guardar las imagenes genradas dataset/train (output_base_dir)
- Si se usan las imagenes de referencia de references/test guardar las imagenes genradas dataset/test (output_base_dir)
- Guardar el script en dataset_generator/03_generate_synthetic_dataset.py
- Generar el codigo en ingles 
- La primera linea del script tiene que ser: import blenderproc as bproc

## 04 - synthetic_dataset_anti_leak


Generar un script que ejecute el siguiente comando en linux
  blenderproc run dataset_generator/03_generate_synthetic_dataset.py
O el siguiente en windows
  python -m blenderproc run dataset_generator/03_generate_synthetic_dataset.py

y cada X minutos pare el proceso y lo ejecute nuevamente. Hasta que la ejecucion del script tarde menos de X minutos.
X es 10 por defecto, el archivo se tiene que llamar synthetic_dataset_anti_leak.py
