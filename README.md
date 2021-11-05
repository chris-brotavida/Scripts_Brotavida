# Brota Superfoods - Scripts proyecto **Artur-Ito** (AI) :robot:+:strawberry:+ :seedling:

Repositorio creado por el equipo I+D de la empresa Brota Superfoods para el desarrollo del modelo IA para creación de recetas basados en requerimientos.

Los códigos que podrían ser de utilidad en el futuro del proyecto se encuentran en archivos `.py` disponibles en la carpeta `main_codes`. 

## 1. Archivos útiles :computer::books:

Tal como se menciona anteriormente, en la carpeta `main_codes` se encuentra al resumen de los archivos útiles desarrollados para el proyecto. Cada función/clase tiene su documentación dentro del mismo *script*, por lo que se recomienda revisar cada una para mayor detalle sobre su funcionamiento.

:warning: **En caso de presentar problemas con las funciones de guardado o apertura de archivos, se recomienda revisar en la función que se desea utilizar cuáles son las direcciones definidas dentro de la función. Algunas funciones fueron directamente trasladads de los *notebook* de prueba desarrollados en los Jupyter Notebook de cada carpeta.** 

:warning::warning:**Algunas funciones realizan gráficos en `plotly`. Sin embargo, `plotly` se visualiza solo en los Jupyter Notebook.**

A continuación se resume el contenido de cada archivo `.py` de esta carpeta, utilizando una escala de colores para la recomendación de estos módulos (:red_circle:: no se recomienda su uso, :large_blue_circle:: se recomienda su uso, :white_circle:: se recomienda su uso, pero su uso como módulo no está comprobado). 

- :red_circle: `food_cluster_analysis.py`: Archivo que contiene funciones para hacer el análisis de *clusters* para la base de datos depurada. Sin embargo, se recomienda correrlos directamente en la carpeta `FOOD_AN` mediante los archivos `FOOD_AN2-Orden.ipynb` y `FOOD_AN4-Orden-Sklearn.ipynb` (no se usa `FOOD_AN3-Orden.ipynb` porque corresponde a un intento de regresión logística mediante `tensorflow.keras`, pero mediante `scikit-learn` presentó resultados más congruentes).
- :large_blue_circle: `foodManagement.py`: Archivo que contiene funciones que permiten manipular la base de datos depurada, retornando la información nutricional de los alimentos escogidos. Este módulo está diseñado para entregar las tablas que se utilizarán con el optimizador del módulo `optimizer.py`.
- :large_blue_circle: `main_test.py`: Módulo de pruebas para coordinar el resto de los archivos con las otras funciones.
- :white_circle: `nlp_data_management`: Archivo que contiene funciones que generan los datos de entrada y salida para entrenar las redes que permiten hacer el procesamiento de lenguaje natural (NLP, del inglés *Natural Language Processing*).  
- :large_blue_circle: `nlp_eval_models.py`: Archivo que contiene funciones para tomar modelos ya entrenados para generar predicciones.
- :large_blue_circle: `nlp_models.py`: Archivo que contiene los modelos NLP utilizados para mapear los `health_effects` de la [**FooDB**](https://foodb.ca/downloads). Para mayor detalle revisar la [sección 3](#3-modelos-nlp-speech_balloonabcd).
- :large_blue_circle: `optimizer.py`: Archivo que contiene la función del cuantificador de los ingredientes.
- :large_blue_circle: `scraping_functions.py`: Funciones desarrolladas para hacer *scraping* de textos en [**Wikipedia**](www.wikipedia.org).
- :large_blue_circle: `sql_functions.py`: Archivo que contiene funciones para hacer consultas a las bases de datos en el servidor `localhost` de MySQL. 

## 2. Pruebas de concepto :memo::wrench:

Algunas carpetas de este repositorio contienen experimentos realizados para pruebas de concepto acerca de distintas aristas del proyecto. En general se recomienda no tomar en consideración estas carpetas ya que no contienen información vital para el proyecto, sino que fueron instancias de prueba para el desarrollo de ideas en cada bloque. Cada carpeta se detalla a continuación:

- `FOOD_AN`: Contiene scripts que permiten el análisis de los vectores depurados que contienen información a nivel de compuestos (macro y micro nutrientes) utilizando técnicas de *clustering* y reducción de dimensiones. Se incluyen también algunas de las bases depuradas de alimentos con sus componentes de interés. 
- `formatterBaseDatos`: Contiene *tests* de manejo de bases de datos basados en SQL o en API. 
- `MVP_0`: Contiene pruebas de la versión más básica del modelo de optimizador, correspondiente al cuantificador de ingredientes. Sin embargo, se recomienda revisar la versión más actualizada del optimizador, disponible en el archivo `optimizer.py` de la carpeta `main_codes`.
- `MVP_1`:  Contiene pruebas de una versión un poco más desarrollada del modelo del optimizador.
- `MVP_2`: Contiene pruebas de la versión más cercana a la última versión del optimizador.
- `MVP_3`: Contiene pruebas "*naïve*" de NLP para la vinculación de alimentos y sus descripciones. Existen algunos scripts que permiten el *scraping* de texto obtenido a partir de [**Wikipedia**](www.wikipedia.org).
- `NLP_INGDS_1`: Contiene los *scripts* de NLP, principalmente basados en generar un diccionario inverso de los conceptos presentes en la tabla `health_effects` de la base de datos obtenidas de la [**FooDB**](https://foodb.ca/downloads).
- `Recipe_generator_tests`: Contiene los *scripts* de prueba iniciales que se usaron para la creación de recetas con RNN's basado en el blog de [**KDNuggets**](https://www.kdnuggets.com/2020/07/generating-cooking-recipes-using-tensorflow.html). Estas primeras pruebas permitieron acercarnos al funcionamiento de las RNN aplicadas a NLP mediante la librería `Keras`.

## 3. Modelos NLP :speech_balloon::abcd:

Para poder relacionar descripciones cualitativas con los requerimientos nutricionales de los ingredientes en la base de datos, se intentó utilizar distintas técnicas de NLP. El objetivo de estos experimentos fue obtener una representación N-dimensional del "significado" de cada palabra en un diccionario de palabras en inglés. Para esto se utilizaron distintas arquitecturas de redes neuronales que actualizan las posiciones de estos vectores en cada iteración. 

A continuación se describirá en rasgos generales en qué consiste cada técnica y sus principales resultados para el mapeo de conceptos disponibles en la tabla `health_effects` de la  [**FooDB**](https://foodb.ca/downloads).

:warning:**<u>Advertencia</u>: En la carpeta `NLP_INGDS_1/Results` se encuentran los modelos entrenados que se describirán a continuación. Sin embargo, no en todas las instancias se guardaron sus tokenizer (objeto que transforma las palabras en cada texto a su id único). En las instancias `model_CBOW_1` y `model_CBOW_1_Siamese_traditional_test_v2(masking)` se guarda un tokenizer para la transformación de los textos.**

### 3.1. Continuous Bag Of Words (CBOW)

Este es uno de los modelos más básicos para entrenar *word embeddings*. En general, en este método se escoge una palabra objetivo y se utilizan las palabras alrededor como contexto. Por ejemplo, con la oración "El perro comió pasto camino hacia el prado." Si tomamos la palabra "pasto", su contexto de 2 palabras sería por la izquierda "perro comió" y por la derecha "camino hacia". En esta aplicación clásica, cada una de estas 4 palabras son utilizadas para entrenar una red que mapee la palabra objetivo "pasto". En este ejercicio, la matriz que representa los vectores de cada palabra se actualiza en conjunto con la capa de perceptrones que genera la salida. 

Como resultado, se obtiene una matriz de *embeddings* que contiene las palabras N-dimensionales entrenadas, en donde los conceptos similares son más cercanos en esta dimensión.

Sin embargo, en esta instancia en vez de mapear un contexto de palabras para descubrir la palabra central, se mapeará cada palabra al concepto objetivo (en nuestro caso, un efecto de salud), tal como se puede apreciar en la figura 1. 



<figure>
	<div style="text-align:center">
		<img src="NLP_INGDS_1/imgs/CBOW-Diagrams-CBOW.drawio.png" height=300/>
    </div>
	<figcaption align = "center"><b>Fig.1 - CBOW simplificado</b></figcaption>
</figure>


Para realizar la predicción de los conceptos de interés en base una oración, se "tokeniza" la oración, obteniendo una lista de palabras para cada oración. Cada palabra es mapeada a un id único, en donde cada id representa una fila de la matriz de *embeddings*. A su vez, la matriz de *embeddings* es una matriz V filas, donde cada una representa a cada palabra del vocabulario; y N columnas, donde cada una representa una dimensión del *embedding*). 

Una vez tokenizada la oración y sabiendo cuál es el id de cada palabra, es posible obtener un vector de *embedding* para cada palabra a partir de la matriz de *embeddings*. Estos vectores se comparan (mediante una métrica de distancia, como la distancia euclidiana o la similaridad de coseno) con cada vector que representa cada concepto de interés. El promedio de estas distancias genera candidatos a este concepto de interés.

El resultado de esta instancia se puede encontrar en el archivo `2_2-CBOW-Modularized.ipynb` de la carpeta `NLP_INGDS_1`. Como comentarios generales, el modelo cae en un mínimo local obteniendo un *accuracy* de 50% aproximadamente. Luego de eso no genera grandes mejoras. Al realizar la predicción tampoco presenta resultados con mucho sentido, por lo que se descarta como método de *word embedding*.

### 3.2. Semi-Siamese Networks

En esta implementación se utiliza la idea de las redes siamesas para generar el *embedding*. Sin embargo, no se utilizan unidades recurrentes como LSTM o GRU en esta arquitectura. Por este motivo se les denominó "Semi-Siamese Networks" (ver figura 2). Esta red utiliza dos caminos, cuya matriz de *embedding* es la misma para cada uno. Mediante una métrica de distancia, se obtiene el resultado de la similaridad entre las entradas de esta red. Una de las entradas será cada palabra de la descripción, y otra de las entradas será el concepto de interés.



<figure>
	<div style="text-align:center">
		<img src="NLP_INGDS_1/imgs/CBOW-Diagrams-Semi-Siamese.drawio.png" height=400/>
    </div>
	<figcaption align = "center"><b>Fig.2 - Semi-Siamese Network</b></figcaption>
</figure>


Los resultados de este experimento no lograron resolver el problema. Por una parte, porque la función de distancia utilizada fue una norma 2, por lo que la capa de perceptrones no podía mapear una baja distancia tendiente a cero (que es lo que se busca, es decir, que conceptos y palabras relacionadas sean parecidas) a un valor cercano a 1 (indicando mayor similaridad). Por lo tanto, los resultados de este entrenamiento no tenían sentido.

A partir de esto se propuso eliminar la capa de perceptrones de salida, de tal forma que la salida de la red fuera precisamente la función de distancia, tal como se puede ver en la figura 2.1.



<figure>
	<div style="text-align:center">
		<img src="NLP_INGDS_1/imgs/CBOW-Diagrams-Semi-Siamese 2.drawio.png" height=400/>
    </div>
	<figcaption align = "center"><b>Fig.2.1 - Semi-Siamese Network sin la capa de perceptrones</b></figcaption>
</figure>



Si bien los resultados de este entrenamiento entregaron una función de costos MSE (del inglés *Mean Squared Error*) muy cercano a cero (lo cual es deseable), al tratar de predecir los resultados eran muy similares entre los distintos conceptos y las palabras de interés. Se sospecha que los *embeddings* de los conceptos se adecuaron de tal forma que todos quedaron muy cerca entre sí. Por lo tanto, tampoco presentó resultados relevantes.

### 3.3. Siamese Networks

Finalmente se intentó con las clásicas redes siamesas. En primera instancia se diseñó un sistema que aplicaba una capa LSTM únicamente sobre el camino de la descripción, tal como se puede ver en la figura 3.



<figure>
	<div style="text-align:center">
		<img src="NLP_INGDS_1/imgs/CBOW-Diagrams-Siamese.drawio.png" height=400/>
    </div>
	<figcaption align = "center"><b>Fig.3 - Siamese Network por un solo lado</b></figcaption>
</figure>


Sin embargo, el computador no soportaba esta arquitectura, y cada vez que se intentó correr una instancia de este modelo, el computador cerraba el kernel de Jupyter Notebook. Se recomienda probar con esta arquitectura quizás en otra plataforma como GCP (Google Cloud Platform) o AWS (Amazon Web Services), o bien probar correrlo con otro tipo de *framework* que no sea `tensorflow.keras`.

Por último, se intentó con una versión clásica de la red siamesa (ver figura 4), y se entrenó utilizando el concepto de "*Negative Sampling*". En este caso, se agregó una nueva variable que indica la relación entre el concepto y su descripción. Si el concepto corresponde a la descripción proporcionada, la relación se indicará con un 1; en caso contrario se indicará con un 0. Por lo tanto, para generar el conjunto de entrenamiento de la red se mezclaron los conceptos con las descripciones, siempre asegurándose de que para cada concepto, al menos estaba su descripción real.

 

<figure>
	<div style="text-align:center">
		<img src="NLP_INGDS_1/imgs/CBOW-Diagrams-Siamese v3.drawio.png" height=400/>
    </div>
	<figcaption align = "center"><b>Fig.4 - Siamese Network clásica</b></figcaption>
</figure>



Como resultado, este modelo entrego un 97% de *accuracy* en el entrenamiento y un valor de la función de costos muy cercano a cero. A partir de esto, se recomienda este tipo de arquitecturas para resolver el problema del diccionario inverso.  

Cabe destacar que si bien es un modelo que funciona bastante bien, se reconocen ciertas dificultades. 

1. Para comprobar el funcionamiento esta arquitectura se utilizaron solamente 7 conceptos, a partir de los cuales se combinaron formando 7*7=49 pares de descripción/concepto. Sin embargo, mientras más conceptos y definiciones se tengan, este número crecerá cuadráticamente. Por lo tanto, se recomienda acotar el número de relaciones descripción/concepto para no hacer una combinatoria de todos entre sí.
2. En caso de agregar un nuevo concepto, probablemente se tendrá que entrenar con el resto de los conceptos incluidos. Esto podría generar una carga computacional no despreciable. Se recomienda de todas formas entrenar únicamente con el nuevo concepto la red, y ver cuánto varía el resto de los resultados, asegurando el correcto funcionamiento. O bien, se recomienda tener previamente definido el universo de descripciones/conceptos, de tal forma que no sea necesario reentrenar.



