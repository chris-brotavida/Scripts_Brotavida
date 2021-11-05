import numpy as np
import tensorflow as tf


def model_CBOW(vocab_size, embedding_matrix, embedding_dim=100, embedding_trainable=True,
               activation_out='sigmoid', lr=0.001, loss='sparse_categorical_crossentropy',
               metrics=['accuracy']):
    '''Función que retorna un modelo de "Continuous Bag Of Words" para realizar 
    word embedding.
    
    Parameters
    ----------
    vocab_size : int
        Tamaño del vocabulario.
    embedding_matrix : ndarray
        Matriz de embeddings.
    embedding_dim : int, optional
        Dimensión de embedding para cada palabra/concepto. Por defecto es 100.
    embedding_trainable : bool, optional
        Booleano que indica si la matriz de embedding se considera como parámetro
        entrenable del modelo. Si es True, la matriz se actualizará con cada paso 
        del backpropagation. Si es False, no se actualizará. Por defecto es True.
    activation_out : str, tf.keras.activations, optional
        Función de activación a la salida de la red. Por defecto es 'sigmoid'.
    lr : float, optional
        Tasa de aprendizaje (learning rate) del optimizador. Por defecto es 0.001.
    loss : str, tf.keras.losses, optional
        Función de pérdida del optimizador. Por defecto es 
        'sparse_categorical_crossentropy'.
    metrics : list, optional
        Lista de métricas a utilizar para visualizar el desempeño del modelo. 
        Por defecto es ['accuracy'].
        
    Returns
    -------
    model : tf.keras.Model
        Modelo CBOW compilado con las opciones proporcionadas. 
    '''
    # Capa de entrada
    x_targ_in = tf.keras.Input(shape=(1,), name='Target_Input')
    # Capa de embebido
    x_targ = tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                                       trainable=embedding_trainable,
                                       weights=[embedding_matrix],
                                       mask_zero=True)(x_targ_in)
    # Capa FF
    x_out = tf.keras.layers.Dense(units=vocab_size, kernel_initializer='he_normal',
                                  activation=activation_out)(x_targ)
    
    # Definición del modelo
    model = tf.keras.Model(inputs=[x_targ_in], outputs=[x_out])
    
    # Compilando el modelo
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model 


def semi_siamese_network(vocab_size, embedding_matrix, embedding_dim=100, 
                         embedding_trainable=True, activation_out='sigmoid',
                         without_dense=False, lr=0.001, loss='mse',
                         metrics=['mse']):
    '''Función que retorna un modelo de red semi-siamesa para realizar word embedding.
    La idea es utilizar o la matriz de embeddings o el modelo mismo. Se le llama 
    semi-siamesa ya que no hace uso de una unidad recurrente.
    
    Parameters
    ----------
    vocab_size : int
        Tamaño del vocabulario.
    embedding_matrix : ndarray
        Matriz de embeddings.
    embedding_dim : int, optional
        Dimensión de embedding para cada palabra/concepto. Por defecto es 100.
    embedding_trainable : bool, optional
        Booleano que indica si la matriz de embedding se considera como parámetro
        entrenable del modelo. Si es True, la matriz se actualizará con cada paso 
        del backpropagation. Si es False, no se actualizará. Por defecto es True.
    activation_out : str, tf.keras.activations, optional
        Función de activación a la salida de la red. Por defecto es 'sigmoid'.
    without_dense : bool, optional
        Booleano que indica si a la salida se utiliza o no una capa densa. Si es
        True, se elimina la capa densa de salida. Si es False, se utiliza la capa 
        densa de salida. Por defecto es False.
    lr : float, optional
        Tasa de aprendizaje (learning rate) del optimizador. Por defecto es 0.001.
    loss : str, tf.keras.losses, optional
        Función de pérdida del optimizador. Por defecto es 'mse'.
    metrics : list, optional
        Lista de métricas a utilizar para visualizar el desempeño del modelo. 
        Por defecto es ['mse'].
        
    Returns
    -------
    model : tf.keras.Model
        Modelo de red semi-siamesa compilado con las opciones proporcionadas. 
    '''
    # Capa de entrada
    x_cont_in = tf.keras.Input(shape=(1,), name='Input_context')
    x_targ_in = tf.keras.Input(shape=(1,), name='Input_target')
    
    # Capa de embebido
    embedd = tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                                       trainable=embedding_trainable,
                                       weights=[embedding_matrix],
                                       mask_zero=True)
    # Aplicando
    x_cont = embedd(x_cont_in)
    x_targ = embedd(x_targ_in)
    
    # Capa de diferencia
    l1_norm = lambda x: tf.norm((x[0] - x[1]), axis=-1)
    distance = tf.keras.layers.Lambda(function=l1_norm, 
                                      output_shape=lambda x: x[0], 
                                      name='L1_distance')([x_cont, x_targ])
    
    # Capa FF
    if not without_dense:
        # Capa FF
        if activation_out == 'sigmoid':
            n_units = 1
        elif activation_out == 'softmax':
            n_units = 2
        x_out = tf.keras.layers.Dense(units=n_units, kernel_initializer='he_normal',
                                      activation=activation_out, name='Output')(distance)
    else:
        x_out = distance
    
    # Definición del modelo
    model = tf.keras.Model(inputs=[x_cont_in, x_targ_in], outputs=x_out)
    
    # Compilando el modelo
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model


def siamese_network(vocab_size, embedding_matrix, max_len=10, embedding_dim=100, 
                    embedding_trainable=True, activation_out='sigmoid', lr=0.001,
                    loss='binary_crossentropy', metrics=['accuracy'], 
                    distance='cosine'):
    '''Función que retorna un modelo de red siamesa para realizar word embedding.
    La idea es utilizar o la matriz de embeddings o el modelo mismo.
    
    Parameters
    ----------
    vocab_size : int
        Tamaño del vocabulario.
    embedding_matrix : ndarray
        Matriz de embeddings.
    max_len : int
        Largo máximo de la secuencia en la entrada 'Input_context'.
    embedding_dim : int, optional
        Dimensión de embedding para cada palabra/concepto. Por defecto es 100.
    embedding_trainable : bool, optional
        Booleano que indica si la matriz de embedding se considera como parámetro
        entrenable del modelo. Si es True, la matriz se actualizará con cada paso 
        del backpropagation. Si es False, no se actualizará. Por defecto es True.
    activation_out : str, tf.keras.activations, optional
        Función de activación a la salida de la red. Por defecto es 'sigmoid'.
    lr : float, optional
        Tasa de aprendizaje (learning rate) del optimizador. Por defecto es 0.001.
    loss : str, tf.keras.losses, optional
        Función de pérdida del optimizador. Por defecto es 'binary_crossentropy'.
    metrics : list, optional
        Lista de métricas a utilizar para visualizar el desempeño del modelo. 
        Por defecto es ['accuracy'].
    distance : {'cosine', 'euclidean'}, optional
        Métrica de distancia a utilizar para comparar los vectores en cada path.
        Por defecto es 'cosine'.
        
    Returns
    -------
    model : tf.keras.Model
        Modelo de red siamesa compilado con las opciones proporcionadas. 
    '''
    # Capa de entrada
    x_cont_in = tf.keras.Input(shape=(max_len,), name='Input_context')
    x_targ_in = tf.keras.Input(shape=(1,), name='Input_target')
    
    # Capa de embebido
    embedd = tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                                       trainable=embedding_trainable,
                                       weights=[embedding_matrix],
                                       mask_zero=True)
    # Aplicando
    x_cont = embedd(x_cont_in)
    x_targ = embedd(x_targ_in)
    
    # Capa LSTM solo sobre la entrada de la descripción
    lstm = tf.keras.layers.Bidirectional( 
                tf.keras.layers.LSTM(units=embedding_dim, kernel_initializer='he_normal',
                                    dropout=0.2, recurrent_dropout=0.2)
            )

    # Aplicando la capa LSTM
    x_cont = lstm(x_cont)
    x_targ = lstm(x_targ)
    
    # Capa de diferencia
    if distance == 'cosine':
        dist_func = \
            lambda x: tf.keras.backend.batch_dot(tf.keras.backend.l2_normalize(x[0], axis=-1), 
                                                 tf.keras.backend.l2_normalize(x[1], axis=-1), 
                                                 axes=-1)
    elif distance == 'euclidean':
        dist_func = lambda x: tf.norm((x[0] - x[1]), axis=-1)
    else:
        raise Exception(f'Opción {distance} no válida para parámetro "distance".')
    
    # Creación capa
    dist_lay = tf.keras.layers.Lambda(function=dist_func, 
                                    output_shape=lambda x: x[0], 
                                    name='distance')([x_cont, x_targ])
    
    # Capa FF
    if activation_out == 'sigmoid':
        n_units = 1
    elif activation_out == 'softmax':
        n_units = 2
    x_out = tf.keras.layers.Dense(units=n_units, kernel_initializer='he_normal',
                                  activation=activation_out, name='Output')(dist_lay)
    
    # Definición del modelo
    model = tf.keras.Model(inputs=[x_cont_in, x_targ_in], outputs=x_out)
    
    # Compilando el modelo
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model


# Módulo de testeo
if __name__ == '__main__':
    a = siamese_network(vocab_size=10, embedding_matrix=np.zeros((10,10)))
    print(a.summary())