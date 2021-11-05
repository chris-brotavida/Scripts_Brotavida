import numpy as np
import tensorflow as tf
from sql_functions import preprocess_HE_decriptions


class custom_tokenizer(object):
    '''Tokenizer creado en base al tokenizer de tensorflow.
    
    Parameters
    ----------
    split : str
        Caracter que define la separación de tokens.
        
    Attributes
    ----------
    word_index : dict
        Diccionario que mapea las palabras a id's únicos.
    index_word : dict
        Diccioario que mapea los id's únicos a palabras.
    '''
    def __init__(self, split):
        self.split = split
        self.word_index = dict()
        self.index_word = dict()
    
    
    def fit_on_vocab(self, vocab):
        '''Método que define el vocabulario, generando los diccionarios que
        mapean id's a palabras y palabras a id's.
        
        Parameters
        ----------
        vocab : set
            Set de palabras del vocabulario.
        '''
        # Agregando caracteres especiales
        self.word_index[''] = 0
        self.word_index['<UNK>'] = 1
        self.index_word[0] = ''
        self.index_word[1] = '<UNK>'
        
        # Para cada palabra en el vocabuluario
        for i, word in enumerate(vocab, start=2):
            self.word_index[word] = i
            self.index_word[i] = word
    
    
    def texts_to_sequences(self, texts, split_bool=True):
        '''Método que permite transformar un string de texto a tokens enteros.
        
        Parameters
        ----------
        texts : str
            Texto a tokenizar.
        split_bool : bool, optional
            Booleano que indica si se aplica una separación de texto. Por defecto es
            True.
            
        Returns
        -------
        sequences : list
            Lista de id de tokens representando cada palabra de la oración. 
        '''
        if split_bool:
            sentence_tokens = texts.split(self.split)
        else:
            sentence_tokens = texts
        
        if isinstance(sentence_tokens, str):
            return self.word_index.get(sentence_tokens, self.word_index[''])
        elif isinstance(sentence_tokens, (list, tuple)):
            return [self.word_index.get(word, self.word_index['']) for word in sentence_tokens]
    
    
    def sequences_to_texts(self, sequences):
        '''Método que permite transformar una secuencia de tokens a texto.
        
        Parameters
        ----------
        sequences : list
            Lista que representa la secuencia de tokens de una oración.
            
        Returns
        -------
        texts : list
            Lista de tokens de texto correspondientes a cada id de la lista de secuencia.
        '''
        if isinstance(sequences, (int, float)):
            return self.index_word.get(sequences, self.index_word[0])
        else:
            return [self.index_word.get(i, self.index_word[0]) for i in sequences]


def data_generator(batch_size, data_x, data_y, shuffle=True):
    '''Función generadora obtenida a partir del curso: NLP
    Specialization - Course 3 - NLP with Sequential Models.
    
      Input: 
        batch_size - integer describing the batch size
        data_x - list containing samples
        data_y - list containing labels
        shuffle - Shuffle the data order
      Output:
        a tuple containing 2 elements:
        X - list of dim (batch_size) of samples
        Y - list of dim (batch_size) of labels
    '''
    
    data_lng = len(data_x) # len(data_x) must be equal to len(data_y)
    index_list = [*range(data_lng)] # Create a list with the ordered indexes of sample data
    
    
    # If shuffle is set to true, we traverse the list in a random way
    if shuffle:
        np.random.shuffle(index_list) # Inplace shuffle of the list
    
    index = 0 # Start with the first element
    # START CODE HERE    
    # Fill all the None values with code taking reference of what you learned so far
    while True:
        X = [0] * batch_size # We can create a list with batch_size elements. 
        Y = [0] * batch_size # We can create a list with batch_size elements. 
        
        for i in range(batch_size):
            
            # Wrap the index each time that we reach the end of the list
            if index >= data_lng:
                index = 0
                # Shuffle the index_list if shuffle is true
                if shuffle:
                    np.random.shuffle(index_list) # re-shuffle the order
            
            X[i] = data_x[index_list[index]] # We set the corresponding element in x
            Y[i] = data_y[index_list[index]] # We set the corresponding element in x
    # END CODE HERE            
            index += 1
        
        yield((X, Y))
        

def get_word_embeddings(d_embed, connection, tokenizer_func='custom', 
                        tokenizer_split=' ', table_to_rev='health_effects', 
                        test=False):
    '''Función que permite obtener la matriz de embeddings y el tokenizer 
    utilizando como base el word embedding pre entrenado por Google 
    mediante GloVe.
    
    Parameters
    ----------
    d_embed : int
        Dimensión de incrustado.
    connection : mysql.connector.connection.MySQLConnection
        Objeto que representa la conexión.
    tokenizer_func : {'custom', 'tensorflow'}, optional
        Tipo de tokenizer a utilizar. Por defecto es 'custom'.
    tokenizer_split : str
        Caracter a utilizar para separar cada token en el tokenizer.
    table_to_rev : {'health_effects'}, optional
        Tabla a revisar. Hasta el momento solo está implementado para
        la tabla 'health_effects'.
    test : bool, optional
        Booleano que indica si se trabaja con un conjunto más acotado
        de conceptos (en el marco de mapear conceptos de efectos de 
        salud). 
    
    Returns
    -------
    embedding_matrix : ndarray
        Matriz de embeddings.
    tokenizer : tokenizer object
        Objeto que permitirá realizar el tokenization.
    '''
    # Creación de un diccionario de embeddings
    embeddings_index = dict()
    # Creación de un set de palabras
    word_set = set()
    
    # Cargando el embedding en memoria
    with open(f'C:/Users/Chris-Brota/Desktop/glove.6b/glove.6B.{d_embed}d.txt', 'r', 
              encoding='utf8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            
            # Agregando a las estructuras correspondientes
            embeddings_index[word] = coefs
            word_set.add(word)
    
    # Obtener los conceptos de la tabla de interés
    if table_to_rev == 'health_effects':
        names, descriptions = preprocess_HE_decriptions(connection, test=test)
    
    # Agregando a un set auxiliar de nuevas palabras
    vocab_extra = set()
    for desc in descriptions:
        vocab_extra.update(desc)
    vocab_extra.update(names)
    
    # Revisando en este nuevo diccionario
    for word in vocab_extra:
        if embeddings_index.get(word, None) is None:
            embeddings_index[word] = np.random.uniform(-1, 1, d_embed)
            word_set.add(word)
    
    # Checkeo de sanidad
    assert len(embeddings_index) == len(word_set)
    print(f'Vector de {len(embeddings_index)} palabras cargadas.')
    
    # Tipo de tokenizer a utilizar
    if tokenizer_func == 'tensorflow':
        # Creación del tokenizer
        tokenizer = tf.keras.preprocessing.text.Tokenizer(split=tokenizer_split)
        # Ajustando
        tokenizer.fit_on_texts(list(word_set))
    elif tokenizer_func == 'custom':
        # Creación del tokenizer
        tokenizer = custom_tokenizer(split=tokenizer_split)
        # Ajustando
        tokenizer.fit_on_vocab(list(word_set))
    else:
        raise Exception(f'Opción {tokenizer_func} no disponible para parámetro "tokenizer_func".')
    
    # Definiendo el tamaño del vocabulario
    vocab_size = len(tokenizer.word_index) + 1
    
    # Definición de la matriz de embedding
    embedding_matrix = np.zeros((vocab_size, d_embed))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word, None)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    return embedding_matrix, tokenizer


def get_XY(connection, tokenizer, tokenizer_func='custom', test=False):
    '''Función que permite generar los datos de entrenamiento X, correspondiente
    a cada palabra de dentro de la secuencia de palabras que describe el concepto, 
    y sus respectivas etiquetas Y, correspondiente al concepto de interés.
    
    Parameters
    ----------
    connection : mysql.connector.connection.MySQLConnection
        Objeto que representa la conexión.
    tokenizer : tokenizer object
        Objeto que permitirá realizar el tokenization. 
    tokenizer_func : {'custom', 'tensorflow'}, optional
        Tipo de tokenizer a utilizar. Por defecto es 'custom'.
    test : bool, optional
        Booleano que indica si se trabaja con un conjunto más acotado
        de conceptos (en el marco de mapear conceptos de efectos de 
        salud).
    
    Returns
    -------
    X : list
        Lista palabra a palabra que representan la descripción de 
        cada concepto.
    Y : list
        Lista de los conceptos objetivo a revisar.
    '''
    # Obtener la base de datos preprocesada
    names, descriptions = preprocess_HE_decriptions(connection, test=test)
    
    # Tokenizando cada concepto y los descriptores
    if tokenizer_func == 'tensorflow':
        names_tok = tokenizer.texts_to_sequences(names)
        desc_tok = tokenizer.texts_to_sequences(descriptions)
    elif tokenizer_func == 'custom':
        names_tok = tokenizer.texts_to_sequences(names, split_bool=False)
        desc_tok = [tokenizer.texts_to_sequences(sent, split_bool=False)
                    for sent in descriptions]
    
    # Checkeo de sanidad
    txt_check = "Conceptos y definición con distinto largo."
    assert len(names_tok) == len(desc_tok), txt_check
    
    # Definición de las listas de salida
    X = list()
    Y = list()
    
    # Haciendo el mapeo uno a uno
    for i in range(len(names_tok)):
        for word in desc_tok[i]:
            X.append(names_tok[i])
            Y.append(word)
    
    return X, Y


def get_raw_XY(connection, tokenizer, tokenizer_func='custom', test=False):
    '''Función que permite generar los datos de entrenamiento X, correspondiente
    a una secuencia de palabras, y sus respectivas etiquetas Y, correspondiente
    a sus conceptos.
    
    Parameters
    ----------
    connection : mysql.connector.connection.MySQLConnection
        Objeto que representa la conexión.
    tokenizer : tokenizer object
        Objeto que permitirá realizar el tokenization. 
    tokenizer_func : {'custom', 'tensorflow'}, optional
        Tipo de tokenizer a utilizar. Por defecto es 'custom'.
    test : bool, optional
        Booleano que indica si se trabaja con un conjunto más acotado
        de conceptos (en el marco de mapear conceptos de efectos de 
        salud).
    
    Returns
    -------
    X : list
        Lista de las palabras  que representan la descripción de cada 
        concepto.
    Y : list
        Lista de los conceptos objetivo a revisar.
    '''
    # Obtener la base de datos preprocesada
    names, descriptions = preprocess_HE_decriptions(connection, test=test)
    
    # Tokenizando cada concepto y los descriptores
    if tokenizer_func == 'tensorflow':
        names_tok = tokenizer.texts_to_sequences(names)
        desc_tok = tokenizer.texts_to_sequences(descriptions)
    elif tokenizer_func == 'custom':
        names_tok = tokenizer.texts_to_sequences(names, split_bool=False)
        desc_tok = [tokenizer.texts_to_sequences(sent, split_bool=False)
                    for sent in descriptions]
    
    # Checkeo de sanidad
    txt_check = "Conceptos y definición con distinto largo."
    assert len(names_tok) == len(desc_tok), txt_check
    
    # Aplicando padding
    desc_tok = tf.keras.preprocessing.sequence.pad_sequences(desc_tok, 
                                                             padding="post")
    
    return desc_tok, names_tok


def get_raw_XY_negativeSampling(connection, tokenizer, tokenizer_func='custom', 
                                test=False):
    '''Función que realiza un negative sampling exhaustivo entre los conceptos a 
    obtener y las descripciones. Permite generar los datos de entrenamiento X, 
    correspondiente a una secuencia de palabras; etiquetas Y, correspondiente a 
    los conceptos; y su nivel de relación Z que indica si es que la descripción 
    con la etiqueta son coincidentes.
    
    Parameters
    ----------
    connection : mysql.connector.connection.MySQLConnection
        Objeto que representa la conexión.
    tokenizer : tokenizer object
        Objeto que permitirá realizar el tokenization. 
    tokenizer_func : {'custom', 'tensorflow'}, optional
        Tipo de tokenizer a utilizar. Por defecto es 'custom'.
    test : bool, optional
        Booleano que indica si se trabaja con un conjunto más acotado
        de conceptos (en el marco de mapear conceptos de efectos de 
        salud).
    
    Returns
    -------
    X : list
        Lista de las palabras  que representan la descripción de cada 
        concepto.
    Y : list
        Lista de los conceptos objetivo a revisar. 
    Z : list
        Lista binaria que indica la relación entre los conceptos preesentes en Y
        y las descripciones presentes en X. Es 0 cuando no tienen relación y 1 
        cuando están relacionados.
    '''
    # Obtener la base de datos preprocesada
    names, descriptions = preprocess_HE_decriptions(connection, test=test)
    
    # Tokenizando cada concepto y los descriptores
    if tokenizer_func == 'tensorflow':
        names_tok = tokenizer.texts_to_sequences(names)
        desc_tok = tokenizer.texts_to_sequences(descriptions)
    elif tokenizer_func == 'custom':
        names_tok = tokenizer.texts_to_sequences(names, split_bool=False)
        desc_tok = [tokenizer.texts_to_sequences(sent, split_bool=False)
                    for sent in descriptions]
    
    # Checkeo de sanidad
    txt_check = "Conceptos y definición con distinto largo."
    assert len(names_tok) == len(desc_tok), txt_check
    
    # Aplicando padding
    desc_tok = tf.keras.preprocessing.sequence.pad_sequences(desc_tok, 
                                                             padding="post")
    
    # Definición de listas para las entradas y las salidas (negative sampling)
    X = list()
    Y = list()
    Z = list()
    
    for i in range(len(names_tok)):
        for j in range(len(desc_tok)):
            X.append(desc_tok[j])
            Y.append(names_tok[i])
            if i == j:
                Z.append(1)
            else:
                Z.append(0)
    
    return np.array(X), np.array(Y), np.array(Z)
