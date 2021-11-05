import pickle
import numpy as np
import tensorflow as tf
from nlp_data_management import preprocess_HE_decriptions, custom_tokenizer
from sql_functions import get_sql_connection


def get_model(model_filepath, tokenizer_filepath):
    '''Rutina que permite obtener el modelo entrenado y su correspondiente 
    tokenizer.
    '''
    # Obtener el modelo
    model = tf.keras.models.load_model(model_filepath)

    # Obtener el tokenizer
    with open(tokenizer_filepath, 'rb') as file:
        tokenizer = pickle.load(file)

    return model, tokenizer
    
    
def model_testing(model, sentence, tokenizer, connection, 
                  tokenizer_type='custom', split_bool=False, 
                  test=False):
    # Definición de los conceptos clave a buscar
    names, descriptions = preprocess_HE_decriptions(connection, test=test)
    
    # Obtener los id de la oración
    if tokenizer_type == 'custom':
        tokens = tokenizer.texts_to_sequences(sentence, split_bool=split_bool)
        names_toks = tokenizer.texts_to_sequences(names, split_bool=False)
    elif tokenizer_type == 'tensorflow':
        tokens = tokenizer.texts_to_sequences(sentence)
        names_toks = tokenizer.texts_to_sequences(names)
    
    # Aplicando padding
    tokens = tf.keras.preprocessing.sequence.pad_sequences([tokens], maxlen=25, padding='post')
    
    # Definición de la lista de resultados
    results = list()
    
    for i, name_i in enumerate(names_toks):
        y_pred = model.predict(x={'Input_context': tokens, 'Input_target': np.array([name_i])})
        results.append((names[i], round(y_pred[0,0] * 100, 3), descriptions[i]))
    
    # Ordenando
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results


def query_embeddings(sentence, tokenizer, embedding_matrix, connection, 
                     tokenizer_type='custom', split_bool=False, 
                     distance_metric='euclidean', test=False):
    # Definición de los conceptos clave a buscar
    names, descriptions = preprocess_HE_decriptions(connection, test=test)
    
    # Obtener los id de la oración
    if tokenizer_type == 'custom':
        tokens = tokenizer.texts_to_sequences(sentence, split_bool=split_bool)
        names_toks = tokenizer.texts_to_sequences(names, split_bool=False)
    elif tokenizer_type == 'tensorflow':
        tokens = tokenizer.texts_to_sequences(sentence)
        names_toks = tokenizer.texts_to_sequences(names)
    
    # Obtener los embeddings asociados
    embedded_tokens = embedding_matrix[tokens]
    embedded_names = embedding_matrix[names_toks]
    
    # Calculando una distancia
    if distance_metric == 'cosine':
        metric = np.dot(embedded_names, embedded_tokens.T)
        metric = metric / np.linalg.norm(embedded_names, 2, axis=1)[:, np.newaxis]
        metric = metric / np.linalg.norm(embedded_tokens, 2, axis=1)[np.newaxis, :]
        metric = metric.mean(axis=1)
    elif distance_metric == 'euclidean':
        metric = list()
        for emb_tok in embedded_tokens:
            dists = np.sum((embedded_names - emb_tok) ** 2, axis=1)
            metric.append(dists)
        metric = np.array(metric)
        metric = metric.mean(axis=0)
    
    # Ordenando las métricas
    metrics_key = [(i, value, descriptions[i]) for i, value in enumerate(metric)]
    if distance_metric == 'cosine':
        metrics_key.sort(key=lambda x: x[1], reverse=True)
    elif distance_metric == 'euclidean':
        metrics_key.sort(key=lambda x: x[1], reverse=False)
    
    # Transformando a palabras
    concepts = [(names[i[0]], i[1], i[2]) for i in metrics_key]
    
    return concepts


# Módulo de testeo
if __name__ == '__main__':
    # Generar conexión SQL
    connection_params = {
        'host': 'localhost',
        'user': 'default',
        'database': 'foodb',
        'password': ''
    }
    connection, _ = get_sql_connection(connection_params)
    
    # Dirección del modelo y tokenizer
    folder_to_rev = '../NLP_INGDS_1/Results/model_CBOW_1_Siamese_traditional_test_v2(masking)/'
    model_filepath = f'{folder_to_rev}/model.h5'
    tokenizer_filepath = f'{folder_to_rev}/tokenizer.pkl'
    model, tokenizer = get_model(model_filepath, tokenizer_filepath)
    print(model.summary())
    
    # Ejemplos de prueba
    sentence = ['hormone', 'stimulation', 'male', 'features']
    print(f'{sentence}\n------------------------------------------------')
    a = model_testing(model, sentence, tokenizer, connection, 
                    tokenizer_type='custom', split_bool=False, 
                    test=True)
    for i in a:
        print(i)
    print('\n\n')

    sentence = ['change', 'heart', 'rate']
    print(f'{sentence}\n------------------------------------------------')
    a = model_testing(model, sentence, tokenizer, connection, 
                    tokenizer_type='custom', split_bool=False, 
                    test=True)
    for i in a:
        print(i)
    print('\n\n')
        
    sentence = ['energy']
    print(f'{sentence}\n------------------------------------------------')
    a = model_testing(model, sentence, tokenizer, connection, 
                    tokenizer_type='custom', split_bool=False, 
                    test=True)
    for i in a:
        print(i)
    print('\n\n')  

    sentence = ['less', 'sensation', 'low']
    print(f'{sentence}\n------------------------------------------------')
    a = model_testing(model, sentence, tokenizer, connection, 
                    tokenizer_type='custom', split_bool=False, 
                    test=True)
    for i in a:
        print(i)
