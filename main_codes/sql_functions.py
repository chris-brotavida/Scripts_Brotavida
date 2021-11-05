import mysql.connector
import re, getpass
import pandas as pd
from nltk.corpus import stopwords


def get_sql_connection(connection_params):
    '''Rutina que permite generar la conexión con la base de datos
    MySQL de interés
    
    Parameters
    ----------
    connection_params : dict
        Diccionario que contiene los parámetros de conexión. 
        Por ejemplo:
        {'host': 'localhost',
        'user': 'default',
        'database': 'foodb',
        'password': ""'}
        
    Returns
    -------
    connection : mysql.connector.connection.MySQLConnection
        Objeto que permite generar la conexión con la base de datos.
    cursor : mysql.connector.cursor.MySQLCursor
        Objeto que permite recorrer la base de datos de interés.
    '''
    try:
        connection = mysql.connector.connect(**connection_params)

        if connection.is_connected():
            db_Info = connection.get_server_info()
            print("Connected to MySQL Server version ", db_Info)
            cursor = connection.cursor()

    except mysql.connector.Error as e:
        print("Error while connecting to MySQL", e)
        
    return connection, cursor


def foodb_ingds_wiki(connection):
    '''Rutina que retorna un dataframe con los nombres de los ingredientes, 
    sus respectivas descripciones y sus id de wikipedia.
    
    Parameters
    ----------
    connection : mysql.connector.connection.MySQLConnection
        Objeto que representa la conexión con la base de datos.
            
    Returns
    -------
    dataframe : pandas.core.frame.DataFrame
        Dataframe con la información solicitada.
    '''
    # Definición de la query
    query = \
    ''' 
        SELECT f.id, f.name, f.name_scientific, f.description, f.wikipedia_id
        FROM foods f
        WHERE f.wikipedia_id IS NOT NULL
        AND f.wikipedia_id != '';
    '''

    # Solicitud de la query
    return pd.read_sql(query, connection)


def foodb_health_effects(connection):
    '''Rutina que retorna un dataframe con los efectos de salud y sus 
    respectivas descripciones.
    
    Parameters
    ----------
    connection : mysql.connector.connection.MySQLConnection
        Objeto que representa la conexión con la base de datos.
            
    Returns
    -------
    dataframe : pandas.core.frame.DataFrame
        Dataframe con la información solicitada.
    '''
    query = \
    ''' 
        SELECT name, chebi_name, IF(description IS NULL, chebi_definition, description) as definitions 
        FROM foodb.health_effects he
        WHERE description IS NOT NULL OR chebi_definition IS NOT NULL
    '''
    
    # Solicitud de la query
    return pd.read_sql(query, connection)


def preprocess_HE_decriptions(connection, test=False, 
                              filename_test='../NLP_INGDS_1/Summary/names_to_rev.txt'):
    '''Rutina que permite entregar los efectos de salud en conjunto con sus
    descripciones.
    
    Parameters
    ----------
    connection : mysql.connector.connection.MySQLConnection
        Objeto que representa la conexión con la base de datos.
    test : bool, optional
        Booleano que indica si es que se utiliza un conjunto acotado 
        de testeo, el cual estará definido en la dirección entregada 
        por "filename_test". Por defecto es False.
    filename_test : str, optional
        Dirección del archivo donde se encuentran los conceptos de testeo.
        Por defecto es '../NLP_INGDS_1/Summary/names_to_rev.txt'.
            
    Returns
    -------
    names : list
        Lista de los tokens de los conceptos con efectos en salud.
    descriptions : list
        Lista de los tokens de las descripciones para cada uno de los efectos 
        en salud.
    '''
    # Definición de la query que se le hace a la base de datos
    query = \
    '''
        SELECT name, chebi_name, IF(description IS NULL, chebi_definition, description) as definitions 
        FROM foodb.health_effects he
        WHERE description IS NOT NULL OR chebi_definition IS NOT NULL
    '''

    # Obtener el dataframe
    dataframe = pd.read_sql(query, con=connection)
    
    # Acortar para un testeo sencillo
    if test:
        with open(filename_test, 'r', encoding='utf8') as file:
            concepts = list()
            for line in file:
                concepts.append(line.strip())

        dataframe = dataframe[dataframe['name'].isin(concepts)]

    # Definición de los tokens de nombres a agregar
    names = list(dataframe['name'])
    
    # Definición de los tokens de descripción
    descriptions = list()
    
    for d in dataframe['definitions']:
        # En primer lugar eliminar el salto de linea
        txt = d.strip()

        # Eliminando los puntos y comas
        txt = ''.join(re.findall('[\w\s]+', txt))

        # Reemplazando los espacios múltiples con espacios
        txt = re.sub('\s+', ' ', txt)

        # Pasando a minúsculas
        txt = txt.lower()

        # Filtrando palabras stop
        txt = [i for i in txt.split() if not i in stopwords.words('english')]
        
        # Agregando a la lista
        descriptions.append(txt)
        
    return names, descriptions


# Módulo de testeo
if __name__ == '__main__':
    connection_params = {
        'host': 'localhost',
        'user': 'default',
        'database': 'foodb',
        'password': getpass.getpass(prompt='Introduzca la contraseña: ')
    }
    
    connection = get_sql_connection(connection_params)
    