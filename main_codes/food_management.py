import pandas as pd


def get_ingredients(filedir, index_ingredients, start_index=6, sep='|', decimal='.'):
    '''Función que retorna los valores numéricos de los ingredientes de interés
    de la base de datos depurada.
    
    Parameters
    ----------
    filedir : str
        Dirección del archivo con la base de datos.
    index_ingredients : list
        Lista con los índices de los ingredientes de interés a obtener.
    start_idx : int, opional
        Índice de la COLUMNA a partir de la cual se comienza a obtener los
        estadísticos. Por defecto es 6.
    sep : str, optional
        Caracter utilizado para la separación de datos en el archivo a leer.
        Por defecto es '|'.
    decimal : str, optional
        Tipo de separación de números decimales. Por defecto es '.'.
        
    Returns
    -------
    values : pandas.DataFrame
        Valores obtenidos para cada ingrediente de interés.
    labels : pandas.DataFrame
        Etiquetas que contienen el "nombre_primario" y "nombre_secundario" de los 
        ingredientes de interés de la base.
    '''
    # Obtención del dataframe
    df = pd.read_csv(filedir, sep=sep, decimal=decimal)
    
    # Obtención de los valores de cada alimento
    values = df.iloc[index_ingredients, start_index:]
    
    # Obtención de las etiquetas (nombre) de cada alimento
    labels = df.loc[index_ingredients, ['nombre_primario', 'nombre_secundario']]
    
    return values, labels


def get_ingredients_comps(filedir, index_ingredients, 
                          specific_comps=['G_carbohydrate', 'G_protein'], 
                          sep='|', decimal='.'):
    '''Función que retorna los valores numéricos de los ingredientes de interés
    de la base de datos depurada, filtrando únicamente sobre componentes específicos.
    Esta función puede ser útil para obtener las matrices que se utilizarán para
    las restricciones del modelo de optimización.
    
    Parameters
    ----------
    filedir : str
        Dirección del archivo con la base de datos.
    index_ingredients : list
        Lista con los índices de los ingredientes de interés a obtener.
    specific_comps : list, opional
        Nombre de los compuestos de interés a incorporar. Es necesario que sean 
        los nombres en el encabezado de la base de datos. Por defecto es 
        ['G_carbohydrate', 'G_protein'].
    sep : str, optional
        Caracter utilizado para la separación de datos en el archivo a leer.
        Por defecto es '|'.
    decimal : str, optional
        Tipo de separación de números decimales. Por defecto es '.'.
        
    Returns
    -------
    values : pandas.DataFrame
        Valores obtenidos para cada ingrediente de interés.
    labels : pandas.DataFrame
        Etiquetas que contienen el "nombre_primario" y "nombre_secundario" de los 
        ingredientes de interés de la base.
    '''
    # Obtención del dataframe
    df = pd.read_csv(filedir, sep=sep, decimal=decimal)
    
    # Obtención de los valores de cada alimento
    values = df.loc[index_ingredients, specific_comps]
    
    # Obtención de las etiquetas (nombre) de cada alimento
    labels = df.loc[index_ingredients, ['nombre_primario', 'nombre_secundario']]

    return values, labels


# Módulo de testeo
if __name__ == '__main__':
    # Dirección del archivo a revisar
    filedir = '../FOOD_AN/CONSOLIDADO MVP v4 (sin En)_id_aux.dat'
    index_ings = [103,104,105]
    sp_comps = ['M_potassium', 'M_sodium']
    v, l = get_ingredients_comps(filedir, index_ings, specific_comps=sp_comps, 
                                 sep='|', decimal=',')

    print(v)
    print(l)
