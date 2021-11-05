import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression 


def metrics_on_columns(dataframe, epsilon=0, start_idx=6):
    '''Rutina que permite obtener estadísticos de interés de un dataframe 
    sobre cada columna.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        Tabla de pandas que contiene la información a analizar.
    epsilon : float, optional
        Umbral mínimo a partir del cual se considera que un elemento dentro
        de la columna es cero (cuando se necesitan contar entradas cero).
        Por defecto es 0.
    start_idx : int, opional
        Índice de la columna a partir de la cual se comienza a obtener los
        estadísticos. Por defecto es 6.
    
    Returns
    -------
    dataframe_out : pandas.Dataframe
        Dataframe que retorna los estadísticos de interés (media, desv. estándar,
        cantidad de ceros y porcentaje de ceros sobre el total de las columnas) 
        para cada columna.
    '''
    # Definición del diccionario a transformar en tabla
    dict_df = dict()
    
    for i in range(start_idx, dataframe.shape[1]):
        # Columna i
        df_i = dataframe.iloc[:, i]

        # Métricas de interés
        mean_i = df_i.mean()
        std_i  = df_i.std()
        zeros_i = int(sum(abs(df_i) <= epsilon))
        zeros_perc = zeros_i / dataframe.shape[0] * 100
        
        # Llenar el diccionario
        dict_df[df.columns[i]] = [mean_i, std_i, zeros_i, zeros_perc]
    
    # Pasarlo a dataframe
    dataframe_out = pd.DataFrame(dict_df)
    
    # Agregar una columna de etiquetas
    dataframe_out.insert(0, 'metrics', ['mean', 'std', 'zeros', 'zeros (%)']) 
    
    return dataframe_out


def metrics_on_rows(dataframe, epsilon=0, start_idx=6):
    '''Rutina que permite obtener estadísticos de interés de un dataframe 
    sobre cada fila.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        Tabla de pandas que contiene la información a analizar.
    epsilon : float, optional
        Umbral mínimo a partir del cual se considera que un elemento dentro
        de la columna es cero (cuando se necesitan contar entradas cero).
        Por defecto es 0.
    start_idx : int, opional
        Índice de la COLUMNA a partir de la cual se comienza a obtener los
        estadísticos. Por defecto es 6.
    
    Returns
    -------
    dataframe_out : pandas.Dataframe
        Dataframe que retorna los estadísticos de interés (media, desv. estándar,
        cantidad de ceros y porcentaje de ceros sobre el total de las columnas) 
        para cada fila.
    '''
    # Definición de la lista de filas
    list_rows = list()
    
    # Definición de los nombres de las primeras columnas
    col_names = [i for i in dataframe][:6]
    
    for i in range(dataframe.shape[0]):
        # Columna i
        df_i = dataframe.iloc[i, start_idx:]
        
        # Métricas de interés
        mean_i = df_i.mean()
        std_i  = df_i.std()
        zeros_i = int(sum(abs(df_i) <= epsilon))
        zeros_perc = zeros_i / dataframe.shape[1] * 100
        
        # Llenar el diccionario
        list_rows.append([dataframe['id_food_table'][i],
                          dataframe['orig_food_id'][i],
                          dataframe['food_group'][i],
                          dataframe['food_subgroup'][i],
                          dataframe['nombre_primario'][i],
                          dataframe['nombre_secundario'][i], 
                          mean_i, std_i, zeros_i, zeros_perc])
    
    # Pasarlo a dataframe
    dataframe_out = pd.DataFrame(list_rows, 
                                 columns=col_names + ['mean', 'std',
                                                      'zeros',
                                                      'zeros (%)'])
    
    return dataframe_out


def get_TSNE_projection(dataframe, filter_type='all', start_idx=6, plot_tsne=False,
                        save_plot=False, save_data=False):
    '''Rutina que permite obtener la proyección T-SNE para el dataframe de alimentos.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        Tabla de pandas que contiene la información a analizar.
    filter_type : {'all', 'AE', 'ANE', 'G', 'M', 'OT', 'V'}, optional
        Tipo de filtro que se le aplica a los compuestos de cada alimento. Por ejemplo,
        al usar 'ANE' se escogen como compuestos solo los aminoácidos no esenciales.
        Por defecto es 'all'.
    start_idx : int, opional
        Índice de la COLUMNA a partir de la cual se comienza a obtener los
        estadísticos. Por defecto es 6.
    plot_tsne : bool, optional
        Booleano que indica si se plotea los resultados del T-SNE. Por defecto es False.
    save_plot : bool, optional
        Booleano que indica si se guarda el plot del T-SNE. Por defecto es False.
    save_data : bool, optional
        Booleano que indica si es que se guardan los datos obtenidos a partir de la
        reducción T-SNE. Por defecto es False.
        
    Returns
    -------
    T_foods : ndarray
        Matriz que contiene la información de los datos alimentos pero en una
        dimensión reducida.
    labels : ndarray
        Cluster para cada uno de los alimentos.
    '''
    # Definición de la lista que almacena los vectores
    food_vectors = list()
    
    # Definición del conjunto de etiquetas/grupos de los alimentos
    group_labels = sorted(list(set([i.split('_')[0] for i in df.columns[6:]])) + ['all'])
    
    # Asegurándose de que la etiqueta se encuentra en el dataframe
    if filter_type in group_labels:
        # Agregando los valores a la lista
        if filter_type == 'all':
            idx_to_append = [i for i in range(start_idx, df.shape[1])]
        elif filter_type == 'AE':
            idx_to_append = [i for i, col in enumerate(dataframe.columns)
                             if col.split('_')[0] == 'AE']
        elif filter_type == 'ANE':
            idx_to_append = [i for i, col in enumerate(dataframe.columns)
                             if col.split('_')[0] == 'ANE']
        elif filter_type == 'G':
            idx_to_append = [i for i, col in enumerate(dataframe.columns)
                             if col.split('_')[0] == 'G']
        elif filter_type == 'M':
            idx_to_append = [i for i, col in enumerate(dataframe.columns)
                             if col.split('_')[0] == 'M']
        elif filter_type == 'OT':
            idx_to_append = [i for i, col in enumerate(dataframe.columns)
                             if col.split('_')[0] == 'OT']
        elif filter_type == 'V':
            idx_to_append = [i for i, col in enumerate(dataframe.columns)
                             if col.split('_')[0] == 'V']
    else:
        raise ValueError(f'"{filter_type}" no es una opción permitida para "filter_type"')
    
    # Para cada fila
    for i, row in dataframe.iterrows():
        food_vectors.append(list(row[idx_to_append]))
    
    # Transformando a array
    food_vectors = np.array(food_vectors)
    
    # Aplicando el algoritmo T-SNE
    tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=3)
    T_foods = tsne.fit_transform(food_vectors)
    
    # Definición de las etiquetas
    labels = dataframe['nombre_secundario']
    
    # Opción de graficar
    if plot_tsne:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=T_foods[:, 0],
                                 y=T_foods[:, 1],
                                 text=labels,
                                 textposition='top center',
                                 mode='markers+text'))
        fig.update_traces(textfont_size=8)
        fig.update_layout(title='Análisis de alimentos objetivos FooDB: '
                                'TSNE sobre Word Embeddings')
        if save_plot:
            fig.write_html(f'Results/TSNE_foodSpace_{filter_type}.html')
        fig.show()
        
    # Opción de guardar valores
    if save_data:
        to_save = (T_foods, labels, 
                   dataframe.iloc[:, [i for i in range(start_idx)] + idx_to_append])
        
        with open(f'Results/TSNE_foodSpace_{filter_type}.pkl', 'wb') as file:
            pickle.dump(to_save, file)
        
    return T_foods, labels


def get_clusters_df(dataframe, filter_type='all', cluster_op=None, start_idx=6, 
                    plot_clusters=False, save_plot=False, save_data=False, 
                    cluster_on_tsne=True, extra_label=''):
    '''Rutina que permite generar los clusters a partir del dataframe original. 
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        Tabla de pandas que contiene la información a analizar.
    filter_type : {'all', 'AE', 'ANE', 'G', 'M', 'OT', 'V'}, optional
        Tipo de filtro que se le aplica a los compuestos de cada alimento. Por ejemplo,
        al usar 'ANE' se escogen como compuestos solo los aminoácidos no esenciales.
        Por defecto es 'all'.
    cluster_op : dict or None
        Diccionario que contiene las opciones 'eps' (epsilon, referida al parámetro
        epsilon del DBSCAN) y 'min_samples' (referida a la cantidad mínima de muestras
        del DBSCAN). Por defecto es None.
    start_idx : int, opional
        Índice de la COLUMNA a partir de la cual se comienza a obtener los
        estadísticos. Por defecto es 6.
    plot_clusters : bool, optional
        Booleano que indica si se plotea los resultados del T-SNE. Por defecto es False.
    save_plot : bool, optional
        Booleano que indica si se guarda el plot del T-SNE. Por defecto es False.
    save_data : bool, optional
        Booleano que indica si es que se guardan los datos obtenidos a partir de la
        reducción T-SNE. Por defecto es False.
    cluster_on_tsne : bool, optional
        Booleano que indica si es que se aplica el algoritmo de clustering sobre
        los datos en la dimensión reducida, o sobre la dimensión original.
    extra_label : str, optional
        Label adicional que se agrega para poder guardar los archivos con un nombre
        característico.
        
    Returns
    -------
    T_foods : ndarray
        Matriz que contiene la información de los datos alimentos pero en una
        dimensión reducida.
    labels : ndarray
        Cluster para cada uno de los alimentos.
    clusters : ndarray
        Etiquetas de cluster para cada uno de los ingredientes.
    '''
    def _get_clusters(features):
        # Aplicando un algoritmo de clustering DBSCAN
        dbscan = DBSCAN(eps=cluster_op['eps'], min_samples=cluster_op['min_samples'])
        clustering = dbscan.fit(features)
        clusters = clustering.labels_
        return clusters
    
    
    # Definición de la lista que almacena los vectores
    food_vectors = list()
    
    # Definición del conjunto de etiquetas/grupos de los alimentos
    group_labels = sorted(list(set([i.split('_')[0] 
                                    for i in dataframe.columns[6:]])) + ['all'])
    
    # Asegurándose de que la etiqueta se encuentra en el dataframe
    if filter_type in group_labels:
        # Agregando los valores a la lista
        if filter_type == 'all':
            idx_to_append = [i for i in range(start_idx, dataframe.shape[1])]
        elif filter_type == 'AE':
            idx_to_append = [i for i, col in enumerate(dataframe.columns)
                             if col.split('_')[0] == 'AE']
        elif filter_type == 'ANE':
            idx_to_append = [i for i, col in enumerate(dataframe.columns)
                             if col.split('_')[0] == 'ANE']
        elif filter_type == 'G':
            idx_to_append = [i for i, col in enumerate(dataframe.columns)
                             if col.split('_')[0] == 'G']
        elif filter_type == 'M':
            idx_to_append = [i for i, col in enumerate(dataframe.columns)
                             if col.split('_')[0] == 'M']
        elif filter_type == 'OT':
            idx_to_append = [i for i, col in enumerate(dataframe.columns)
                             if col.split('_')[0] == 'OT']
        elif filter_type == 'V':
            idx_to_append = [i for i, col in enumerate(dataframe.columns)
                             if col.split('_')[0] == 'V']
    else:
        raise ValueError(f'"{filter_type}" no es una opción permitida para "filter_type"')
    
    
    # Para cada fila
    for i, row in dataframe.iterrows():
        food_vectors.append(list(row[idx_to_append]))
    
    # Transformando a array
    food_vectors = np.array(food_vectors)
    
    # Aplicando el algoritmo T-SNE
    tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=3)
    T_foods = tsne.fit_transform(food_vectors)
    
    # Definición de las etiquetas
    labels = dataframe['nombre_secundario']
    
    # Opción de cluster
    if cluster_on_tsne:
        # Obteniendo los clusters mediante T-SNE
        clusters = _get_clusters(T_foods)
    else:
        # Obteniendo los clusters mediante valores originales
        clusters = _get_clusters(food_vectors)
    
    # Opción de guardar valores
    if save_data:
        to_save = (T_foods, labels, 
                   dataframe.iloc[:, [i for i in range(start_idx)] + idx_to_append],
                   clusters)
        
        if extra_label != '':
            filename = f'Results/TSNE_foodSpace_{filter_type}_{extra_label}.pkl'
        else:
            filename = f'Results/TSNE_foodSpace_{filter_type}.pkl'
        
        with open(filename, 'wb') as file:
            pickle.dump(to_save, file)
        
    # Creando un dataframe que condense toda la información
    dfi = np.concatenate((T_foods, np.array([labels]).T, np.array([clusters]).T), axis=1)
    dfi = pd.DataFrame(dfi, columns=['x1', 'x2', 'label', 'cluster'])
    
    if plot_clusters:
        fig = px.scatter(dfi, x='x1', y='x2', color='cluster', labels='label',
                         hover_data=['label', 'cluster'], text='label')
        fig.update_traces(textfont_size=7, textposition='top center')
        fig.update_traces(marker={'size': 15, 'opacity': 0.5})
        fig.update_layout(title='Análisis de ingredientes objetivos FooDB: '
                                f'TSNE con filtro {filter_type}.')
        if save_plot:
            if extra_label != '':
                filename = f'Results/TSNE_foodSpace_{filter_type}_{extra_label}.html'
            else:
                filename = f'Results/TSNE_foodSpace_{filter_type}.html'
            fig.write_html(filename)
        fig.show()
    
    return T_foods, labels, clusters


def get_logreg_weights(dataframe, clusters, lr_params):
    '''Rutina que permite obtener los pesos de una regresión logística
    sobre las columnas de datos para un dataframe de interés. Se 
    obtiene un peso para cada columna.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        Tabla de pandas que contiene la información a analizar.
    clusters : ndarray
        Etiquetas de cluster para cada uno de los ingredientes.
    lr_params : dict
        Diccionario que contiene los parámetros de "random_state",
        "max_iter", "verbose" y "solver" de la regresión logística
        de la librería sklearn.
        
    Returns
    -------
    weights : ndarray
        Pesos de la regresión logística para cada una de las columnas 
        del dataframe.
    '''
    # Obtener los valores de entrada y salida
    X = dataframe.to_numpy()
    # Y = np.array(clusters == cluster_i, dtype=int)
    Y = clusters
    
    # Obtener el clasificador de regresión logística
    logreg = LogisticRegression(random_state=lr_params['random_state'], 
                                max_iter=lr_params['max_iter'], 
                                verbose=lr_params['verbose'], 
                                solver=lr_params['solver'])
    
    # Ajustando
    logreg = logreg.fit(X, Y)
    
    # Obteniendo los pesos
    weights = logreg.coef_
    
    return weights


def order_weights(dataframe, clusters, weights):
    '''Rutina que entrega una lista ordenada de los pesos más importantes en
    base a la regresión logística.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        Tabla de pandas que contiene la información a analizar.
    clusters : ndarray
        Etiquetas de cluster para cada uno de los ingredientes.
    weights : ndarray
        Pesos de la regresión logística para cada una de las columnas 
        del dataframe.
        
    Returns
    -------
    dict_out : dict
        Diccionario que, para cada cluster, contiene los nombres ordenados 
        de las variables que más peso tienen en el cluster.
    '''
    # Definición del número de clusters
    clusters = np.unique(clusters)
    
    # Definición del nombre de las columnas de la tabla
    col_names = tuple(dataframe.columns)
    
    # Definición del diccionario de pesos ordenados para cada cluster
    dict_out = dict()
    
    # Revisando cada cluster
    for i, clus in enumerate(clusters):
        # Pesos de cada cluster con nombres
        named_weights = [(col_names[num], w) for num, w in enumerate(weights[i])]
        # Ordenando por los valores
        named_weights.sort(key=lambda x: x[1], reverse=True)
        
        # Agregando al diccionario
        dict_out[clus] = named_weights
        
    return dict_out


def get_heatmap(dataframe, clusters, weights, plot_type='matplotlib',
                cmap='inferno', logheatmap=False, eps=1e-6):
    '''Rutina que permite retornar una tabla con los pesos asociados
    a cada compuesto para cada cluster distinto.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        Tabla de pandas que contiene la información a analizar.
    clusters : ndarray
        Etiquetas de cluster para cada uno de los ingredientes.
    weights : ndarray
        Pesos de la regresión logística para cada una de las columnas 
        del dataframe.
    plot_type : {'matplotlib', 'pandas'}, optional
        Tipo de método para el plot. Por defecto es 'matplotlib'.
    cmap : str, optional
        Colormap a utilizar. Por defecto se usa 'inferno'. Para más
        opciones, se recomienda revisar: 
        https://matplotlib.org/stable/tutorials/colors/colormaps.html
    logheatmap : bool, optional
        Booleano que permite graficar el mapa de calor en una escala
        logarítmica. Por defecto es False.
    eps : float, optional
        Cantidad ínfima a agregar al logheatmap para evitar indefiniciones.
        Por defecto es 1e-6.
    
    Returns
    -------
    dataframe_out : pandas.dataframe
        Dataframe con que contiene los pesos y los clusters.
    '''
    # Definición del nombre de cada componente
    compound_names = tuple(dataframe.columns)
    
    # Definición de los nombres de las columnas de la tabla (clusters)
    col_names = [f'Cluster {c}' for c in np.unique(clusters)]
    
    # Creación del dataframe
    dataframe_out = pd.DataFrame(np.exp(weights.T), index=compound_names, 
                                 columns=col_names)
    
    if plot_type == 'matplotlib':
        fig, ax = plt.subplots(figsize=(9,12))
        if not logheatmap:
            im = ax.pcolormesh(np.exp(weights.T), cmap=cmap)
        else:
            im = ax.pcolormesh(np.log(weights.T - np.min(weights) + eps), 
                               cmap=cmap)
        plt.gca().invert_yaxis()
        fig.colorbar(im)

        # Setear la cantidad de ticks...
        ax.set_xticks(np.arange(weights.shape[0]) + 0.5)
        ax.set_yticks(np.arange(weights.shape[1]) + 0.5)
        # Y asignando las etiquetas
        ax.set_yticklabels(compound_names)
        ax.set_xticklabels(col_names)

        # Rotando y alineando.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor", )

        ax.set_title("Coeficientes para cada cluster")
        fig.tight_layout()
        plt.show()
    
    elif plot_type == 'pandas':
        if not logheatmap:
            print(dataframe_out.style.background_gradient(cmap=cmap))
        else:
            pass
        
    return dataframe_out


def get_heatmap_classicStats(dataframe, clusters, cmap='inferno', logheatmap=False, 
                             metric='mean'):
    '''Rutina que permite retornar una tabla con los estadísticos clásicos
    de interés asociados a cada compuesto para cada cluster distinto.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        Tabla de pandas que contiene la información a analizar.
    clusters : ndarray
        Etiquetas de cluster para cada uno de los ingredientes.
    cmap : str, optional
        Colormap a utilizar. Por defecto se usa 'inferno'. Para más
        opciones, se recomienda revisar: 
        https://matplotlib.org/stable/tutorials/colors/colormaps.html
    logheatmap : bool, optional
        Booleano que permite graficar el mapa de calor en una escala
        logarítmica. Sin embargo, no está implementada la opción 
        logarítmica. Por defecto es False.
    metric : {'mean', 'std', 'median', 'zeros'}, optional
        Métrica acerca de la cual se construirá la tabla. Por defecto 
        es 'mean'.
    
    Returns
    -------
    dataframe_out : pandas.dataframe
        Dataframe con que contiene los pesos y los clusters.
    '''
    # Definición de las lista de interés para cada cluster
    col_names = list()
    metric_vals = list()
    
    for c in np.unique(clusters):
        # Definición de los nombres de las columnas
        col_names.append(f'Cluster {c}')
        
        # Obtener las métricas
        if metric == 'mean':
            metric_vals.append(dataframe.loc[np.where(clusters == c)].mean(axis=0))
        elif metric == 'std':
            metric_vals.append(dataframe.loc[np.where(clusters == c)].std(axis=0))
        elif metric == 'median':
            metric_vals.append(dataframe.loc[np.where(clusters == c)].median(axis=0))
        elif metric == 'zeros':
            metric_vals.append(np.sum(dataframe.loc[np.where(clusters == c)] == 0, axis=0))
    
    # Creación del dataframe
    dataframe_out = pd.concat(metric_vals, axis=1)
    dataframe_out.columns = col_names
    
    if not logheatmap:
        print(dataframe_out.style.background_gradient(cmap=cmap, axis=1))
    else:
        pass
        
    return dataframe_out 


# Módulo de testeo
if __name__ == '__main__':
    df = pd.read_csv('../FOOD_AN/CONSOLIDADO MVP v2_id_aux.dat', sep='|', decimal='.')
    # Opciones de cluster
    cluster_op = {'eps': 20, 'min_samples': 5}
    
    # Prueba de get_clusters
    get_clusters_df(df, cluster_op=cluster_op, plot_clusters=True)
