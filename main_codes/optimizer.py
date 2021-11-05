import numpy as np
import cvxpy as cp
from scipy.spatial import distance


def optimizer(ingredients_matrix, obj_food, solver=cp.SCS, verbose=False, 
              grames=None, weighted=False, alpha_weight=100,
              min_ingredients=None, min_x=None, K=None, D=None, CP=None,
              thr_mining=0, M=1e9):
    '''Función que permite generar las recetas de un producto, retornando
    unidades adimensionales de cada elemento que lo conforma.
    
    Parameters
    ----------
    ingredients_matrix : ndarray
        Matriz de ingredientes a usar. Cada fila corresponde a un ingrediente
        distinto, y cada columna a un compuesto del ingrediente. Cada fila de 
        la matriz solo debe contener valores numéricos que indiquen la 
        cantidad de cada compuesto en el alimento.
    obj_food : ndarray o pandas.DataFrame
        Vector del alimento objetivo a generar. Debe tener la misma cantidad de 
        compuestos que los alimentos de la matriz de ingredientes.
    solver : str, optional
        Tipo de solver a utilizar para el problema de optimización. Por 
        defecto es 'ECOS'.
    verbose : bool, optional
        Booleano que define los prints para cada problema de optimización.
        Por defecto es False.
    grames : float, optional
        Dimensión del producto (en gramos) que se espera a la salida. Por
        defecto es 250g.
    weighted : bool, optional
        Si es que se pondera cada compuesto en base a su presencia en el 
        alimento. Por defecto es False.
    alpha_weight : float, optional
        Constante que se le añade al peso cuando weighted es True. Por defecto
        es 100.
    min_ingredients : None or int, optional
        Cantidad mínima de ingredientes a usar en la preparación. Por 
        defecto es None.
    min_x : list or None, optional
        Cantidad mínima explícita de un alimento en la receta. Por defecto es 
        None.
    K : ndarray or None, optional
        Matriz cuyas columnas corresponden a la cantidad de cafeína de cada 
        alimento (fila). Cuando no es None, se activa esta restricción. 
        Por defecto es None.
    D : ndarray or None, optional
        Matriz cuyas columnas corresponden a las cantidades de sodio y 
        potasio de cada alimento (fila). Cuando no es None, se activa esta 
        restricción. Por defecto es None.
    CP : ndarray or None, optional
        Matriz cuyas columnas corresponden a las cantidades de carbohidratos y 
        protenias de cada alimento (fila). Cuando no es None, se activa esta 
        restricción. Por defecto es None.
    thr_mining: int, optional
        En caso de definir un valor para la cantidad mínima de ingredientes
        con valores distintos de cero con "min_ingredients", este parámetro
        define el valor mínimo a partir del cual se considera que la variable
        binaria y se activa. Por defecto es 0.
    M : float, optional
        Parámetro del modelo para definir valores cercanos a infinito. Por
        defecto es 1e9.
        
    Returns
    -------
    info_dict : dict
        Diccionario con información útil para la elaboración del producto.
        - ['result']: Obtiene el valor de la función objetivo.
        - ['x']: Obtiene el valor del vector x.
        - ['y']: Retorna el valor del vector y.
        - ['euc_dist']: Retorna el valor de la distancia euclidiana entre
                        la tabla generada y la objetivo.
        - ['cos_sim']: Retorna el valor de la similaridad coseno entre
                       la tabla generada y la objetivo.
        - ['T_gen']: Retorna el valor de la tabla generada.
    '''
    def _constraints(x, y):
        '''Rutina que inicializa las restricciones del problema
        
        Parámetros
        ----------
        x : cvxpy.expressions.variable.Variable
            Variable definida para la cantidad de cada ingrediente en la
            receta.
        y : cvxpy.expressions.variable.Variable
            Variable definida para la activación de cada ingrediente en la
            receta. Es decir, 1 si es que x >= thr_mining, 0 en caso 
            contrario.
        '''
        # Definición de las restricciones como funciones
        def __base_const():
            '''Función que presenta la naturaleza de la variable de cantidad
            '''
            return x >= 0
        
        
        def __sport_const():
            '''Restricción de sodio y potasio (Art. 540 h.)
            '''
            # Obteniendo la restricción (sodio, potasio)
            return D.T @ x <= np.array([1.610, 3.715]) # gramos (1610, 3715) mg
        
        
        def __y_up_activation_const():
            '''Restricción de activación de la variable y
            Referencia: https://or.stackexchange.com/questions/33/
                        in-an-integer-program-how-i-can-force-a-
                        binary-variable-to-equal-1-if-some-cond
            '''
            return x - thr_mining <= y * M
        
        
        def __y_down_activation_const():
            '''Restricción de activación de la variable y
            Referencia: https://or.stackexchange.com/questions/33/
                        in-an-integer-program-how-i-can-force-a-
                        binary-variable-to-equal-1-if-some-cond
            '''
            return thr_mining - x <= (1 - y) * M
        
        
        def __soft_activation_const():
            '''Restricción de cantidad mínima de cada alimento
            en la receta.
            '''
            # Checkear condición de asignación de la lista
            assert len(min_x) == len(ingredients_matrix), "Problemas con la definición de 'min_x'"
            return x >= np.array(min_x)
        
        
        def __ingredients_limit():
            return cp.sum(y) >= min_ingredients
        
        
        def __carbProt_const():
            '''Restricción de carbohidratos (Art. 540 c.) y proteínas
            (Art. 540 e.) 
            '''
            # Obteniendo la restricción (carbohidrato, proteína)
            return CP.T @ x <= np.array([350, 50]) # gramos
        
        
        def __caffeine_const():
            '''Restricción de cafeína (Art. 540 j.)
            '''            
            # Obteniendo la restricción
            return K.T @ x <= 0.5     # g/dia (500 mg/dia)
        
        
        # Definición de la lista de restricciones
        constraint_list = list()
        
        ####  Agregando las restricciones de interés ####
        # Restricción base: Naturaleza de la variable
        constraint_list.append(__base_const())
        
        # Restricción 1: Límite de proteínas y carbohidratos
        if CP is not None:
            constraint_list.append(__carbProt_const())
        # Restricción 2: Límite de cafeina
        if K is not None:
            constraint_list.append(__caffeine_const())
        # Restricción 3: Bebida para deportistas
        if D is not None:
            constraint_list.append(__sport_const())
        
        # Restricción condicional: Condición de activación hard
        if isinstance(min_ingredients, int):
            constraint_list.append(__y_up_activation_const())
            constraint_list.append(__y_down_activation_const())
            constraint_list.append(__ingredients_limit())
        
        # Restricción condicional: Condición de activación soft
        if isinstance(min_x, list):
            constraint_list.append(__soft_activation_const())
        
        return constraint_list
        
    
    def _objective_function(T_obj, T_rec, x):
        '''Rutina que define la función objetivo del problema.
        '''
        if weighted:
            # Definición de los pesos
            W = T_obj / np.sum(T_obj) * alpha_weight
            
            return cp.Minimize((W @ (T_obj - T_rec @ x)) ** 2)
            
        else:
            return cp.Minimize(cp.sum_squares((T_obj - T_rec @ x)))
    
    
    def _model():
        # Variable de cantidad
        x = cp.Variable(len(ingredients_matrix))
        
        # Variable de activación de x
        y = None
        if isinstance(min_ingredients, int):
            y = cp.Variable(len(ingredients_matrix), boolean=True)
        
        # Restricciones
        constraints = _constraints(x, y)

        # Definición de la función objetivo
        objective_func = _objective_function(T_obj, T_ings, x)
        
        # Definición del problema de optimización
        prob = cp.Problem(objective=objective_func, constraints=constraints)
        
        # Calculando el óptimo
        result = prob.solve(solver=solver, verbose=verbose)
        
        # Obteniendo el valor de y en caso de que se haya usado
        if isinstance(min_ingredients, int):
            y = y.value
            
        return result, x.value, y
     
    
    def _cosine_similarity(a, b):
        return 1 - distance.cosine(a, b)
    
    
    def _euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))
    
    
    # Condición de sanidad para el desarrollo del modelo
    assert_txt = "Diferencia de dimensiones entre tabla objetivo e ingredientes."
    assert ingredients_matrix.shape[-1] == obj_food.shape[-1], assert_txt
    
    # Obteniendo las tablas
    if grames is not None:
        T_obj = np.squeeze(obj_food) * grames
        T_ings = ingredients_matrix.T * grames
    else:
        T_obj = np.squeeze(obj_food)
        T_ings = ingredients_matrix.T
    
    # Obtención de los valores del modelo
    result, x, y = _model()
    
    # Asegurarse del resultado del optimizador
    if x is not None:
        # Definición de la tabla nutricional de salida
        T_gen = T_ings @ x
        
        # Obteniendo la distancia euclidiana y coseno
        euclidean_dist = _euclidean_distance(T_obj, T_gen)
        cosine_sim     = _cosine_similarity(T_obj, T_gen)
    else:
        euclidean_dist = cosine_sim = T_gen = None
        print('Unfeasible problem.')
    
    # Retornar el diccionario de resultados relevantes
    return {'result': result, 'x': x, 'y': y,
            'euc_dist': euclidean_dist,
            'cos_sim': cosine_sim,
            'T_gen': T_gen}


# Módulo de testeo
if __name__ == '__main__':
    from food_management import get_ingredients
    filedir = '../FOOD_AN/CONSOLIDADO MVP v4 (sin En)_id_aux.dat'
    index_ings = [2,3,7,8,1]
    v, l = get_ingredients(filedir, index_ings, start_index=7, sep='|', decimal=',')
    v_obj, l_obj = get_ingredients(filedir, [40], start_index=7, sep='|', decimal=',')
    
    # Pasando a array
    n = -3
    v = v.to_numpy()[:, 14:17] / 100
    v_obj = v_obj.to_numpy()[:, 14:17] / 100
    
    # print(v)
    # print(v_obj)

    # Optimizador
    a = optimizer(v, v_obj, solver=cp.SCS, verbose=True, 
                  grames=None, weighted=False, alpha_weight=100,
                  min_ingredients=None, min_x=None, thr_mining=0, M=1e9)
    print(np.sum((v_obj - a['T_gen']) ** 2))
    print(a['x'])
    print(v_obj, v_obj.shape)
    print(v)
    print(a['T_gen'], a['T_gen'].shape)
