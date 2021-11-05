import numpy as np
import cvxpy as cp
from scipy.spatial import distance


def optimizer(ingredients, obj_table_pandas, food_tables_dict,
              solver=cp.ECOS, verbose=False, grames=250, 
              sport_drink=False, weighted=False, alpha_weight=100,
              min_ingredients=None, thr_mining=0, M=1e6):
    '''Función que permite generar las recetas de un producto, retornando
    unidades adimensionales de cada elemento que lo conforma.
    
    Parameters
    ----------
    ingredients : list or ndarray
        Lista de ingredientes a usar.
    obj_table_pandas : pandas.Dataframe
        Tabla nutricional del producto objetivo.
    food_tables_dict : dict
        Diccionario con las tablas nutricionales de los ingredientes que se
        desea utilizar.
    solver : str, optional
        Tipo de solver a utilizar para el problema de optimización. Por 
        defecto es 'ECOS'.
    verbose : bool, optional
        Booleano que define los prints para cada problema de optimización.
        Por defecto es False.
    grames : float, optional
        Dimensión del producto (en gramos) que se espera a la salida. Por
        defecto es 250g.
    sport_drink : bool, optional
        Booleano que indica si es que la bebida a hacer corresponde a
        bebida para deportistas.
    min_ingredients : None or int, optional
        Cantidad mínima de ingredientes a usar en la preparación. Por 
        defecto es None.
    thr_mining: int, optional
        En caso de definir un valor para la cantidad mínima de ingredientes
        con valores distintos de cero con "min_ingredients", este parámetro
        define el valor mínimo a partir del cual se considera que la variable
        binaria y se activa. Por defecto es 0.
        
    Returns
    -------
    info_dict : dict
        Diccionario con información útil para la elaboración del producto.
    '''
    def _objective_table():
        # Definición de la tabla nutricional objetivo
        return np.array(obj_table_pandas.loc[:, 'proportion'], dtype=float)
    
    
    def _recipes_table():
        # Creación del problema de optimización para la definición de la receta
        recipe_matrix = list()

        for ing_i in ingredients:
            # Obteniendo el valor de las recetas
            ing_values_i = np.array(food_tables_dict[ing_i].loc[:, 'proportion'], dtype=float)
            
            # Agregando a la matriz de recetas
            recipe_matrix.append(ing_values_i)
    
        return np.array(recipe_matrix).T
    
    
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
            # Definición del vector de carbohidratos y proteínas
            na_k_matrix = list()

            for ing_i in ingredients:
                food_t_i = food_tables_dict[ing_i]
                
                # Obteniendo solo el sodio y potasio
                na_k = food_t_i.loc[food_t_i['public_id'].isin(['FDB003523', 'FDB003521'])]
                
                # Obteniendo solo los valores
                na_k = na_k.loc[:, 'proportion']

                # Agregando a la matriz de recetas
                na_k_matrix.append(na_k)
            
            # Obteniendo la matriz de ponderaciones
            N = np.array(na_k_matrix).T
            
            # Obteniendo la restricción (proteína, carbohidrato)
            return N @ x <= np.array([1.610, 3.715]) # gramos (1610, 3715) mg
        
        
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
        
        
        def __ingredients_limit():
            return cp.sum(y) >= min_ingredients
        
        
        def __const_1():
            '''Restricción de carbohidratos (Art. 540 c.) y proteínas
            (Art. 540 e.) 
            '''
            # Definición del vector de carbohidratos y proteínas
            carb_prot_matrix = list()

            for ing_i in ingredients:
                food_t_i = food_tables_dict[ing_i]
                
                # Obteniendo solo las proteinas y carbohidratos
                c = food_t_i.loc[food_t_i['public_id'].isin(['FDBN00002', 'FDBN00003'])]
                
                # Obteniendo solo los valores
                c = c.loc[:, 'proportion']

                # Agregando a la matriz de recetas
                carb_prot_matrix.append(c)
            
            # Obteniendo la matriz de ponderaciones
            C = np.array(carb_prot_matrix).T
            
            # Obteniendo la restricción (proteína, carbohidrato)
            return C @ x <= np.array([50, 350]) # gramos
        
        
        def __const_2():
            '''Restricción de cafeína (Art. 540 j.)
            '''
            # Definición del vector de cafeína para cada ingrediente
            caffeine_vect = list()

            for ing_i in ingredients:
                # Armando la tabla
                food_t_i = food_tables_dict[ing_i]
                
                # Obteniendo solo la cafeína
                c = food_t_i.loc[food_t_i['public_id'].isin(['FDB002100', 'FDBN00003'])]
                # Obteniendo solo los valores
                c = c.loc[:, 'proportion'].tolist()[0]
                
                # Agregando a la matriz de recetas
                caffeine_vect.append(c)
            
            # Obteniendo la matriz de ponderaciones
            K = np.array(caffeine_vect).T
            
            # Obteniendo la restricción
            return K @ x <= 0.5     # g/dia (500 mg/dia)
        
        
        # Definición de la lista de restricciones
        constraint_list = list()
        
        ####  Agregando las restricciones de interés ####
        # Restricción base: Naturaleza de la variable
        constraint_list.append(__base_const())
        
        # Restricción 1: Límite de proteínas y carbohidratos
        constraint_list.append(__const_1())
        # Restricción 2: Límite de cafeina
        constraint_list.append(__const_2())
        
        # Restricción condicional: Bebida para deportistas
        if sport_drink:
            constraint_list.append(__sport_const())
            
        if isinstance(min_ingredients, int):
            constraint_list.append(__y_up_activation_const())
            constraint_list.append(__y_down_activation_const())
            constraint_list.append(__ingredients_limit())
        
        return constraint_list
        
    
    def _objective_function(T_obj, T_rec, x):
        '''Rutina que define la función objetivo del problema.
        '''
        if weighted:
            # Definición de los pesos
            W = T_obj.value / np.sum(T_obj.value) * alpha_weight
            
            return cp.Minimize(W @ (T_obj - T_rec @ x) ** 2)
            
        else:
            return cp.Minimize(cp.sum_squares((T_obj - T_rec @ x)))
    
    
    def _model():
        # Variable de cantidad
        x = cp.Variable(len(ingredients))
        
        # Variable de activación de x
        y = None
        if isinstance(min_ingredients, int):
            y = cp.Variable(len(ingredients), boolean=True)
            
        # Parámetros
        T_obj = cp.Parameter(shape=obj_table.shape)
        T_obj.value = obj_table
        T_rec = cp.Parameter(shape=recipe_matrix.shape)
        T_rec.value = recipe_matrix

        # Restricciones
        constraints = _constraints(x, y)

        # Definición de la función objetivo
        objective_func = _objective_function(T_obj, T_rec, x)
        
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
    
    
    # Obteniendo las tablas
    obj_table = _objective_table() * grames
    recipe_matrix = _recipes_table() * grames
    
    # Obtención de los valores del modelo
    result, x, y = _model()
    
    # Obteniendo la distancia euclidiana y coseno
    euclidean_dist = _euclidean_distance(obj_table, recipe_matrix @ x)
    cosine_sim     = _cosine_similarity(obj_table, recipe_matrix @ x)
    
    # Retornar el diccionario de resultados relevantes
    return {'result': result, 'x': x, 'y': y,
            'euc_dist': euclidean_dist,
            'cos_sim': cosine_sim,
            'T_ing': recipe_matrix}


# Módulo de testeo
if __name__ == '__main__':
    pass

