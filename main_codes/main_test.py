from food_management import get_ingredients, get_ingredients_comps
from optimizer import optimizer
import cvxpy as cp


# Módulo de testeo
if __name__ == '__main__':
    # Definición de las matrices de interés
    filedir = '../FOOD_AN/CONSOLIDADO MVP v4 (sin En)_id_aux.dat'
    
    # Definición de los ingredientes a elegir
    idx_ings = [1,2,3]
    # Definición del alimento objetivo
    idx_obj = [4]
    
    # Matriz de ingredientes 
    ing_matrix, labels = get_ingredients(filedir, idx_ings, start_index=7, sep='|', decimal=',')
    ing_matrix = ing_matrix.to_numpy()
    
    # Vector objetivo
    obj_food, label_obj = get_ingredients(filedir, idx_obj, start_index=7, sep='|', decimal=',')
    obj_food = obj_food.to_numpy()
    
    # Matrices anexas
    K, _ = get_ingredients_comps(filedir, idx_ings, 
                              specific_comps=['OT_caffeine'], 
                              sep='|', decimal=',')
    D, _ = get_ingredients_comps(filedir, idx_ings, 
                              specific_comps=['M_sodium', 'M_potassium'], 
                              sep='|', decimal=',')
    CP,_ = get_ingredients_comps(filedir, idx_ings, 
                               specific_comps=['G_carbohydrate', 'G_protein'], 
                               sep='|', decimal=',')
    
    # Aplicando el optimizador
    results = \
        optimizer(ing_matrix, obj_food, solver=cp.SCS, verbose=True, 
                  grames=None, weighted=False, alpha_weight=100,
                  min_ingredients=None, min_x=None, K=None, D=None, CP=None,
                  thr_mining=0, M=1e9)
        
    print(results)
