# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:57:57 2019

@author: user_PC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools  
import math

win_in_list = [50, 100, 200, 500, 1000]
win_out_list = win_in_list
eps_1_list = [0.00037239633952116573, 0.0005286578450299491, 0.0007030029241460432, 0.000941533060511261, 0.0011924601374296065]
eps_2_list = [0.0011828638518590808, 0.0012735305640746491, 0.0014325988487024583, 0.0018329870305135638, 0.0023282247065069756]
eps_3_list = [0.0002896597196905918, 0.00036582487902154614, 0.0004867803385263723, 0.000748114889110442, 0.0010496774186230877]

popSize = 5
#elite_proportion = 0.5
mutation_coef = 0.5
elite_percent = 0.5
generations = 30

df_train = pd.read_csv('E:\\history5month\\SBER_RTS_Si.csv', header= 0, error_bad_lines=False, nrows=1000000)
df_train.dropna(inplace=True)

df_test = pd.read_csv('E:\\history5month\\SBER_RTS_Si.csv', header= 0, error_bad_lines=False, skiprows=range(1, 1000000))
df_test.dropna(inplace=True)


def initialPopulation(popSize):
    '''возвращает набор особей на первой итерации 
       за 1 мутацию меняется лишь 1 параметр у особи'''
    population = [] 
    for i in range(0, popSize):
        index_to_begin =i   # random.randrange(len(win_in_list))
        
        rand_multiply = random.uniform(0, mutation_coef)
        rand_space_index = random.randrange(5)
        individual = [win_in_list[index_to_begin], eps_1_list[index_to_begin], 
                       eps_2_list[index_to_begin], eps_3_list[index_to_begin], 
                       win_out_list[index_to_begin]]
        a = individual[rand_space_index]
        if rand_space_index in [1,2,3]: # меняем eps
            if random.choice([True, False]):
                individual[rand_space_index] = a + rand_multiply*a
            else:    
                individual[rand_space_index] = a - rand_multiply*a
        else : # меняем входные и выходящие окна
            if random.choice([True, False]):
                if a + 10 <=1000:
                    individual[rand_space_index] = a + int(rand_multiply*a)
            else:
                if a - 10 >= 50:
                    individual[rand_space_index] = a - int(rand_multiply*a)
        population += [individual] 
    return population

def mutatePopulation(population):
    '''вносит 1 мутацию в каждую особь и возвращает список обновленных особей'''
    mutated_population = [] 
    for individual in population:
        rand_multiply = random.uniform(0, mutation_coef)
        rand_space_index = random.randrange(5)
        
        a = individual[rand_space_index]
        if rand_space_index in [1,2,3]: # меняем eps
            if random.choice([True, False]):
                individual[rand_space_index] = a + rand_multiply*a
            else:    
                individual[rand_space_index] = a - rand_multiply*a
        else : # меняем входные и выходящие окна
            if random.choice([True, False]):
                if a + 10 <=1000:
                    individual[rand_space_index] = a + int(rand_multiply*a)
            else:
                if a - 10 >= 50:
                    individual[rand_space_index] = a - int(rand_multiply*a)
        mutated_population += [individual] 
    return mutated_population

def grade(population, df):
    '''оценивает индивидов по результатам бектеста, возвращает список 
    оценок'''
    population_grade = []
    for  individual in population:
        individual[0] = int(individual[0])
        individual[-1] = int(individual[-1])
        def price_moove_up_over_boarder1(values):
            return (values[-1] - values[0])/values[0] < individual[1]
        def price_moove_up_over_boarder2(values):
            return (values[-1] - values[0])/values[0] > individual[2]
        def price_moove_up_over_boarder3(values):
            return (values[-1] - values[0])/values[0] < -individual[3]
        
        def price_moove_down_over_boarder1(values):
            return (values[-1] - values[0])/values[0] > -individual[1]
        def price_moove_down_over_boarder2(values):
            return (values[-1] - values[0])/values[0] < -individual[2]
        def price_moove_down_over_boarder3(values):
            return (values[-1] - values[0])/values[0] > individual[3]
            
        df['move1up'] = df['Price1'].rolling(individual[0]).apply(price_moove_up_over_boarder1, raw=True)
        df['move2up'] = df['Price2'].rolling(individual[0]).apply(price_moove_up_over_boarder2, raw=True)   
        df['move3up'] = df['Price3'].rolling(individual[0]).apply(price_moove_up_over_boarder3, raw=True)    
    
        df['move1down'] = df['Price1'].rolling(individual[0]).apply(price_moove_down_over_boarder1, raw=True)    
        df['move2down'] = df['Price2'].rolling(individual[0]).apply(price_moove_down_over_boarder2, raw=True)   
        df['move3down'] = df['Price3'].rolling(individual[0]).apply(price_moove_down_over_boarder3, raw=True)    
        
        df['right_move_up'] = df['move1up'] + df['move2up'] + df['move3up'] == 3
        df['right_move_down'] = df['move1down'] + df['move2down'] + df['move3down'] == 3
    
        index_position_up_list = df.index[df['right_move_up'] == True].tolist()
        index_position_down_list = df.index[df['right_move_down'] == True].tolist()
    
        df_position_deal_result_up = pd.DataFrame(df[df.index.isin(index_position_up_list)]['Price1'])
        df_position_deal_result_up['out'] = (df.Price1.shift(-individual[4]))
        df_position_deal_result_up.dropna(inplace=True)
        df_position_deal_result_up['result'] = df_position_deal_result_up['out'] - df_position_deal_result_up['Price1']
        
        df_position_deal_result_down = pd.DataFrame(df[df.index.isin(index_position_down_list)]['Price1'])
        df_position_deal_result_down['out'] = (df.Price1.shift(-individual[4]))
        df_position_deal_result_down.dropna(inplace=True)
        df_position_deal_result_down['result'] = df_position_deal_result_down['Price1'] - df_position_deal_result_down['out']
        
        
        
        cumsum_up = df_position_deal_result_up['result'].cumsum() 
#        cumsum_up.plot(x='num_children',y='num_pets',color='red')
#        plt.show()
        cumsum_up = cumsum_up.values.tolist()
        cumsum_down = df_position_deal_result_down['result'].cumsum().values.tolist() 
        
#        grade = (df_position_deal_result_up['result'].sum() + df_position_deal_result_down['result'].sum())/(df_position_deal_result_up.shape[0] + df_position_deal_result_down.shape[0] + 1)        

        grade_up = np.mean(cumsum_up)/len(cumsum_up)
        grade_down = np.mean(cumsum_down)/len(cumsum_down)
        
        print(grade_up, grade_down, len(cumsum_up), len(cumsum_down))
        
        if math.isnan(grade_down) or math.isnan(grade_up):
            population_grade += [0]        
        else:
            population_grade += [abs(grade_up + grade_down)/2]
    return(population_grade)
    
def breed (population, population_grade):
    '''отсеивает элиту и позволяет ей размножаться со всей популяцией
       и возвращает список детей+элиты'''
    elite = [x  for (x, y) in zip(population, population_grade) if y >= np.quantile(population_grade, elite_percent)]
    loosers = [x  for(x, y) in zip(population, population_grade) if y < np.quantile(population_grade, elite_percent)]
    
    if len (elite) > int((1-elite_percent)*len(population)+0.5): # overcrowded elite
        while len (elite) > int((1-elite_percent)*len(population)+0.5):
            index_of_element_to_downgrade = random.randrange(len(elite))
            loosers.append(elite[index_of_element_to_downgrade])
            elite.pop(index_of_element_to_downgrade)
            
    super_childs = list(itertools.combinations(elite, 2))
    super_childs = [list((np.array(i[0]) + np.array(i[1]))/2) for i in super_childs]
    ordinary_childs = list(itertools.product(elite, loosers))
    ordinary_childs = [list((np.array(i[0]) + np.array(i[1]))/2) for i in ordinary_childs]
    
    all_individuals = ordinary_childs + super_childs + elite
    print('kids', len(ordinary_childs), len(super_childs), len(elite))
    return(all_individuals)

def death (population, population_grade):
    '''оставляет лучшие индивиды в количестве popSize и их оценки'''
    arr = np.array(population_grade)
    stayed_alive_indexes = list(arr.argsort()[-popSize:][::-1])
    stayed_alive = [population[x] for x in stayed_alive_indexes]
    grades_alive = [population_grade[x] for x in stayed_alive_indexes]
    return(stayed_alive, grades_alive)

def nextGeneration (population, grades, df):
    '''пародирует эволюцию'''
    family = breed(population, grades)
    mutated_family = mutatePopulation(family)
    grades = grade(mutated_family, df)
    new_population, grades = death(mutated_family, grades)
    return(new_population, grades)





population = initialPopulation(popSize)
grades = grade(population, df_train)

progress_train = []
progress_test = []

for i in range(generations):
    population, grades = nextGeneration(population, grades, df_train)
    progress_train += [max(grades)]
    alpha_index = grades.index(max(grades))
    alpha_individ = population[alpha_index]
    
    print('alpha_individ', alpha_individ)
    print('train', max(grades))
    test_grade = grade([alpha_individ], df_test)
    print('test', test_grade)
    print()
    progress_test += [max(test_grade)]
    
    
lines = plt.plot(list(range(generations)), progress_train, list(range(generations)), progress_test)
l1, l2= lines
plt.setp(lines, linestyle='-')
plt.setp(l1, linewidth=1, color='b')
plt.setp(l2, linewidth=1, color='r')
plt.title('train-blue, test-red' )
plt.grid()
plt.show()

for i, j in zip(population, test_grade):
    print('individ', i, 'grade', j)


















