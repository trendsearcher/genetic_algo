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
from scipy.stats import norm
from math import sqrt
import seaborn as sns
from scipy.stats import shapiro


win_in_list = [ 1208, 1201, 1208, 1208, 1208]
win_out_list = [ 1208, 1208, 1208, 1208, 1208]
# do not delete [1208, 0.00029014748209449853, 0.0012819841679180361, 0.0006076796223813778, 1208]
#eps_1_list = [0.0003104906701795232, 0.0003104906701795232, 0.0002859545811549263, 0.0003104906701795232, 0.00029014748209449853]
#eps_2_list = [0.0012819841679180361, 0.0012819841679180361, 0.0012888284222635164, 0.0012819841679180361, 0.0012819841679180361]
#eps_3_list = [0.0006158595194085028, 0.0006158595194085028, 0.0006658229662212834, 0.0006158595194085028, 0.0006076796223813778]

# do not delete backward [1201, 0.002749249486490757, 0.0003702675182819587, 0.00020350907795851323, 1208]
eps_1_list = [0.003749249486490757, 0.0030547216516563966, 0.002781354622714396, 0.0024982155603140615, 0.00196171995122209873]
eps_2_list = [0.0003702675182819587, 0.0004072942701101546, 0.00018774054257016804, 9.092562284051645e-05, 9.092562284051645e-05]
eps_3_list = [0.00020350907795851323, 0.00020350907795851323, 0.00011653823182367766, 4.326507066628209e-05, 4.326507066628209e-05]

popSize = 5
mutation_coef = 0.1
elite_percent = 0.55
generations = 30

df_train = pd.read_csv('F:\\SBER_RTS_Si1219.csv', header= 0, error_bad_lines=False, nrows = 2000000)
df_train.dropna(inplace=True)
print(df_train.columns, df_train.shape)
#df_test = pd.read_csv('F:\\SBER_RTS_Si319.csv', header= 0, error_bad_lines=False, skiprows=range(1, 3000000))
#df_test.dropna(inplace=True)


def initialPopulation(popSize):
    '''возвращает набор особей на первой итерации 
       без мутаций'''
    population = [] 
    for i in range(0, popSize):
        individual = [win_in_list[i], eps_1_list[i], 
                       eps_2_list[i], eps_3_list[i], 
                       win_out_list[i]]
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
                if a + 10 <=5000:
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
            return (values[-1] - values[0])/values[0] > individual[1]
        def price_moove_up_over_boarder2(values):
            return (values[-1] - values[0])/values[0] < individual[2]
        def price_moove_up_over_boarder3(values):
            return (values[-1] - values[0])/values[0] >  -individual[3]
        
        def price_moove_down_over_boarder1(values):
            return (values[-1] - values[0])/values[0] < -individual[1]
        def price_moove_down_over_boarder2(values):
            return (values[-1] - values[0])/values[0] > -individual[2]
        def price_moove_down_over_boarder3(values):
            return (values[-1] - values[0])/values[0] < individual[3]
            
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
        previous_position_close_index = 0
        indexes = []
        difference = []
        for index, row in df_position_deal_result_up.iterrows():
            if index > previous_position_close_index:
                indexes += [index]
                difference += [-row['out'] + row['Price1']]
                previous_position_close_index = index + individual[4]
        df_up = pd.DataFrame({'index':indexes, 'difference':difference})
        
        df_position_deal_result_down = pd.DataFrame(df[df.index.isin(index_position_down_list)]['Price1'])
        df_position_deal_result_down['out'] = (df.Price1.shift(-individual[4]))
        previous_position_close_index = 0
        indexes = []
        difference = []
        for index, row in df_position_deal_result_down.iterrows():
            if index > previous_position_close_index:
                indexes += [index]
                difference += [-row['Price1'] + row['out']]
                previous_position_close_index = index + individual[4]
        df_down = pd.DataFrame({'index':indexes, 'difference':difference})

        df_concat_positions = pd.concat([df_up, df_down])
        df_concat_positions = df_concat_positions.sort_values(by=['index'])
        df_concat_positions.dropna(inplace=True)
        
        
        
        plt.hist(df_concat_positions['difference'].values.tolist(), color = 'blue', edgecolor = 'black', bins = 20)
        plt.show()
        plt.pause(0.5)
        stat, p = shapiro(df_concat_positions['difference'].values.tolist())
        print('Statistics=%.3f, p=%.3f' % (stat, p))

        
        
        cumsum_results = df_concat_positions['difference'].cumsum().values.tolist()
        print('sredn profit', df_concat_positions['difference'].mean())
        total_actions = len(cumsum_results)
        indexes_list = list(range(total_actions))

        try:
            lines = plt.plot(indexes_list, cumsum_results)
            l1 = lines
            plt.setp(lines, linestyle='-')
            plt.setp(l1, linewidth=1, color='b')
            plt.grid()
            plt.show() 
            print(individual)
        except:
            pass
        
        if total_actions == 0:
            population_grade += [0]        
        else:
            #           выигрыш                     биржа                            брокер             спред
            grade = int(cumsum_results[-1] -  2*int(20500*total_actions*0.000038) - 2*total_actions - 2*total_actions)
            population_grade += [grade]
            print(total_actions, grade)

    return(population_grade)
    
def breed (population, population_grade):
    '''отбирает элиту и позволяет ей размножаться со всей популяцией
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
    '''пародирует эволюцию, если за 1 шаг не выявлено superior особи, то
    деградация с лучшей особью на входе не происходит'''
    family = breed(population, grades)
    mutated_family = mutatePopulation(family)
    mutated_grades = grade(mutated_family, df)
    general_population = mutated_family + population
    general_grades = mutated_grades + grades
    new_population, new_grades = death(general_population, general_grades)
    return(new_population, new_grades)





population = initialPopulation(popSize)

grades = grade(population, df_train)

progress_train = []
progress_test = []

for i in range(generations):
    population, grades = nextGeneration(population, grades, df_train)
    progress_train += [max(grades)]
    alpha_index = grades.index(max(grades))
    alpha_individ = population[alpha_index]
    
    print(i, 'alpha_individ', alpha_individ)
    print('train', max(grades))
#    test_grade = grade([alpha_individ], df_test)
#    print('test', test_grade)
#    print()
#    progress_test += [max(test_grade)]
    
    
#lines = plt.plot(list(range(generations)), progress_train, list(range(generations)), progress_test)
#l1, l2= lines
#plt.setp(lines, linestyle='-')
#plt.setp(l1, linewidth=1, color='b')
#plt.setp(l2, linewidth=1, color='r')
#plt.title('train-blue, test-red' )
#plt.grid()
#plt.show()
#
#for i, j in zip(population, test_grade):
#    print('individ', i, 'grade', j)


















