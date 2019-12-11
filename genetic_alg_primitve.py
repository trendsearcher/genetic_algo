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
#from scipy.stats import norm
#from math import sqrt
#import seaborn as sns
#from scipy.stats import shapiro


win_in_list = [ 10, 10, 100, 100, 1000, 1000]
win_out_list = [ 1, 1, 2, 2, 2, 2]
# [300, 0.001981270921628502, 432]  7983****
# [500, 0.003541139948268511, 574]  6043****
# [1069, 0.004991015884379593, 1278] 4440**

#rts [300, 0.0023939306004137225, 257] 13228
#   [1000, 0.0037103100875716137, 930] 10687
# do not delete backward [1201, 0.002749249486490757, 0.0003702675182819587, 0.00020350907795851323, 1208]
eps_1_list = [0.007569030973777015, 0.016260162601626032, 0.02417615632379257, 0.04905068663845736, 0.07600575720690701, 0,12581417783356078]

popSize = 6
mutation_coef = 0.5
elite_percent = 0.55
generations = 20

df_train = pd.read_csv('C:\\Users\\user_PC\\Desktop\\pure_BR.csv', header= 0, error_bad_lines=False)
df_train.rename(columns={" <TIME>": "Time", " <VOLUME>": "Volume", "<PRICE>": "Price"}, inplace = True)

df_train.dropna(inplace=True)
print(df_train.columns, df_train.shape)


def initialPopulation(popSize):
    '''возвращает набор особей на первой итерации 
       без мутаций'''
    population = [] 
    for i in range(0, popSize):
        individual = [win_in_list[i], eps_1_list[i], win_out_list[i]]
        population += [individual] 
    return population

def mutatePopulation(population):
    '''вносит 1 мутацию в каждую особь и возвращает список обновленных особей'''
    mutated_population = [] 
    for individual in population:
        rand_multiply = random.uniform(0, mutation_coef)
        rand_space_index = random.randrange(1, 3)
        
        a = individual[rand_space_index]
        if rand_space_index == 1: # меняем eps
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
        
        def price_moove_down_over_boarder1(values):
            return (values[-1] - values[0])/values[0] < -individual[1]
            
        df['move1up'] = df['Price'].rolling(individual[0]).apply(price_moove_up_over_boarder1, raw=True)
    
        df['move1down'] = df['Price'].rolling(individual[0]).apply(price_moove_down_over_boarder1, raw=True)    
        
        df['right_move_up'] = df['move1up'] == 1
        df['right_move_down'] = df['move1down']  == 1
    
        index_position_up_list = df.index[df['right_move_up'] == True].tolist()
        index_position_down_list = df.index[df['right_move_down'] == True].tolist()
        
        df_position_deal_result_up = pd.DataFrame(df[df.index.isin(index_position_up_list)]['Price'])
        df_position_deal_result_up['out'] = (df.Price.shift(-individual[2]))
        previous_position_close_index = 0
        indexes = []
        difference = []
        for index, row in df_position_deal_result_up.iterrows():
            if index > previous_position_close_index:
                indexes += [index]
                difference += [-row['out'] + row['Price']]
                previous_position_close_index = index + individual[2]
        df_up = pd.DataFrame({'index':indexes, 'difference':difference})
        
        df_position_deal_result_down = pd.DataFrame(df[df.index.isin(index_position_down_list)]['Price'])
        df_position_deal_result_down['out'] = (df.Price.shift(-individual[2]))
        previous_position_close_index = 0
        indexes = []
        difference = []
        for index, row in df_position_deal_result_down.iterrows():
            if index > previous_position_close_index:
                indexes += [index]
                difference += [-row['Price'] + row['out']]
                previous_position_close_index = index + individual[2]
        df_down = pd.DataFrame({'index':indexes, 'difference':difference})

        df_concat_positions = pd.concat([df_up, df_down])
        df_concat_positions = df_concat_positions.sort_values(by=['index'])
        df_concat_positions.dropna(inplace=True)
        
        
        
#        plt.hist(df_concat_positions['difference'].values.tolist(), color = 'blue', edgecolor = 'black', bins = 20)
#        plt.show()
#        plt.pause(0.5)
#        stat, p = shapiro(df_concat_positions['difference'].values.tolist())
#        print('Statistics=%.3f, p=%.3f' % (stat, p))

        
        
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
            grade = int(cumsum_results[-1] -  2*int(60*total_actions*0.000038) - 0.02*total_actions - 0.01*total_actions)
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
    
    
lines = plt.plot(list(range(generations)), progress_train)#, list(range(generations)), progress_test)
l1= lines
plt.setp(lines, linestyle='-')
plt.setp(l1, linewidth=1, color='b')
#plt.setp(l2, linewidth=1, color='r')
plt.title('train-blue, test-red' )
plt.grid()
plt.show()

#for i, j in zip(population, test_grade):
#    print('individ', i, 'grade', j)


















