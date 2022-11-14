import numpy as np
from sage.crypto.sbox import SBox
import random
import matplotlib.pyplot as plt


def gen_rand_sboxes(n, b):
    s_boxes = []
    for i in range(n):
        s_box = np.random.permutation(range(0, b)).tolist()
        s_boxes.append(s_box)
    return s_boxes


rand_s_boxes = gen_rand_sboxes(100, 256)
non_lin = 0
diffe = 0
for i in rand_s_boxes:
    i = SBox(i)
    non_lin += i.nonlinearity()
    diffe += i.differential_uniformity()
print(non_lin)
print(diffe)


####################### Genetic algorithm ##########################################


def get_init_population(n, b):
    s_boxes = []
    for i in range(n):
        s_box = np.random.permutation(range(0, b)).tolist()
        s_boxes.append(s_box)
    return s_boxes


def fit(s_box):
    non_lin = SBox(s_box).nonlinearity()
    diff = SBox(s_box).differential_uniformity()
    score1 = non_lin * 5
    score2 = (100 - diff) * 5
    score = score1 + score2
    return score, non_lin, diff


def cross1(parent1, parent2):
    max_random_index = len(parent1)-1
    child_1 = parent1[0:random.randint(0, max_random_index)]
    child_2 = parent2[0:random.randint(0, max_random_index)]
    for gene in parent2:
        if gene not in child_1:
            child_1.append(gene)
    for gene in parent1:
        if gene not in child_2:
            child_2.append(gene)
    return child_1, child_2


def mut(s_box):
    idx1 = random.randint(0, len(s_box)-1)
    idx2 = random.randint(0, len(s_box)-1)
    s_box[idx1], s_box[idx2] = s_box[idx2], s_box[idx1]
    return s_box


def sort_tuples(tup):
    tup.sort(key=lambda x: x[1], reverse=True)
    return tup


def sort_population(sorted_tuples, population):
    new_population = []
    for i in sorted_tuples:
        new_population.append(population[i[0]])
    return new_population


def get_avg_score(scores):
    score = 0
    for i in scores:
        score += i[1]
    return score / len(scores)


def get_avgs(arr):
    score = 0
    for i in arr:
        score += i
    return score / len(arr)


def genetic_algo(n):
    population = get_init_population(100, 256)
    prev_avg_nonlin = 0
    prev_avg_diff = 100
    for i in range(n):
        arr_fit = []
        arr_nonlin = []
        arr_diff = []
        for j in population:
            f = fit(j)
            arr_fit.append(f[0])
            arr_nonlin.append(f[1])
            arr_diff.append(f[2])
        scores = list(zip(list(range(0, 256)), arr_fit))
        print("epoch: ", i)
        sort_by_fitness_scores = sort_tuples(scores)
        print("sorted scores:", sort_by_fitness_scores)
        best = population[sort_by_fitness_scores[0][0]]
        avg_score = get_avg_score(sort_by_fitness_scores)
        avg_nonlin = get_avgs(arr_nonlin)
        avg_diff = get_avgs(arr_diff)
        if (prev_avg_nonlin > avg_nonlin) & (prev_avg_diff < avg_diff):
            break
        prev_avg_diff = avg_diff
        prev_avg_nonlin = avg_nonlin
        write_sboxes(population, i, avg_score, avg_nonlin, avg_diff, best)
        population = sort_population(sort_by_fitness_scores, population)
        for idx in range(20):
            population.remove(population[len(population)-1])
        b = population[0:20]
        crossed = []
        for p in range(10):
            child1, child2 = cross1(b[p], b[19-p])
            crossed.append(child1)
            crossed.append(child2)
        rand_idx_1 = random.randint(0, (len(crossed)/2)-1)
        rand_idx_2 = random.randint((len(crossed)/2), len(crossed)-1)
        crossed[rand_idx_1] = mut(crossed[rand_idx_1])
        crossed[rand_idx_2] = mut(crossed[rand_idx_2])
        population = population + crossed
    return population


def write_sboxes(s_boxes, i, avg_score, avg_nonlin, avg_diff, best):
    with open("\\s_boxes_epoch_n_" + str(i) + ".txt", "a+") as f:
        for s in s_boxes:
            f.write('nonlinearity = ' + str(float(SBox(s).nonlinearity())))
            f.write('  differential_uniformity = ' + str(float(SBox(s).differential_uniformity())))
            f.write('  [')
            for item in s:
                f.write(str(item) + ", ")
            f.write("]\n")
        f.write(("\navg score: " + str(float(avg_score)) + "\navg nonlin: " + str(float(avg_nonlin)) + "\navg diff: " + str(float(avg_diff))))
        f.write("\nbest = \n")
        f.write('[')
        for item in best:
            f.write(str(item)+", ")
        f.write("]\n")