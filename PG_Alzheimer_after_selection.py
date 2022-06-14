#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

import math
import operator
import random
import itertools
import numpy
import os

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import csv

import matplotlib.pyplot as plt
import networkx as nx

# Generator Seed
GENERATOR_SEED = 900
random.seed(GENERATOR_SEED)

# K-Fold cross validation and seed
FOLD_NUMBER = 6
SEED_NUMBER = 2 #100
POPULATION_SIZE = 2400#2400

# Crossover and muta
# Mutatiotion probability
CX_PB = 0.5
MUT_PB = 0.2

# Generations
N_GEN = 40

# Min and max depth of the tree
MIN_DEPTH = 1
MAX_DEPTH = 2

# Min and max
# depth of the tree during mutation
MIN_DEPTH_MUT = 0
MAX_DEPTH_MUT = 1

# Bloat control
MAX_DEPTH_BLOAT = 17

# Tournament
TOURNAMENT_SIZE = 3

with open("Alzheimer_reduced.csv") as dataset:
    dataset = csv.reader(dataset)
    relevant_indices = {71,13,1,5,27,4,40,92,6,28,-1}
    samples = list(list(float(row[i]) for i in relevant_indices) for row in dataset)
    random.shuffle(samples)

# CUSTOM OPERATIONS
# =============================
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2

def average(input1, input2):
    return (input1 + input2) / 2

def quadratic_similarity(input1, input2):
    return numpy.sqrt(input1**2 + input2**2)
# =============================


# CUSTOM SETS
# =============================
def if_then_else_pset():
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 107), bool)

    # Math operators
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(protectedDiv, [float, float], float)

    # Confront operators
    pset.addPrimitive(operator.lt, [float, float], bool)
    pset.addPrimitive(operator.gt, [float, float], bool)
    pset.addPrimitive(operator.eq, [float, float], bool)
    pset.addPrimitive(if_then_else, [bool, float, float], float)

    # Terminals
    pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
    pset.addTerminal(False, bool)
    pset.addTerminal(True, bool)

    return pset

def if_then_else_math_pset():
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 107), bool)

    # Math operators
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(protectedDiv, [float, float], float)

    # Accumulation operators
    pset.addPrimitive(max, [float, float], float)
    pset.addPrimitive(min, [float, float], float)
    pset.addPrimitive(average, [float, float], float)

    # Advanced math operators
    pset.addPrimitive(quadratic_similarity, [float, float], float)
    pset.addPrimitive(operator.abs, [float], float)

    # Confront operators
    pset.addPrimitive(operator.lt, [float, float], bool)
    pset.addPrimitive(operator.eq, [float, float], bool)
    pset.addPrimitive(if_then_else, [bool, float, float], float)

    pset.addTerminal(False, bool)
    pset.addTerminal(True, bool)

    return pset

def if_then_else_math_reduced_pset():
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 107), bool)

    # Accumulation operators
    pset.addPrimitive(max, [float, float], float)
    pset.addPrimitive(min, [float, float], float)

    # Advanced math operators
    pset.addPrimitive(quadratic_similarity, [float, float], float)

    # Confront operators
    pset.addPrimitive(operator.lt, [float, float], bool)
    pset.addPrimitive(operator.eq, [float, float], bool)
    pset.addPrimitive(if_then_else, [bool, float, float], float)

    pset.addTerminal(False, bool)
    pset.addTerminal(True, bool)

    return pset

def if_then_else_reduced_pset():
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, len(samples[0]) - 1), bool)

    # Boolean operators
    pset.addPrimitive(operator.and_, [bool, bool], bool)
    pset.addPrimitive(operator.or_, [bool, bool], bool)
    pset.addPrimitive(operator.not_, [bool], bool)

    # Confront operators
    pset.addPrimitive(operator.lt, [float, float], bool)
    pset.addPrimitive(operator.eq, [float, float], bool)
    pset.addPrimitive(if_then_else, [bool, float, float], float)

    # Terminals
    pset.addTerminal(False, bool)
    pset.addTerminal(True, bool)

    return pset

def if_then_else_pset_after_selection():
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, len(samples[0]) - 1), bool)

    # Math operators
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)

    # Confront operators
    pset.addPrimitive(operator.gt, [float, float], bool)
    pset.addPrimitive(if_then_else, [bool, float, float], float)

    # Terminals
    pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
    pset.addTerminal(False, bool)
    pset.addTerminal(True, bool)

    return pset

def if_then_else_pset_after_selection_constants():
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, len(samples[0]) - 1), bool)

    # Math operators
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)

    # Confront operators
    pset.addPrimitive(operator.gt, [float, float], bool)
    pset.addPrimitive(if_then_else, [bool, float, float], float)

    # Terminals
    pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
    pset.addEphemeralConstant("rand100_2", lambda: random.random() * 100, float)
    pset.addEphemeralConstant("rand100_3", lambda: random.random() * 100, float)
    pset.addEphemeralConstant("rand100_4", lambda: random.random() * 100, float)
    pset.addEphemeralConstant("rand100_5", lambda: random.random() * 100, float)
    pset.addEphemeralConstant("rand100_6", lambda: random.random() * 100, float)
    pset.addEphemeralConstant("rand100_7", lambda: random.random() * 100, float)
    pset.addEphemeralConstant("rand100_8", lambda: random.random() * 100, float)
    pset.addEphemeralConstant("rand100_9", lambda: random.random() * 100, float)
    pset.addTerminal(False, bool)
    pset.addTerminal(True, bool)

    return pset
# =============================

def creator_init():
    """
    Initialize the creator
    """
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

def toolbox_init(pset):
    """
    Initialize the toolbox
    :param pset: the set of operations
    :return: the toolbox
    """
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=MIN_DEPTH, max_=MAX_DEPTH)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    return toolbox

def toolbox_init_tournament(toolbox):
    """
    :param toolbox: the toolbox to update
    """
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=MIN_DEPTH_MUT, max_=MAX_DEPTH_MUT)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH_BLOAT))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH_BLOAT))


# Initialization
# Program Set
P_SET = if_then_else_pset_after_selection
pset = P_SET()
creator_init()
toolbox = toolbox_init(pset)

# Evaluation function
def evalAlzheimer(individual, set):
    """
    Evalute the fitness of and individual of the population
    :param individual: the individual to evaluate
    :param set: the set to verify the the performances on
    :return: the resulted fitness
    """
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    # Evaluate the sum of correctly identified subject
    n_feature = len(samples[0]) - 1
    result = sum(bool(func(*subject[:n_feature])) is bool(subject[-1]) for subject in set) / len(set)

    return result,

def get_sens_spec(individual, set):
    """
    Evaluate sensitivity and specificity of an individual of the populatioj
    :param individual: the individual to evaluate
    :param set: the set to verify the performances on
    :return: the resulted sensitivity and specificity
    """
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    n_feature = len(samples[0]) - 1
    FP, FN, TP, TN = 0, 0, 0, 0

    for subject in set:
        pred = bool(func(*subject[:n_feature]))
        truth = bool(subject[-1])

        if truth is True:
            if pred is True:
                TP += 1
            else:
                FN += 1
        if truth is False:
            if pred is False:
                TN += 1
            if pred is True:
                FP += 1

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return sensitivity, specificity


toolbox_init_tournament(toolbox)

#toolbox.register("evaluate", evalAlzheimer)

def GP_run_tournament():
    """
    Runs a torunament to find the performances of the GP
    :return: a list of results for each fold and each seed
    """
    # Initialize k seeds
    seed_list = random.sample(range(1000), k=SEED_NUMBER)
    print(seed_list)

    # Create a list with the results for each fold
    fold_results = []
    fold_size = len(samples) // FOLD_NUMBER

    # Tracking the loop
    TOT_ITER = FOLD_NUMBER * SEED_NUMBER
    current_iter = 0

    # For each fold
    for k in range(FOLD_NUMBER):
        # Initialize the start and end index of the test set in the fold
        start = k * fold_size
        end = (k + 1) * fold_size if k < FOLD_NUMBER - 1 else len(samples) # Reaches the end of the list for the last k

        # Get test and training sets
        test = samples[start:end]
        training = samples[:start] + samples[end:] # All the dataset excluding the start:end interval

        # Initialize a list for the results enumerated by seed
        result_list = []

        # For each seed it runs the GP algorithm
        for seed in seed_list:
            # Print current state
            current_iter += 1
            print(f"\n============\n     Current iteration: {current_iter} / {TOT_ITER}")

            # Set the seed
            random.seed(seed)

            # Initialize population and hall of fame
            pop = toolbox.population(n=POPULATION_SIZE)
            hof = tools.HallOfFame(1)

            # Initialize mstats class
            stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
            stats_size = tools.Statistics(len)
            mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
            mstats.register("avg", numpy.mean)
            mstats.register("std", numpy.std)
            mstats.register("min", numpy.min)
            mstats.register("max", numpy.max)

            # Run algorithm
            toolbox.register("evaluate", evalAlzheimer, set=training)
            algorithms.eaSimple(pop, toolbox, CX_PB, MUT_PB, N_GEN, stats=mstats, halloffame=hof, verbose=True)
            toolbox.unregister("evaluate")


            # Get the best tree and fitness
            #print("hof phenotype: ", hof.items[0])
            tree = hof.items[0]

            # print("hof fitness: ", hof.items[0].fitness)
            fitness = hof.items[0].fitness

            #Run on Test Set
            test_fitness = evalAlzheimer(tree, test)

            train_sens, train_spec = get_sens_spec(tree, training)
            test_sens, test_spec = get_sens_spec(tree, test)

            # Add it to the result list for this seed
            result_list.append([seed, tree, fitness, test_fitness, train_sens, test_sens, train_spec, test_spec])

        # Add the result to the fold list
        fold_results.append(result_list)

    return fold_results

def print_tree(expr):

    #print tree with best fitness
    nodes, edges, labels = gp.graph(expr)

    G = nx.DiGraph(directed = True)
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    pos = nx.nx_pydot.graphviz_layout(G, prog="dot",root = nodes[0])
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels)

    plt.show()

def save_results(fold_results, train_acc, test_acc,
                 avg_max_train_acc, avg_max_test_acc, avg_min_train_acc, avg_min_test_acc, avg_std_train_acc, avg_std_test_acc,
                 avg_train_sens, avg_test_sens, avg_train_spec, avg_test_spec):
    """
    Save the results on a file
    :param fold_results: the fold results to save
    :param train_acc: avg train accuracy
    :param test_acc: avg test accuracy
    """
    i = 0

    # Chooses filename to save the result
    filename = f'output/results_{i}.txt'
    while os.path.exists(filename):
        i += 1
        filename = f'output/results_{i}.txt'

    # Writes the content of the file
    with open(filename, 'w') as f:
        content = str()
        content += f"-- HYPERPARAMETERS -- \n"
        content += f"GENERATOR SEED: {GENERATOR_SEED}\n"
        content += f"FOLD NUMBER: {FOLD_NUMBER}\n"
        content += f"SEED_NUMBER: {SEED_NUMBER}\n"
        content += f"PSET: {P_SET.__name__}\n"
        content += f"POPULATION_SIZE: {POPULATION_SIZE}\n"
        content += f"CROSSOVER PROBABILITY: {CX_PB}\n"
        content += f"MUTATION PROBABILITY: {MUT_PB}\n"
        content += f"NUMBER OF GENERATIONS: {N_GEN}\n"
        content += f"MIN TREE DEPTH: {MIN_DEPTH}\n"
        content += f"MAX TREE DEPTH: {MAX_DEPTH}\n"
        content += f"MIN TREE DEPTH FOR MUTATION: {MIN_DEPTH_MUT}\n"
        content += f"MAX TREE DEPTH FOR MUTATION: {MAX_DEPTH_MUT}\n"
        content += f"BLOAT CONTROL MAX DEPTH: {MAX_DEPTH_BLOAT}\n"
        content += f"TOURNAMENT SIZE: {TOURNAMENT_SIZE}\n"
        content += "\n"
        content += "\n-- PERFORMANCES --\n"
        content += f"AVG TRAIN ACCURACY: {train_acc}\n"
        content += f"AVG TEST ACCURACY: {test_acc}\n"
        content += f"AVG MAX TRAIN ACCURACY: {avg_max_train_acc}\n"
        content += f"AVG MAX TEST ACCURACY: {avg_max_test_acc}\n"
        content += f"AVG MIN TRAIN ACCURACY: {avg_min_train_acc}\n"
        content += f"AVG MIN TEST ACCURACY: {avg_min_test_acc}\n"
        content += f"AVG STD TRAIN ACCURACY: {avg_std_train_acc}\n"
        content += f"AVG STD TEST ACCURACY: {avg_std_test_acc}\n"
        content += f"AVG TRAIN SENSITIVITY: {avg_train_sens}\n"
        content += f"AVG TEST SENSITIVITY: {avg_test_sens}\n"
        content += f"AVG TRAIN SPECIFICITY: {avg_train_spec}\n"
        content += f"AVG TEST SPECIFICITY: {avg_test_spec}\n"
        content += "\n"
        content += "\n-- ADVANCED STATS --\n"

        # Stats for each fold
        trees = []
        for result_list in fold_results:
            for result in result_list:
                trees.append(result[1])
        #tree_analysis(trees)

        for i in range(len(fold_results)):
            result_list = fold_results[i]
            content += f"FOLD N° {i}\n"
            for result in result_list:
                seed, tree, train, test, train_sens, test_sens, train_spec, test_spec = result
                content += f"    SEED: {seed} -> TRAIN ACC: {train}; TEST ACC: {test}; " \
                           f"TRAIN SENS ACC: {train_sens}; TEST SENS ACC: {test_sens}; " \
                           f"TRAIN SPEC ACC: {train_spec}; TEST SPEC ACC: {test_spec}; " \
                           f"TREE: {str(tree)}\n"
            print("")

        f.write(content)
        f.close()

# TRAIN MODELS
if __name__ == "__main__":
    # For running graphviz if it doesn't work on Windows
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

    # Run the tournament
    fold_results = GP_run_tournament()
    mean_list_train = []
    mean_list_test = []

    max_list_train = []
    max_list_test = []

    min_list_train = []
    min_list_test = []

    std_list_train = []
    std_list_test = []

    mean_list_sens_train = []
    mean_list_sens_test = []

    mean_list_spec_train = []
    mean_list_spec_test = []

    # For each result list in the fold results
    for result_list in fold_results:
        # Get the mean on the fitness for each seed
        mean_list_train.append(numpy.mean([elem[2].values[0] for elem in result_list]))
        # Get the mean on the fitness for each seed in the test set
        mean_list_test.append(numpy.mean([elem[3] for elem in result_list]))

        # Get the max on the fintess for each seed
        max_list_train.append(numpy.max([elem[2].values[0] for elem in result_list]))
        # Get the max on the fintess for each seed in the test set
        max_list_test.append(numpy.max([elem[3] for elem in result_list]))

        # Get the min on the fintess for each seed
        min_list_train.append(numpy.min([elem[2].values[0] for elem in result_list]))
        # Get the min on the fintess for each seed in the test set
        min_list_test.append(numpy.min([elem[3] for elem in result_list]))

        # Get the sd on the fintess for each seed
        std_list_train.append(numpy.std([elem[2].values[0] for elem in result_list]))
        # Get the sd on the fintess for each seed in the test set
        std_list_test.append(numpy.std([elem[3] for elem in result_list]))

        # Get the mean on the sensitivity for each seed
        mean_list_sens_train.append(numpy.mean([elem[4] for elem in result_list]))
        # Get the mean on the sensitivity for each seed in the test set
        mean_list_sens_test.append(numpy.mean([elem[5] for elem in result_list]))

        # Get the mean on the specificity for each seed
        mean_list_spec_train.append(numpy.mean([elem[6] for elem in result_list]))
        # Get the mean on the specificity for each seed in the test set
        mean_list_spec_test.append(numpy.mean([elem[7] for elem in result_list]))


    print("FITNESS MEANS for each k fold:", mean_list_train)
    print("TEST RESULTS:", mean_list_test)
    print("FITNESS MAX for each k fold:", max_list_train)
    print("TEST RESULTS:", max_list_test)
    print("FITNESS MIN for each k fold:", min_list_train)
    print("TEST RESULTS:", min_list_test)
    print("FITNESS STD for each k fold:", std_list_train)
    print("TEST RESULTS:", std_list_test)
    print("FITNESS SENSITIVITY for each k fold:", mean_list_sens_train)
    print("TEST RESULTS:", mean_list_sens_test)
    print("FITNESS SPECIFICITY for each k fold:", mean_list_spec_train)
    print("TEST RESULTS:", mean_list_spec_test)

    # Final means
    avg_train_acc = numpy.mean(mean_list_train)
    avg_test_acc = numpy.mean(mean_list_test)

    avg_max_train_acc = numpy.mean(max_list_train)
    avg_max_test_acc = numpy.mean(max_list_test)

    avg_min_train_acc = numpy.mean(min_list_train)
    avg_min_test_acc = numpy.mean(min_list_test)

    avg_std_train_acc = numpy.mean(std_list_train)
    avg_std_test_acc = numpy.mean(std_list_test)

    avg_train_sens = numpy.mean(mean_list_sens_train)
    avg_test_sens = numpy.mean(mean_list_sens_test)

    avg_train_spec = numpy.mean(mean_list_spec_train)
    avg_test_spec = numpy.mean(mean_list_spec_test)


    print("Final training results:", avg_train_acc)
    print("Final test results:", avg_test_acc)
    print("Avg Max train acc:", avg_max_train_acc)
    print("Avg Max test acc:", avg_max_test_acc)
    print("Avg Min train acc:", avg_min_train_acc)
    print("Avg Min test acc:", avg_min_test_acc)
    print("Avg Std train acc:", avg_std_train_acc)
    print("Avg Std test acc:", avg_std_test_acc)
    print("Avg train sens:", avg_train_sens)
    print("Avg test sens:", avg_test_sens)
    print("Avg train spec:", avg_train_spec)
    print("Avg test spec:", avg_test_spec)

    # Save the result on an output file
    save_results(fold_results, avg_train_acc, avg_test_acc, avg_max_train_acc,
                 avg_max_test_acc, avg_min_train_acc, avg_min_test_acc, avg_std_train_acc, avg_std_test_acc,
                 avg_train_sens, avg_test_sens, avg_train_spec, avg_test_spec)

    #Print the first tree of the first fold
    print_tree(fold_results[0][0][1])




