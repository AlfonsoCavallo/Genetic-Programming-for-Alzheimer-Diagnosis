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

import re


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

def if_then_else_pset_after_selection():
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 10), bool)

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

pset = if_then_else_pset_after_selection()

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

# TRAIN MODELS
if __name__ == "__main__":
    # For running graphviz if it doesn't work on Windows
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

    trees = []
    stats = []


    doc_list = ["definitive/result1.txt", "definitive/result2.txt",
                "definitive/result3.txt", "definitive/result4.txt",
                "definitive/result5.txt", "definitive/result6.txt"]

    for doc in doc_list:
        with open(doc, "r") as f:
            lines = f.readlines()
            for line in lines:
                # Ignore lines that are not seed lines
                if line[:4] != "    ": continue
                # Find the stats
                tree = line.split("; TREE: ")[1]
                line = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)
                # Append the stats
                stats.append([float(elem) for elem in line[0:7]] + [tree]) #seed, train acc, test acc, train sens, test sens, train spec, test spec
            f.close()

    train_avg_acc = numpy.mean([stat[1] for stat in stats])
    test_avg_acc = numpy.mean([stat[2] for stat in stats])
    train_std_acc = numpy.std([stat[1] for stat in stats])
    test_std_acc = numpy.std([stat[2] for stat in stats])
    train_max_acc = numpy.max([stat[1] for stat in stats])
    test_max_acc = numpy.max([stat[2] for stat in stats])

    train_avg_sens = numpy.mean([stat[3] for stat in stats])
    test_avg_sens = numpy.mean([stat[4] for stat in stats])
    train_std_sens = numpy.std([stat[3] for stat in stats])
    test_std_sens = numpy.std([stat[4] for stat in stats])
    train_max_sens = numpy.max([stat[3] for stat in stats])
    test_max_sens = numpy.max([stat[4] for stat in stats])

    train_avg_spec = numpy.mean([stat[5] for stat in stats])
    test_avg_spec = numpy.mean([stat[6] for stat in stats])
    train_std_spec = numpy.std([stat[5] for stat in stats])
    test_std_spec = numpy.std([stat[6] for stat in stats])
    train_max_spec = numpy.max([stat[5] for stat in stats])
    test_max_spec = numpy.max([stat[6] for stat in stats])

    print(f"Number of runs: {len(stats)}\n"
          f"train avg acc = {train_avg_acc}\n"
          f"test avg acc = {test_avg_acc}\n"
          f"train std acc = {train_std_acc}\n"
          f"test std acc = {test_std_acc}\n"
          f"train max acc = {train_max_acc}\n"
          f"test max acc = {test_max_acc}\n"
          f"\n"
          f"train avg sens = {train_avg_sens}\n"
          f"test avg sens = {test_avg_sens}\n"
          f"train std sens = {train_std_sens}\n"
          f"test std sens = {test_std_sens}\n"
          f"train max sens = {train_max_sens}\n"
          f"test max sens = {test_max_sens}\n"
          f"\n"
          f"train avg spec = {train_avg_spec}\n"
          f"test avg spec = {test_avg_spec}\n"
          f"train std spec = {train_std_spec}\n"
          f"test std spec = {test_std_spec}\n"
          f"train max spec = {train_max_spec}\n"
          f"test max spec = {test_max_spec}\n"
          )

    sorted_stats = sorted(stats, key=lambda x: x[3])


    seed1 = sorted_stats[-1][0]
    seed2 = sorted_stats[-2][0]
    seed3 = sorted_stats[-3][0]

    print(seed1, seed2, seed3)

    def visualize(string):
        re.sub("if_then_else(.*,.*,.*)", "if .*:")
        # string = string.replace("(", "\n(\n")
        # string = string.replace(")", "\n)\n")
        # string = string.replace(",", "\n")
        #
        # indent = 0
        # lines = string.splitlines()
        # new_lines = []
        # for line in lines:
        #     line = "   " * indent + line
        #     if "(" in line: indent += 1
        #     if ")" in line: indent -= 1
        #     new_lines.append(line)
        #
        # print("\n".join(new_lines))

    print(sorted_stats[-1][7])
    print(sorted_stats[-2][7])
    print(sorted_stats[-3][7])
    tree1 = gp.PrimitiveTree.from_string(sorted_stats[-1][7], pset)
    tree2 = gp.PrimitiveTree.from_string(sorted_stats[-2][7], pset)
    tree3 = gp.PrimitiveTree.from_string(sorted_stats[-3][7], pset)
    print_tree(tree1)
    print_tree(tree2)
    print_tree(tree3)




