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
from scipy.stats import kruskal

# TRAIN MODELS
if __name__ == "__main__":
    # For running graphviz if it doesn't work on Windows
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

    trees = []
    stats = [[],[]]

    doc_list = [["definitive/result1.txt", "definitive/result2.txt",
                "definitive/result3.txt", "definitive/result4.txt",
                "definitive/result5.txt", "definitive/result6.txt"], ["definitive/result1.txt", "definitive/result2.txt",
                  "definitive/result3.txt", "definitive/result4.txt",
                  "definitive/result5.txt", "definitive/result6.txt"]]

    for i in range(2):
        for doc in doc_list[i]:
            with open(doc, "r") as f:
                lines = f.readlines()
                for line in lines:
                    # Ignore lines that are not seed lines
                    if line[:4] != "    ": continue
                    # Find the stats
                    tree = line.split("; TREE: ")[1]
                    line = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)
                    # Append the stats
                    stats[i].append(float(line[2])) #seed, train acc, test acc, train sens, test sens, train spec, test spec
                f.close()

    dist1 = numpy.mean([stat[0] for stat in stats])
    dist2 = numpy.mean([stat[1] for stat in stats])

    stat, p = kruskal(dist1, dist2)




