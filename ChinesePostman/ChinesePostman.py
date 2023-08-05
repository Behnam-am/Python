import copy
from dataclasses import dataclass
import numpy as np
import random


@dataclass
class Edge:
    id: int
    start: str
    end: str
    weight: float


def find_edge(v1, v2):
    for i in edges:
        if (i.start == v1 and i.end == v2) or \
                (i.start == v2 and i.end == v1) or \
                (v1 == v2):
            return True
    return False


def valid(ch):
    # not a circuit
    if ch[0] != ch[-1]:
        return False

    # if all nearby vertices have edges
    for i in range(len(ch) - 1):
        if not find_edge(ch[i], ch[i + 1]):
            return False

    # if all edges are not in chromosome
    e = edges.copy()
    for i in range(len(ch) - 1):
        for j in range(len(e)):
            if (e[j].start == ch[i] and e[j].end == ch[i + 1]) or (e[j].start == ch[i + 1] and e[j].end == ch[i]):
                del e[j]
                break
        if len(e) == 0:
            break
    if len(e) != 0:
        return False

    # everything is OK
    return 1


def fitness(ch):
    fit = 0
    for i in range(len(ch) - 1):
        if ch[i] == ch[i + 1]:
            continue
        for j in edges:
            if (j.start == ch[i] and j.end == ch[i + 1]) or \
                    (j.start == ch[i + 1] and j.end == ch[i]):
                fit += j.weight
                break
        else:
            fit += 1e10
            break
    ch.append(fit)


def select(type, rate):
    num_of_parents = int(popSize * rate)
    parents = []
    # random
    if type == 1:
        parents = random.sample(range(0, popSize), num_of_parents)

    # rank-base
    elif type == 2:
        total = int((popSize / 2) * (popSize + 1))
        pick = []
        for _ in range(num_of_parents // 2):
            rnd = random.randint(1, total)
            t = total
            for j in range(popSize, 0, -1):
                t -= j
                if rnd > t:
                    pick.append(popSize - j)
                    break
            while True:
                rnd = random.randint(1, total)
                t = total
                flag = False
                for j in range(popSize, 0, -1):
                    t -= j
                    if rnd > t:
                        if pick[0] == popSize - j:
                            continue
                        pick.append(popSize - j)
                        flag = True
                        break
                if flag:
                    break
            parents += pick
            pick = []

        if num_of_parents % 2 == 1:
            rnd = random.randint(1, total)
            t = total
            for j in range(popSize, 0, -1):
                t -= j
                if rnd > t:
                    parents.append(popSize - j)
                    break
    # elitism
    elif type == 3:
        parents = [i for i in range(num_of_parents)]

    return parents


def crossover(type, ch1, ch2):
    # two point
    if type == 1:
        rnd = random.sample(range(0, chromosomeSize), 2)
        rnd.sort()
        children.append(ch1[:rnd[0]] + ch2[rnd[0]:rnd[1] + 1] + ch1[rnd[1] + 1:])
        children.append(ch2[:rnd[0]] + ch1[rnd[0]:rnd[1] + 1] + ch2[rnd[1] + 1:])

    # uniform
    elif type == 2:
        child1 = []
        child2 = []
        mask = [random.randint(0, 1) for _ in range(chromosomeSize)]
        for i in range(len(mask)):
            if mask[i]:
                child1.append(ch1[i])
                child2.append(ch2[i])
            else:
                child1.append(ch2[i])
                child2.append(ch1[i])

        children.append(child1)
        children.append(child2)

    # shuffle
    elif type == 3:
        rnd = random.randint(0, chromosomeSize - 1)
        parts = [ch1[:rnd], ch1[rnd:], ch2[:rnd], ch2[rnd:]]
        for i in parts:
            random.shuffle(i)

        children.append(parts[0] + parts[3])
        children.append(parts[2] + parts[1])


def mutation(type, ch):
    ch = copy.deepcopy(ch)
    # substitution
    if type == 1:
        rnd = random.sample(range(0, chromosomeSize), 2)
        ch[rnd[0]], ch[rnd[1]] = ch[rnd[1]], ch[rnd[0]]

    # update
    elif type == 2:
        rnd = random.randint(0, chromosomeSize - 1)
        rndV = random.randint(0, len(vertices) - 1)
        while ch[rnd] == vertices[rndV]:
            rndV = random.randint(0, len(vertices) - 1)
        ch[rnd] = vertices[rndV]

    # scramble
    elif type == 3:
        rnd = random.sample(range(0, chromosomeSize), 2)
        rnd.sort()
        random.shuffle(ch[rnd[0]: rnd[1] + 1])

    children.append(ch)


if __name__ == '__main__':
    with open("edges.txt", "r") as file:
        data = [[x for x in line.split()] for line in file]
    for i in range(len(data)):
        data[i][2] = float(data[i][2])

    # define vertices
    vertices = np.array(data)[:, [0, 1]]
    vertices = np.unique(vertices.ravel())

    # randomly select selection, crossover, mutation algorithm
    selectionType = random.randint(1, 3)
    crossoverType = random.randint(1, 3)
    mutationType = random.randint(1, 3)

    # variables
    edges = []
    pop = []
    popSize = 100
    crossoverRate = 0.8
    mutationRate = 0.04
    bestFitness = 0
    noChangeCounter = 0

    # print parameters
    print("Initial Population: {} individuals".format(popSize))
    selectDict = {1: "random", 2: "rank based", 3: "elitism"}
    crossoverDict = {1: "two point", 2: "uniform", 3: "shuffle"}
    mutationDict = {1: "substitution", 2: "update", 3: "scramble"}
    print("Selection Type: {}".format(selectDict[selectionType]))
    print("Crossover Type: {}, Crossover Rate: {}".format(crossoverDict[crossoverType], crossoverRate))
    print("Mutation Type: {}, Mutation Rate: {}".format(mutationDict[mutationType], mutationRate))

    # define edges
    for i in range(len(data)):
        edges.append(Edge(i, data[i][0], data[i][1], data[i][2]))

    chromosomeSize = 2 * len(edges) + 1

    # Initialize the Population
    for i in range(popSize):
        while True:
            chromosome = np.random.choice(vertices, chromosomeSize, replace=True)
            if valid(chromosome):
                break
        pop.append(chromosome.tolist())
        fitness(pop[-1])
        print("{} of {} Valid Individual Found.".format(len(pop), popSize))

    # sort according to fitness
    pop = sorted(pop, key=lambda x: x[-1], reverse=False)

    iteration = 0
    while True:
        iteration += 1
        print("Iteration Number: {}".format(iteration))
        children = []
        parents = select(selectionType, crossoverRate)
        for i in range(0, len(parents), 2):
            crossover(crossoverType, pop[parents[i]][:-1], pop[parents[i + 1]][:-1])
        parents = select(selectionType, mutationRate)
        for i in range(len(parents)):
            mutation(mutationType, pop[parents[i]][:-1])

        # calculate fitness for valid children and delete invalid children
        for i in range(len(children) - 1, -1, -1):
            if valid(children[i]):
                fitness(children[i])
            else:
                del (children[i])

        if len(children):
            pop += children
            # remove duplicate individual
            pop = [list(i) for i in set(tuple(j) for j in pop)]
            # sort according to fitness
            pop = sorted(pop, key=lambda x: x[-1], reverse=False)
            # select top individuals
            pop = pop[:popSize]

        # stop condition
        if bestFitness == pop[0][-1]:
            noChangeCounter += 1
            if noChangeCounter == 10000:
                break
        else:
            bestFitness = pop[0][-1]
            noChangeCounter = 0

    print("path is {} and length is {}".format(pop[0][:-1], pop[0][-1]))
