# CS561 Homework 1, 2022 Fall
# Naiyu Wang

import random
import math


def loc_to_matrix(locations, numoflocs):
    # print("locations: ", locations)
    # print("numoflocs:",numoflocs)
    # create empty matrix loc_matrix to store all locations later
    matrix = []
    coordinate = []
    # convert each location's coordinate to a 3d numpy array
    for i in range(numoflocs):
        for j in range(3):
            component = locations[i].strip().split(" ")[j]
            coordinate.append(int(component))
            # fill in the location matrix
        matrix.append(coordinate)
        coordinate = []
    # print("matrix: ", matrix, "\nlen of matrix = ", len(matrix))
    return matrix


def get_adjacent_matrix(loc_num):
    # distances is a 2d array of adjacent matrix
    distances = [[0] * loc_num for i in range(loc_num)]
    for loc_start in range(num_of_loc):
        for loc_to in range(num_of_loc):
            if loc_start == loc_to:
                distances[loc_start][loc_to] = 0
                distances[loc_to][loc_start] = 0
            else:
                dist = default_arr[loc_start].get_distance_3d(default_arr[loc_to])
                distances[loc_start][loc_to] = dist
                distances[loc_to][loc_start] = dist

    return distances


def sort_locations(path, orig_path, distances):
    # Sort the given path (list) and return a sorted array (new)
    # based on each location's distance to the starting point

    sorted_arr = []
    # keep the starting point
    # sorted_arr[0] = location_ref[0]
    sorted_arr.append(path.locations[0])

    visited = []
    visited.append(path.locations[0])

    while len(visited) < path.size:

        # get every distance value starting with current location being visited
        possible_dist = distances[sorted_arr[-1].index]

        # iterate through the row to find the minimum distance that is nonzero
        # i is the index of the location in adjacent matrix, find through orig_path
        min = 9999999
        destination = None

        for i in range(len(possible_dist)):
            dst = orig_path.locations[i]
            if possible_dist[i] != 0 and possible_dist[i] < min and dst not in visited:
                min = possible_dist[i]
                destination = dst
            else:
                continue

        sorted_arr.append(destination)
        visited.append(destination)

    # for i in sorted_arr:
    #     print(i.coordinate, end="\t")
    # print("")
    return sorted_arr


# returns a list of paths
def create_init_pop(location_list, n):
    pop = []
    # seed = 42
    for i in range(n):
        chromo = location_list.copy()
        # shuffle the copied list
        random.shuffle(chromo)
        chromopath = Path(chromo)
        pop.append(chromopath)
        # seed += 2
    return pop


# sort the generation based on their distances, from short to long.
def sort_generation(gen):
    # for i in gen:
    #     i.get_total_dist()
    gen.sort(key=lambda x: x.distance, reverse=False)


# Crossover using OX1 (https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm)
# returns a new path
def crossover(parent1, parent2):
    chromosome_len = parent1.size

    # get 2 random crossover points
    xover_idx1 = random.randint(0, chromosome_len - 1) # random.randint is inclusive on the right
    xover_idx2 = random.randint(0,  chromosome_len - 1)
    while xover_idx1 == xover_idx2:
        xover_idx2 = random.randint(0,  chromosome_len - 1)
    if xover_idx1 > xover_idx2:
        temp = xover_idx1
        xover_idx1 = xover_idx2
        xover_idx2 = temp

    # lookup table to keep track of locations visited
    values = [0] * chromosome_len
    path1 = parent1.locations
    path2 = parent2.locations
    lookup_table = {loc.to_string(): visited for loc, visited in zip(path1, values)}
    #
    # print("parent1: ", end="\t")
    # parent1.printout()
    # print("parent2: ", end="\t")
    # parent2.printout()
    # print("crossover starts: ",xover_idx1, "ends at: ",xover_idx2)

    # A list of locations (a new path)
    tester_loc = Location(([-1, -1, -1]), -1) # the dummy location for debugging
    child = [tester_loc] * chromosome_len
    # copy parent1[idx1:idx2] to child[idx1, idx2]
    for i in range(xover_idx1, xover_idx2 + 1):
        loc = path1[i].to_string()
        if lookup_table[loc] == 0:
            child[i] = path1[i]
            lookup_table[loc] = 1
    #
    # for i in child:
    #     if i == tester_loc: # Not filled in
    #         print("?")
    #     else:
    #         i.printout()
    # print("\n")

    # fill the rest of the chromosome with parent2's unvisited locations, starting with xover_idx1
    p2idx = xover_idx1

    for i in range(chromosome_len):

        if p2idx == chromosome_len:
            p2idx = 0 # print("p2idx is set to 0")

        if child[i] == tester_loc: # if current space is empty, fill it
            # if this location is visited, then skip to the next.
            while lookup_table[(path2[p2idx].to_string())] != 0:
                # start from parent2[xover_idx1] and moves to the right
                if p2idx <= chromosome_len - 2:
                    p2idx += 1
                else:  # if p2idx == chromosome_len - 1, aka the last legal index in path2
                    p2idx = 0
                    # print("p2idx is set to 0")
            if lookup_table[(path2[p2idx].to_string())] == 0:
                child[i] = path2[p2idx]
                lookup_table[(path2[p2idx].to_string())] = 1
                p2idx += 1

    return Path(child)


def make_prob_roulette(pop): # each p in pop is a path
    # keys = pop
    probabilities = {}
    sumdist = 0
    for key in pop:
        sumdist += 1 / key.distance
    for key in pop:
        probabilities[key] = (1 / key.distance) / sumdist
    probabilities[key] = (1 / key.distance)
    return probabilities


def parent_selection(roulette):
    selection1 = random.random()
    selection2 = random.random()
    min1 = 9999
    min2 = 9999
    closest1 = None
    closest2 = None
    for key in roulette:
        if abs(roulette[key] - selection1) < min1:
            closest1 = key
            min1 = (roulette[key] - selection1)
        if abs(roulette[key] - selection2) < min2:
            closest2 = key
            min2 = (roulette[key] - selection2)
    return closest1, closest2


def next_gen(parents, population):
    nextgen = []
    # A generation is a list of Paths
    probabilities = make_prob_roulette(parents)
    for i in range(population):
        p1, p2 = parent_selection(probabilities)
        child = crossover(p1, p2)
        nextgen.append(child)
    # print("poplulation = ", population,". Nextgen has ", len(nextgen))
    # for i in nextgen:
    #     i.printout()
    #      print(i.distance)

    # Sort the offsprings
    sort_generation(nextgen)
    # print("------------------------OFFSPRINGS: ------------------------------")
    offsprings = [None] * population
    for i in range(population):
        offsprings[i] = nextgen[i]
        # offsprings[i].printout()
        # print(offsprings[i].distance)

    # Add mutation
    mutate(offsprings, 0.5)

    return offsprings


def mutate(gen, rate):
    length = len(gen)
    mutation_start = (int)(length * 2 / 3)
    for chromosome in gen[mutation_start:]:
        if random.random() < rate:
            mutate_by_shuffling(chromosome)
            mutate_by_swapping(chromosome)


def mutate_by_swapping(chromosome):
    swapidx1 = random.randint(0, chromosome.size - 1)
    swapidx2 = random.randint(0, chromosome.size - 1)
    temp = chromosome.locations[swapidx2]
    chromosome.locations[swapidx2] = chromosome.locations[swapidx1]
    chromosome.locations[swapidx1] = temp


def mutate_by_reverse(chromosome):
    chromosome.locations.reverse()


def mutate_by_shuffling(chromosome):
    random.shuffle(chromosome.locations)

"""
A class used to represent a path

...

Attributes
----------
locations : np.array
    an array of locations 
    first element = last element for a complete path
distance : float 
    the total length of the path, starting with max
size: int
    the length of np.array locations

Methods
-------
get_total_dist: 
   get the total distance of the path (including end -> start)

printout:
    print all the coordinates for debugging purposes
"""


class Path:
    def __init__(self, array):
        self.locations = array
        self.size = len(array)
        self.distance = self.get_total_dist()

    def get_total_dist(self):
        dist = 0
        for locindex in range(self.size):
            # for each location in this path, get the distance between itself and next location
            start_index = self.locations[locindex].index
            if locindex < self.size-1:
                end_index = self.locations[locindex+1].index
            else: # the last location in path. Add a distance from this loc to the starting point
                end_index = self.locations[0].index
            tempdist = distances[start_index][end_index]
            dist += tempdist
        self.distance = dist
        return dist

    def printout(self):
        print("Start: ", end="")
        for i in self.locations:
            print(i.coordinate, end="\t")
        print(" End.\n")

"""
A class used to represent a location
...

Attributes
----------
x, y, z : integer 
    x, y, z coordinate 

coordinate: np.array
    the original coordinate
    
index: int
    indicates where to find this location in the distance matrix 

Methods
-------
get_distance_2d(self, loc2)
    get 2d distance between this location and loc2
    
get_distance_3d(self, loc2)
    get 3d distance between this location and loc2
    
equals(self, loc2)
    compare if the coordinates are the same
    
to_string()
    convert coordinate (np array) into a string for hashing purposes

printout()
    print the coordinate
"""


class Location:
    def __init__(self, coord, index):
        self.x = coord[0]
        self.y = coord[1]
        self.z = coord[2]
        self.coordinate = coord
        self.index = index

    def __eq__(self, loc2):
        return self.x == loc2.x and self.y == loc2.y and self.z == loc2.z

    def get_distance_2d(self, loc2):
        # compute the 2d distance between 2 coordinates
        dis_x = abs(self.x - loc2.x)
        dis_y = abs(self.y - loc2.y)
        distance = math.sqrt(dis_x ** 2 + dis_y ** 2)

        return distance

    def get_distance_3d(self, loc2):
        # compute the 3d distance between 2 coordinates
        dis_x = abs(self.x - loc2.x)
        dis_y = abs(self.y - loc2.y)
        dis_z = abs(self.z - loc2.z)
        distance = math.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)

        return distance

    def to_string(self):
        coord = " ".join(str(self.coordinate))
        return coord

    def printout(self):
        print(self.to_string())

    def output_format(self):
        string = str(self.x) + " " + str(self.y) + " " + str(self.z)
        return string


if __name__ == '__main__':
    with open("input.txt") as file:
        lines = file.readlines()
        if int(lines[0]) != len(lines[1:]):
            print("Number of locations does not match file content, exiting.")
            exit(1)
    num_of_loc = int(lines[0])
    loc_matrix = loc_to_matrix(lines[1:], num_of_loc)

    default_arr = []
    sorted_def = []
    # Fill default path using loc_matrix
    for item_index in range(len(loc_matrix)):
        new_loc = Location(loc_matrix[item_index], item_index)
        default_arr.append(new_loc)
        sorted_def.append(new_loc)

    sorted_def.sort(key=lambda x: x.x)

    # calculate distances between any pair of locations and store it in a 2d matrix (NOT NUMPY!!!!)
    distances = get_adjacent_matrix(num_of_loc)

    default_path = Path(default_arr)
    sorted_path = Path(sorted_def)
    # print("Default path: ",end =" ")
    # default_path.printout()

    # print("sorted input path: ",end =" ")
    # sorted_path.printout()

    sorted_ar = sort_locations(sorted_path, default_path, distances)
    sorted_path = Path(sorted_ar)
    # print(default_path.distance)
    # print(sorted_path.distance)
    # determine population based on number of locations
    pop_max = 450
    pop_min = 50
    dynamic_pop = int(num_of_loc * 2)
    if dynamic_pop > pop_max:
        dynamic_pop = pop_max
    if dynamic_pop < pop_min:
        dynamic_pop = pop_min

    # generate initial random population using shuffle
    gen1 = create_init_pop(default_arr, dynamic_pop)

    gen1max = 0
    longestpath = -1
    for i in range(len(gen1)):
        dist = gen1[i].get_total_dist()
        if gen1max <= dist:
            gen1max = dist
            longestpath = i

    if longestpath >= 0: # if a longest path is found in gen1
        # replace the path of max distance with sorted_path
        gen1[i] = sorted_path

    #  create the next generation
    offs = [None] * dynamic_pop
    for g in range(200):
        gen1 = next_gen(gen1, dynamic_pop)
        # print(gen1[0].distance)

    optimal_path = gen1[0]
    # for p in gen1:
    #     for i in p.locations:
    #         print(i.output_format(), end="\t\t")
    #     print(p.get_total_dist())

    with open("output.txt", "w") as out:
        # print( optimal_path.locations)
        # print(optimal_path.distance)
        for loc in optimal_path.locations:
            out.write(loc.output_format())
            out.write(" ")
            out.write("\n")
        out.write(optimal_path.locations[0].output_format())