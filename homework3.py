# CS561 Homework 1, 2022 Fall
# Naiyu Wang

import numpy as np
import pandas as pd
# from scipy import NearestNeighbors


def loc_to_matrix(locations, numoflocs):
    # create empty matrix loc_matrix to store all locations later
    matrix = np.empty((numoflocs, 3))

    # convert each location's coordinate to a 3d numpy array
    for i in range(numoflocs):
        location = locations[i].split()
        arr = np.array(location) # the 3d coordinate for one location

        # fill in the location matrix
        matrix[i] = np.array([arr])

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
    # Sort the given path(np array) and return a sorted array (new)
    # based on each location's distance to the starting point

    sorted_arr = np.array([])
    # keep the starting point
    # sorted_arr[0] = location_ref[0]
    sorted_arr = np.append(sorted_arr, path.locations[0])

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

        sorted_arr = np.append(sorted_arr, destination)
        visited.append(destination)

    # for i in sorted_arr:
    #     print(i.coordinate, end="\t")
    # print("")
    return sorted_arr


#def sort_based_on_x ():
    # TODO: optimization.
    # path = path[:, path[0, :].argsort()]


# returns a list of paths
def create_init_pop(location_list, n):
    pop = []
    seed = 42
    for i in range(n):
        chromo = location_list.copy()
        # shuffle the copied list
        np.random.seed(seed)
        np.random.shuffle(chromo)
        chromopath = Path(chromo)
        pop.append(chromopath)
        seed += 2
    return pop


# sort the generation based on their distances, from short to long.
def sort_generation(gen):
    gen.sort(key=lambda x: x.distance, reverse=False)


# Mutate the last half (the longer distant paths) only!
def mutate(gen):
    len = len(gen)
    half = len // 2

    # for i in range(half, len):

"""
A class used to represent a path

...

Attributes
----------
path : np.array
    an array of locations 
    first element = last element for a complete path
distance : float 
    the total length of the path, starting with max
fitness: float
    1 / distance. As the less the distance is, the better the path is.

Methods
-------
get_total_dist: 
   get the total distance of the path (including end -> start)
   
"""


class Path:
    def __init__(self, array):
        self.locations = array
        self.distance = 0
        self.size = array.size

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
        distance = np.sqrt(dis_x ** 2 + dis_y ** 2)

        return distance

    def get_distance_3d(self, loc2):
        # compute the 3d distance between 2 coordinates
        dis_x = abs(self.x - loc2.x)
        dis_y = abs(self.y - loc2.y)
        dis_z = abs(self.z - loc2.z)
        distance = np.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)

        return distance


if __name__ == '__main__':
    with open("input.txt") as file:
        lines = file.readlines()
        if int(lines[0]) != len(lines[1:]):
            print("Number of locations does not match file content, exiting.")
            exit(1)
    num_of_loc = int(lines[0])
    loc_matrix = loc_to_matrix(lines[1:], num_of_loc)

    default_arr = np.empty(0)

    # Fill default path using loc_matrix
    for item_index in range(len(loc_matrix)):
        new_loc = Location(loc_matrix[item_index], item_index)
        default_arr = np.append(default_arr, new_loc)

    # calculate distances between any pair of locations and store it in a 2d matrix (NOT NUMPY!!!!)
    distances = get_adjacent_matrix(num_of_loc)

    default_path = Path(default_arr)

    sorted_ar = sort_locations(default_path, default_path, distances)
    sorted_path = Path(sorted_ar)
    default_dist = default_path.get_total_dist()
    sorted_dist = sorted_path.get_total_dist()

    # generate initial random population using shuffle
    gen1 = create_init_pop(default_arr, 10)

    # for i in range(10):
    #     gen1[i].printout()
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

    # for i in gen1:
    #     i.printout()
    #     print(i.get_total_dist())

    # Sort gen 1 based on their distance
    sort_generation(gen1)
