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


def get_distance_2d(loc1, loc2):
    # compute the 2d distance between 2 coordinates
    dis_x = abs(loc1[0] - loc2[0])
    dis_y = abs(loc1[1] - loc2[1])
    distance = np.sqrt(dis_x ** 2 + dis_y ** 2)

    return distance


def get_distance_3d(loc1, loc2):
    # compute the 3d distance between 2 coordinates
    dis_x = abs(loc1[0] - loc2[0])
    dis_y = abs(loc1[1] - loc2[1])
    dis_z = abs(loc1[2] - loc2[2])
    distance = np.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2,)

    return distance


def get_distances_list(loc_num):
    distances = []
    for loc_start in range(num_of_loc):
        for loc_to in range(num_of_loc):
            if loc_start == loc_to:
                distances.append(Distance(default_arr[loc_start], default_arr[loc_to], 0))
                distances.append(Distance(default_arr[loc_to], default_arr[loc_start], 0))
            else:
                dist = default_arr[loc_start].get_distance_3d(default_arr[loc_to])
                distances.append(Distance(default_arr[loc_start], default_arr[loc_to], dist))
                distances.append(Distance(default_arr[loc_to], default_arr[loc_start], dist))

    return distances


def sort_locations(path, orig_path, distances):
    # Sort the given path(np array) and return a sorted array (new)
    # based on each location's distance to the starting point

    location_ref = np.empty([path.size], dtype=object)

    for loc_index in range(path.size):
        for orig_path_index in range(path.size):
            if path.array[loc_index] == (orig_path.array[orig_path_index]):
                location_ref[loc_index] = orig_path.array[orig_path_index]
                continue

    # sorted_arr = np.empty([path.size], dtype=object)
    sorted_arr = np.array([])
    # keep the starting point
    # sorted_arr[0] = location_ref[0]
    sorted_arr = np.append(sorted_arr, location_ref[0])

    # sort the distances array
    distances.sort(key=lambda x: x.dist)

    visited = []
    visited.append(location_ref[0])

    while len(visited) < path.size:
        filtered = filter_dists(distances, sorted_arr[-1])
        filtered.sort(key=lambda x: x.dist)
        for i in range(len(filtered)):
            if filtered[i].end in visited:
                continue
            else:
                sorted_arr = np.append(sorted_arr, filtered[i].end)
                visited.append(filtered[i].end)
                break

    return sorted_arr


def filter_dists(distance_array, start):
    filtered = []
    for dis in distance_array:
        if dis.dist > 0 and dis.start == start:
            filtered.append(dis)
    return filtered


def create_init_pop(location_list):
    population = np.empty([])


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

"""


class Path:
    def __init__(self, array):
        self.array = array
        self.distance = np.finfo(np.float32).max
        self.fitness = 0
        self.size = array.size

    def calculate_fitness(self):
        if self.distance != 0:
            fitness = 1 / float(self.distance)
        return fitness


class Distance:
    def __init__(self, start, end, dist):
        self.start = start
        self.end = end
        self.dist = dist

    def printout(self):
        print("from ", self.start.x, self.start.y, self.start.z, "to ", self.end.x, self.end.y, self.end.z, ", distance is ", self.dist)


"""
A class used to represent a location
...

Attributes
----------
x, y, z : integer 
    x, y, z coordinate 

coordinate: np.array
    the original coordinate

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
    def __init__(self, coord):
        self.x = coord[0]
        self.y = coord[1]
        self.z = coord[2]
        self.coordinate = coord

    def get_distance_2d(self, loc2):
        # compute the 2d distance between 2 locations
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

    def __eq__(self, loc2):
        return self.x == loc2.x and self.y == loc2.y and self.z == loc2.z


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
    for item in loc_matrix:
        new_loc = Location(item)
        default_arr = np.append(default_arr, new_loc)

    # for i in range(num_of_loc):
    #     print(default_arr[i].x, default_arr[i].y, default_arr[i].z)

    default_path = Path(default_arr)
    population = 10
    gene_pool_init = np.empty(0)

    # calculate distances between any pair of locations
    distances = get_distances_list(num_of_loc)

    # for d in distances:
    #     d.printout()

    sorted_ar = sort_locations(default_path, default_path, distances)



    # for chromosome in range(population):




    #  for gene in range(population):


