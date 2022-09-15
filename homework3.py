# CS561 Homework 1, 2022 Fall
# Naiyu Wang

import numpy as np, pandas as pd


def loc_to_matrix(locations, numoflocs):
    # create empty matrix loc_matrix to store all locations later
    loc_matrix = np.empty((numoflocs, 3), dtype=np.uint32)

    # convert each location's coordinate to a 3d numpy array
    for i in range(numoflocs):
        location = locations[i].split()
        arr = np.array(location, dtype=np.uint32) # the 3d coordinate for one location

        # fill in the location matrix
        loc_matrix[i] = np.array([arr])

    return loc_matrix


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
    distance = np.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)

    return distance


"""
A class used to represent a path

...

Attributes
----------
path : np.array
    an 2d array with each row being a coordinate (1x3 numpy array) of a location
    column number = number of coordinates in the array
    first row = last row for a complete path
distance : float 
    the total length of the path
fitness: float
    1 / distance. As the less the distance is, the better the path is.

Methods
-------
says(sound=None)
    Prints the animals name and what sound it makes
"""


class Path:
    def __init__(self, path):
        self.path = path
        self.distance = np.finfo(np.float32).max
        self.fitness = 0

    def calculate_fitness(self):
        if self.distance != 0:
            fitness = 1 / float(self.distance)
        return fitness


if __name__ == '__main__':
    with open("input.txt") as file:
        lines = file.readlines()
        if int(lines[0]) != len(lines[1:]):
            print("Number of locations does not match file content, exiting.")
            exit(1)
    loc_matrix = loc_to_matrix(lines[1:], int(lines[0]))

    # Create paths using loc_matrix

