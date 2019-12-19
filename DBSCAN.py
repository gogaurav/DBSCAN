#!/usr/bin/env python
# coding: utf-8
# Author: Gourang Gaurav

import numpy as np


class Dbscan:

    def __init__(self, minpts, eps):
        self.minpts = minpts
        self.eps = eps
        self.cluster_id = 1
        self.point_cl_id = None
        self.set_of_points = None
        self.len_data = None

    def region_query(self, point):
        eps_neigh = []
        for i in range(self.len_data):
            dist = np.linalg.norm(self.set_of_points[i]
                                  - self.set_of_points[point])  # check for multiple clusters
            if dist < self.eps:
                eps_neigh.append(i)
        return eps_neigh

    def expand_cluster(self, point):
        seeds = self.region_query(point)
        if len(seeds) < self.minpts:
            self.point_cl_id[point] = -1
            return False
        else:
            for seed in seeds:
                self.point_cl_id[seed] = self.cluster_id
            seeds.remove(point)
            while len(seeds) != 0:
                cur_p = seeds[0]
                cur_seeds = self.region_query(cur_p)
                len_cur_seeds = len(cur_seeds)
                if len_cur_seeds >= self.minpts:
                    for i in range(len_cur_seeds):
                        if self.point_cl_id[cur_seeds[i]] in (0, -1):
                            if self.point_cl_id[cur_seeds[i]] == 0:
                                seeds.append(cur_seeds[i])
                            self.point_cl_id[cur_seeds[i]] = self.cluster_id
                seeds.remove(cur_p)
            return True

    def dbscan(self, set_of_points):
        self.set_of_points = set_of_points
        self.len_data = len(set_of_points)
        self.point_cl_id = np.zeros(self.len_data, dtype=int)
        for point in range(self.len_data):
            if self.point_cl_id[point] == 0:
                if self.expand_cluster(point):
                    self.cluster_id += 1

        return self.point_cl_id

