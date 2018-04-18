#!/usr/bin/env python2.7
# Nick Rocco and Ryan Loizzo
# Analysis of Fourth Downs in College Football
# Machine Learning Final Project

import sys
import os
import string
import math
import operator
from sklearn import tree
from sklearn import cluster
import pandas as pd
import matplotlib.pyplot as plt

def get_bin(field_pos):
    b = 3
    poss_bins = []
    while b < 100:
        poss_bins.append(b)
        b += 5
    # find the bin that is closest to the field position in question
    poss_bins = map(lambda x: x - field_pos, poss_bins)
    for val in poss_bins:
        if abs(val) <= 2:
            return val + field_pos

def get_class_decision(field_pos, exp_pts_bins, counts_bins):
    b = get_bin(field_pos)
    # determine if one class is overwhelming majority
    # i = 0: punt
    if counts_bins[b][1] + counts_bins[b][2] > 0:
        ratio = float(counts_bins[b][0]) / float(counts_bins[b][1] + counts_bins[b][2])
        if ratio >= 10:
            return 1
    else:
        if counts_bins[b][0] > 0:
            return 1
        else:
            return None # all 3 have count 0
    # i = 1: FG
    if counts_bins[b][0] + counts_bins[b][2] > 0:
        ratio = float(counts_bins[b][1]) / float(counts_bins[b][0] + counts_bins[b][2])
        if ratio >= 10:
            return 2
    else:
        if counts_bins[b][1] > 0:
            return 2
        else:
            return None # all 3 have count 0
    # i = 2: offensive play
    if counts_bins[b][0] + counts_bins[b][1] > 0:
        ratio = float(counts_bins[b][2]) / float(counts_bins[b][0] + counts_bins[b][1])
        if ratio >= 10:
            return 3
    else:
        if counts_bins[b][2] > 0:
            return 3
        else:
            return None # all 3 have count 0
    # if no class has majority, return class with highest expected points
    if exp_pts_bins[b][0] > exp_pts_bins[b][1] and exp_pts_bins[b][0] > exp_pts_bins[b][2]:
        return 1
    elif exp_pts_bins[b][1] > exp_pts_bins[b][0] and exp_pts_bins[b][1] > exp_pts_bins[b][2]:
        return 2  
    elif exp_pts_bins[b][2] > exp_pts_bins[b][0] and exp_pts_bins[b][2] > exp_pts_bins[b][1]:
        return 3
    else:
        return None

def find_dist(point, centroid):
    running_sum = 0.0
    for i in range(len(point)):
        running_sum += float(pow((point[i] - centroid[i]),2))
    return pow(running_sum,0.5)


if __name__ == '__main__':
    filenames = ['punt.csv', 'field_goals.csv', 'off_plays.csv']
    
    data_headers = ["Yds_to_Gain","Field_Pos","Time_Rem","Score_Diff","Pts_Next_Poss","Class"]
    data_full = []
    data = []
    relevant_stats = set([1,2,5,7,8,9])
    quarters = [0]
    yds_to_gain = None
    if len(sys.argv) > 1:
        yds_to_gain = int(sys.argv[1])

    for item in filenames:
        with open(item) as f:
            for line in f:
                line = line.rstrip()
                if item == 'punt.csv':
                    line += ',1'
                elif item == 'field_goals.csv':
                    line += ',2'
                elif item == 'off_plays.csv':
                    line += ',3'
                l = []
                line = line.split(',')
                quarters.append(float(line[4]))
                for i in range(len(line)):
                    if i in relevant_stats:
                        if i != 5:
                            l.append(float(line[i].strip()))
                        else:
                            l.append(line[i].strip())
                data_full.append(l)

    for line in data_full:
        if yds_to_gain >= 10:
            if line[0] >= 10:
                data.append(line)
        elif yds_to_gain == None:
            data.append(line)
        else:
            if line[0] == yds_to_gain:
                data.append(line)

    for i in range(len(data)):
        line = data[i]
        quarter = quarters[i]
        # convert field position to distance from endzone
        if line[1] < 0:
            line[1] = float((50 -abs(line[1])) + 50)
        # convert time to decimal as opposed to mm:ss
        time = line[2]
        minutes = float(time[:-3])
        seconds = float(time[-2:])
        seconds = seconds / 60.
        line[2] = (15 * (4 - quarter)) + minutes + seconds
  
    data = pd.DataFrame(data, columns = data_headers)
    X_headers = list(data)[:5]
    data = data.as_matrix()

    # create dictionary of 5 yard bins for every possible field position
    # field_pos_bins = {bin: [pts from punts, pts from FG, pts from off plays]}
    exp_pts_bins = {}
    counts_bins = {}
    b = 3
    while b < 100:
        exp_pts_bins[b] = [0,0,0]
        counts_bins[b] = [0,0,0]
        b += 5

    # get the average points earned/lost after each play type in each bin
    for line in data:
        b = get_bin(line[1])
        Class = int(line[5] - 1)
        exp_pts_bins[b][Class] += line[4]
        counts_bins[b][Class] += 1

    for b in exp_pts_bins.keys():
        for i in range(3):
            if counts_bins[b][i] != 0:
                exp_pts_bins[b][i] /= float(counts_bins[b][i])
  
    cccc = 0
    for line in data:
        b = get_bin(line[1])
        new_class = get_class_decision(b,exp_pts_bins,counts_bins)
        if new_class != None: 
            line[5] = new_class
        #otherwise, class remains unchanged

    X = data[:,:5]
    y = data[:,5]

    dt = tree.DecisionTreeClassifier(criterion="entropy",min_impurity_decrease=.01)
    dt.fit(X,y)
    
    with open("tree.txt", "w") as f:
        f = tree.export_graphviz(dt, out_file=f,feature_names=X_headers)
   
    km = cluster.KMeans(n_clusters=3)
    km.fit(X,y)

    cluster1 = set([])
    cluster2 = set([])
    cluster3 = set([])

    for i in range(len(X)):
        point = X[i]
        d1 = find_dist(point,km.cluster_centers_[0])
        d2 = find_dist(point,km.cluster_centers_[1])
        d3 = find_dist(point,km.cluster_centers_[2])
  
        if d1 < d2 and d1 < d3:
            cluster1.add(i)
        elif d2 < d1 and d2 < d3:
            cluster2.add(i)
        else:
            cluster3.add(i)
    
    cluster1_counts = [0,0,0]
    cluster2_counts = [0,0,0]
    cluster3_counts = [0,0,0]

    for i in range(len(y)):
        if i in cluster1:
            cluster1_counts[int(y[i]-1)] += 1
        elif i in cluster2:           
            cluster2_counts[int(y[i]-1)] += 1
        else:
            cluster3_counts[int(y[i]-1)] += 1

    print cluster1_counts
    print cluster2_counts
    print cluster3_counts

