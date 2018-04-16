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

def find_dist(point, centroid):
    running_sum = 0.0
    for i in range(len(point)):
        running_sum += float(pow((point[i] - centroid[i]),2))
    return pow(running_sum,0.5)


if __name__ == '__main__':
    filenames = ['punt.csv', 'field_goals.csv', 'off_plays.csv']
    
    data_headers = ["Yds_to_Gain","Field_Pos","Time_Rem","Score_Diff","Pts_Next_Poss","Class"]
    data = []
    relevant_stats = set([1,2,5,7,8,9])
    quarters = [0]

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
                data.append(l)

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
        
            
