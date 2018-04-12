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
import pandas as pd

def entropy(data):
    
    n_punt = 0
    n_fg = 0
    n_off = 0
    n = 0

    for line in data:
        if line[-1] == '1':
            n_punt += 1
        elif line[-1] == '2':
            n_fg += 1
        elif line[-1] == '3':
            n_off += 1
        n += 1

    prop_punt = n_punt/n
    prop_fg = n_fg/n
    prop_off = n_off/n
   
    print (prop_punt, prop_fg, prop_off)
 
    h = -prop_punt * math.log2(prop_punt) - prop_fg * math.log2(prop_fg) - prop_off * math.log2(prop_off)
        
    return h

if __name__ == '__main__':
    filenames = ['punt.csv', 'field_goals.csv', 'off_plays.csv']
    
    #data = [["Field_Pos","Yds_to_Gain","Time_Rem","Score_Diff","Pts_Next_Poss","Class"]]
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
            line[1] = float(abs(line[1]) + 50)
        # convert time to decimal as opposed to mm:ss
        time = line[2]
        minutes = float(time[:-3])
        seconds = float(time[-2:])
        seconds = seconds / 60.
        line[2] = (15 * (4 - quarter)) + minutes + seconds
  
    data = pd.DataFrame(data).as_matrix()

    X = data[:,:4]
    y = data[:,5]

    dt = tree.DecisionTreeClassifier(criterion="entropy",min_impurity_decrease=.01)
    dt.fit(X,y)
    
    with open("tree.txt", "w") as f:
        f = tree.export_graphviz(dt, out_file=f)
   



   
   # for instance in sys.stdin:
   #     instance = instance.rstrip()
   #     i = [x.strip() for x in instance.split()]
   #     testing_data.append(i)

    #h = entropy(data)
