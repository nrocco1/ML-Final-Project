#!/usr/bin/env python3
# Nick Rocco and Ryan Loizzo
# Analysis of Fourth Downs in College Football
# Machine Learning Final Project

import sys
import os
import string
import math
import operator
import graphviz

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
    
    data = []

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
                l = [x.strip() for x in line.split(',')]
                data.append(l)

    testing_data = []

   # for instance in sys.stdin:
   #     instance = instance.rstrip()
   #     i = [x.strip() for x in instance.split()]
   #     testing_data.append(i)

    h = entropy(data) 
