# -*- coding: utf-8 -*-
"""
@author: kiliane
"""

import os
import shutil

source_folder = "/data/datasets/Lanhong/E1D3/Training_data"
destination_folder = "/data/datasets/Lanhong/E1D3/Validation_data"

# source_folder = "C:/Users/kilia/Desktop/training"
# destination_folder = "C:/Users/kilia/Desktop/validation"

i=0
# fetch all files
for folder_name in os.listdir(source_folder):
    # print(file_name)
    # construct full file path
    i+=1
    source = source_folder + "/"  + folder_name 
    # print(source)
    destination = destination_folder +"/"  +  folder_name 
    if i%5==0:
        # print(source)
        shutil.move(source, destination)
        print('moved', folder_name)

