#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import fire
import json
import random
import pickle
import os
import glob

import pandas as pd
import numpy as np

class Helper:
    
    def __init__(self, config_path=os.path.dirname(os.path.abspath("__file__"))+'/config.json'):
      
        with open(config_path) as json_file:
            data = json.load(json_file)
            
        curnt_dir = os.path.dirname(os.path.abspath("__file__"))
        
        data_path = data['data_path']
        self.X_path = curnt_dir+data_path['X_path']
        self.Y_path = curnt_dir+data_path['Y_path']
        self.Z_path = curnt_dir+data_path['Z_path']
        
        self.eig_path = curnt_dir+data_path['eig_path'] 
                                          
        params = data['params']

    
# if __name__ == "__main__":
#     fire.Fire(Helper)