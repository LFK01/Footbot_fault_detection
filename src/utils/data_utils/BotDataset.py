import os
import subprocess
import xml.etree.ElementTree
import numpy as np


class BotDataset:

    def __init__(self,
                 train_dataset: np.ndarray,
                 target_train_dataset: np.ndarray,
                 validation_dataset: np.ndarray,
                 target_validation_dataset: np.ndarray,
                 test_dataset: np.ndarray,
                 target_test_dataset: np.ndarray):
        self.train_dataset = train_dataset
        self.target_train_dataset = target_train_dataset
        self.validation_dataset = validation_dataset
        self.target_validation_dataset = target_validation_dataset
        self.test_dataset = test_dataset
        self.target_test_dataset = target_test_dataset


if __name__ == "__main__":
    # NORD min = "-1,12,0" max = "1,14,0"
    # EST min = "12,-1,0" max = "14,1,0"
    # SUD min = "-1,-14,0" max = "1,-12,0"
    # WEST min = "-14,-1,0" max = "-12,1,0"
    positions = {'North': {'min': '-1,12,0',
                           'max': '1,14,0'},
                 'South': {'min': '-1,-14,0',
                           'max': '1,-12,0'},
                 'West': {'min': '-14,-1,0',
                          'max': '-12,1,0'},
                 'East': {'min': '12,-1,0',
                          'max': '14,1,0'}}
    fault_modules = [10, 5, 3, 2]
    fault_timesteps = [0, 100, 500, 800]
    # os.chdir('/Users/lucianofranchin/Documents/Github_repos/argos3-examples/')
    # subprocess.run('ls')
    # for position in positions.keys():
    # open xml file
    xml_file = xml.etree.ElementTree.parse('/Users/lucianofranchin/Documents/Github_repos/'
                                           'argos3-examples/experiments/flocking.argos')
    root = xml_file.getroot()

    for element in xml_file.iter():
        if element.tag == 'position':
            element.attrib['min'] = positions['North']['min']
            element.attrib['max'] = positions['North']['max']
        if element.tag == 'visualization':
            root.remove(element)

    xml_file.write('/Users/lucianofranchin/Documents/Github_repos/'
                   'argos3-examples/experiments/flocking_new.argos')
