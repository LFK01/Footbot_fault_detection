import os
import random
import subprocess
from subprocess import Popen, PIPE, STDOUT
import xml.etree.ElementTree
from xml.etree.ElementTree import ElementTree
import numpy as np


class GeneralDataset:
    """
    Class to store a generic dataset split in train and test
    """

    def __init__(self,
                 bot_identifier: int,
                 train_dataset: np.ndarray,
                 target_train_dataset: np.ndarray,
                 test_dataset: np.ndarray,
                 target_test_dataset: np.ndarray):
        self.bot_identifier: int = bot_identifier
        self.train_dataset: np.ndarray = train_dataset
        self.target_train_dataset: np.ndarray = target_train_dataset
        self.test_dataset: np.ndarray = test_dataset
        self.target_test_dataset: np.ndarray = target_test_dataset


def modify_xml_file(element_tree: ElementTree):
    for element in element_tree.iter():
        if element.tag == 'position':
            element.attrib['min'] = positions[position]['min']
            element.attrib['max'] = positions[position]['max']
        if element.tag == 'entity':
            element.attrib['quantity'] = '15'
        if element.tag == 'experiment':
            element.attrib['length'] = '800'
        if element.tag == 'visualization':
            root.remove(element)


if __name__ == "__main__":
    """
    script to generate more simulations
    """
    positions = {'North': {'min': '-1,12,0',
                           'max': '1,14,0'},
                 'South': {'min': '-1,-14,0',
                           'max': '1,-12,0'},
                 'West': {'min': '-14,-1,0',
                          'max': '-12,1,0'},
                 'East': {'min': '12,-1,0',
                          'max': '14,1,0'}}

    fault_experiments = [False, True]
    fault_modules = [10, 5, 3, 2]
    fault_timesteps = [0, 500, 1500, 4000]

    # open xml file
    xml_file = xml.etree.ElementTree.parse('/Users/lucianofranchin/Documents/Github_repos/'
                                           'argos3-examples/experiments/flocking.argos')
    root = xml_file.getroot()

    for inject_fault in fault_experiments:
        if inject_fault:
            for i in range(4):
                for position in positions.keys():
                    modify_xml_file(element_tree=xml_file)

    for position in positions.keys():
        modify_xml_file(element_tree=xml_file)

        xml_file.write('/Users/lucianofranchin/Documents/Github_repos/'
                       'argos3-examples/experiments/flocking_execution.argos')

        for fault_module in fault_modules:
            for fault_timestep in fault_timesteps:
                module_offset = random.randint(0, 10)
                os.chdir('/Users/lucianofranchin/Documents/Github_repos/argos3-examples/')
                subprocess.run('argos3 -c experiments/flocking_execution.argos', shell=True)
                p = Popen(['myapp'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
                stdout_data = p.communicate(input='N')[0]
