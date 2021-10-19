import os
import random
import subprocess
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

from src.Utils.Parser import Parser


def create_fault_flocking_csv_logs():
    random.seed(Parser.read_seed())
    positions_dict = {'North': {'min': '-1,12,0',
                                'max': '1,14,0'},
                      'South': {'min': '-1,-14,0',
                                'max': '1,-12,0'},
                      'West': {'min': '-14,-1,0',
                               'max': '-12,1,0'},
                      'East': {'min': '12,-1,0',
                               'max': '14,1,0'}}

    experiment_bot_quantity = 15
    single_bot_fault_repetitions = 4
    # integer numbers to compute the percentage of robots that are non-nominal
    fault_modules = [10, 5, 3]
    fault_timesteps: list[int] = [0, 500, 1500, 4000]

    # open xml file
    xml_file_element = ElementTree.parse('/Users/lucianofranchin/Documents/Github_repos/'
                                         'argos3-examples/experiments/flocking.argos')
    xml_root = xml_file_element.getroot()
    # fault experiments, single bot faults are repeated more times since they are more likely to appear
    random_seeds: list[int] = random.sample(range(0, 1000), single_bot_fault_repetitions)
    for i in range(single_bot_fault_repetitions):
        for position in positions_dict.keys():
            execute_fault_flocking_simulation_command(par_xml_file=xml_file_element,
                                                      par_root=xml_root,
                                                      par_positions=positions_dict,
                                                      par_position=position,
                                                      par_random_seed=random_seeds[i],
                                                      par_fault_timesteps=fault_timesteps,
                                                      par_experiment_bot_quantity=experiment_bot_quantity,
                                                      par_fault_module=experiment_bot_quantity)
    random_seeds = random.sample(range(0, 10000), len(fault_modules)*len(positions_dict))
    counter = 0
    for fault_module in fault_modules:
        for position in positions_dict:
            execute_fault_flocking_simulation_command(par_xml_file=xml_file_element,
                                                      par_root=xml_root,
                                                      par_positions=positions_dict,
                                                      par_position=position,
                                                      par_random_seed=random_seeds[counter],
                                                      par_fault_timesteps=fault_timesteps,
                                                      par_experiment_bot_quantity=experiment_bot_quantity,
                                                      par_fault_module=fault_module)
            counter += 1


def modify_fault_flocking_xml_file(par_element_tree: ElementTree.ElementTree,
                                   par_root: Element,
                                   par_positions: dict,
                                   par_position: str,
                                   par_bot_quantity: int,
                                   par_random_seed: int):
    for element_iterator in par_element_tree.iter():
        if element_iterator.tag == 'position':
            element_iterator.attrib['min'] = par_positions[par_position]['min']
            element_iterator.attrib['max'] = par_positions[par_position]['max']
        if element_iterator.tag == 'entity':
            element_iterator.attrib['quantity'] = str(par_bot_quantity)
        if element_iterator.tag == 'experiment':
            element_iterator.attrib['length'] = '800'
        if element_iterator.tag == 'experiment':
            element_iterator.attrib['random_seed'] = str(par_random_seed)
        if element_iterator.tag == 'visualization':
            par_root.remove(element_iterator)

    par_element_tree.write('/Users/lucianofranchin/Documents/Github_repos/'
                           'argos3-examples/experiments/flocking_fault_execution.argos')


def execute_fault_flocking_simulation_command(par_xml_file: ElementTree,
                                              par_root: Element,
                                              par_positions: dict,
                                              par_position: str,
                                              par_experiment_bot_quantity: int,
                                              par_random_seed: int,
                                              par_fault_timesteps: list[int],
                                              par_fault_module: int):
    modify_fault_flocking_xml_file(par_element_tree=par_xml_file,
                                   par_root=par_root,
                                   par_positions=par_positions,
                                   par_position=par_position,
                                   par_bot_quantity=par_experiment_bot_quantity,
                                   par_random_seed=par_random_seed)
    os.chdir('/Users/lucianofranchin/Documents/Github_repos/argos3-examples/')
    for fault_timestep in par_fault_timesteps:
        s_proc = subprocess.Popen('argos3 -c experiments/flocking_fault_execution.argos',
                                  stdin=subprocess.PIPE,
                                  shell=True,
                                  text=True)
        # Nominal Experiment?
        module_offset = random.randint(0, par_experiment_bot_quantity)
        s_proc.communicate(input='N\n'
                                 + par_position + '\n'
                                 + str(par_experiment_bot_quantity) + '\n'
                                 + str(par_fault_module) + '\n'
                                 + str(module_offset) + '\n'
                                 + str(fault_timestep) + '\n')


if __name__ == "__main__":
    create_fault_flocking_csv_logs()
