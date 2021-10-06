import os
import random
import subprocess
import xml.etree.ElementTree
from xml.etree.ElementTree import ElementTree
import numpy as np

from src.Utils.Parser import Parser


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


def modify_xml_file(par_element_tree: ElementTree,
                    par_positions: dict,
                    par_position: str,
                    par_bot_quantity: int,
                    par_random_seed: int):
    for element in par_element_tree.iter():
        if element.tag == 'position':
            element.attrib['min'] = par_positions[par_position]['min']
            element.attrib['max'] = par_positions[par_position]['max']
        if element.tag == 'entity':
            element.attrib['quantity'] = str(par_bot_quantity)
        if element.tag == 'experiment':
            element.attrib['length'] = '800'
        if element.tag == 'experiment':
            element.attrib['random_seed'] = str(par_random_seed)
        if element.tag == 'visualization':
            root.remove(element)

    par_element_tree.write('/Users/lucianofranchin/Documents/Github_repos/'
                           'argos3-examples/experiments/flocking_execution.argos')


def modify_cpp_file(par_quantity: int,
                    par_position: str):
    filename = '/Users/lucianofranchin/Documents/Github_repos/argos3-examples' \
               '/controllers/footbot_flocking/footbot_flocking.cpp'
    file = open(filename, 'r')
    new_file_text = ''
    for file_line in file.readlines():
        if 'myfile.open("log_files_directory/' in file_line:
            if 'nominal' not in file_line:
                file_line = '        myfile.open("log_files_directory/flocking_{}_{}_" +\n' \
                    .format(par_quantity, par_position)
            elif 'nominal' in file_line:
                file_line = '        myfile.open("log_files_directory/flocking_{}_{}_nominal_gain_1000_" +\n' \
                    .format(par_quantity, par_position)
        new_file_text = new_file_text + file_line

    file.close()

    output_file = open('/Users/lucianofranchin/Documents/Github_repos/argos3-examples'
                       '/controllers/footbot_flocking/footbot_flocking.cpp', 'w')
    output_file.write(new_file_text)
    output_file.close()

    os.chdir('/Users/lucianofranchin/Documents/Github_repos/argos3-examples/build/')
    subprocess.call('make', shell=True)


def execute_command_script(par_xml_file: ElementTree,
                           par_positions: dict,
                           par_position: str,
                           par_experiment_bot_quantity: int,
                           par_random_seed: int,
                           par_fault_timesteps: list[int],
                           par_fault_module: int):
    modify_xml_file(par_element_tree=par_xml_file,
                    par_positions=par_positions,
                    par_position=par_position,
                    par_bot_quantity=par_experiment_bot_quantity,
                    par_random_seed=par_random_seed)
    os.chdir('/Users/lucianofranchin/Documents/Github_repos/argos3-examples/')
    for fault_timestep in par_fault_timesteps:
        s_proc = subprocess.Popen('argos3 -c experiments/flocking_execution.argos',
                                  stdin=subprocess.PIPE,
                                  shell=True,
                                  text=True)
        # Nominal Experiment?
        module_offset = random.randint(0, par_experiment_bot_quantity)
        s_proc.communicate(input='N\n'
                                 + str(par_fault_module) + '\n'
                                 + str(module_offset) + '\n'
                                 + str(fault_timestep) + '\n')


if __name__ == "__main__":
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
    nominal_exp_repetitions = 10
    single_bot_fault_repetitions = 4
    fault_experiments = [True, False]
    fault_modules = [10, 5, 3]
    fault_timesteps: list[int] = [0, 500, 1500, 4000]

    # open xml file
    xml_file = xml.etree.ElementTree.parse('/Users/lucianofranchin/Documents/Github_repos/'
                                           'argos3-examples/experiments/flocking.argos')
    root = xml_file.getroot()

    # cycle between nominal experiments and fault experiments
    for inject_fault in fault_experiments:
        if not inject_fault:
            # nominal experiments should be repeated more times with different seed in order to depict
            # a realistic scenario
            random_seeds: list[int] = random.sample(range(0, 1000), nominal_exp_repetitions)
            for i in range(nominal_exp_repetitions):
                for position in positions_dict.keys():
                    modify_cpp_file(par_quantity=experiment_bot_quantity,
                                    par_position=position)
                    modify_xml_file(par_element_tree=xml_file,
                                    par_positions=positions_dict,
                                    par_position=position,
                                    par_bot_quantity=experiment_bot_quantity,
                                    par_random_seed=random_seeds[i])

                    os.chdir('/Users/lucianofranchin/Documents/Github_repos/argos3-examples/')
                    subproc = subprocess.Popen('argos3 -c experiments/flocking_execution.argos',
                                               stdin=subprocess.PIPE,
                                               shell=True)
                    # nominal experiment?
                    subproc.communicate(input=b'Y')
        else:
            # fault experiments, single bot faults are repeated more times since they are more likely to appear
            random_seeds: list[int] = random.sample(range(0, 1000), single_bot_fault_repetitions)
            for i in range(single_bot_fault_repetitions):
                random_seed = random.randint(0, 100)
                for position in positions_dict.keys():
                    modify_cpp_file(par_quantity=experiment_bot_quantity,
                                    par_position=position)
                    execute_command_script(par_xml_file=xml_file,
                                           par_positions=positions_dict,
                                           par_position=position,
                                           par_random_seed=random_seeds[i],
                                           par_fault_timesteps=fault_timesteps,
                                           par_experiment_bot_quantity=experiment_bot_quantity,
                                           par_fault_module=experiment_bot_quantity)

            for fault_module in fault_modules:
                random_seed = random.randint(0, 100)
                for position in positions_dict:
                    modify_cpp_file(par_quantity=experiment_bot_quantity,
                                    par_position=position)
                    execute_command_script(par_xml_file=xml_file,
                                           par_positions=positions_dict,
                                           par_position=position,
                                           par_random_seed=random_seed,
                                           par_fault_timesteps=fault_timesteps,
                                           par_experiment_bot_quantity=experiment_bot_quantity,
                                           par_fault_module=fault_module)
