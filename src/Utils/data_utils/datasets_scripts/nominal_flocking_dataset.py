import os
import random
import subprocess
from xml.etree import ElementTree
from xml.etree.ElementTree import Element
from typing import List, Dict

from src.Utils.Parser import Parser
from src.data_writing import build_swarm_no_foraging_stats, build_foraging_swarm, build_feature_set_datasets
from src.model_training import execute_training_feature_set_datasets


def create_nominal_flocking_csv_logs():
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

    # open xml file
    xml_file_element = ElementTree.parse('/Users/lucianofranchin/Documents/Github_repos/'
                                         'argos3-examples/experiments/flocking.argos')
    xml_root = xml_file_element.getroot()

    # nominal experiments should be repeated more times with different seed in order to depict
    # a realistic scenario
    random_seeds: List[int] = random.sample(range(0, 1000), nominal_exp_repetitions)
    for i in range(nominal_exp_repetitions):
        for position in positions_dict.keys():
            modify_nominal_flocking_xml_file(par_element_tree=xml_file_element,
                                             par_root=xml_root,
                                             par_positions=positions_dict,
                                             par_position=position,
                                             par_bot_quantity=experiment_bot_quantity,
                                             par_random_seed=random_seeds[i])

            os.chdir('/Users/lucianofranchin/Documents/Github_repos/argos3-examples/')
            subproc = subprocess.Popen('argos3 -c experiments/flocking_nominal_execution.argos',
                                       stdin=subprocess.PIPE,
                                       shell=True,
                                       text=True)
            # nominal experiment?
            subproc.communicate(input='Y\n'
                                      + position + '\n'
                                      + str(experiment_bot_quantity) + '\n')


def modify_nominal_flocking_xml_file(par_element_tree: ElementTree.ElementTree,
                                     par_root: Element,
                                     par_positions: Dict,
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
                           'argos3-examples/experiments/flocking_nominal_execution.argos')


if __name__ == "__main__":
    for feature_set_name in Parser.return_feature_sets_dict().keys():
        main_task_name = 'FLOC'
        build_swarm_no_foraging_stats(task_name=main_task_name,
                                        do_swarm_saving=False,
                                      feature_set_name=feature_set_name,
                                      feature_set_feature_list=Parser.
                                      read_features_set(feature_set_name=feature_set_name),
                                      experiments_number_down_sampling=50)
        main_task_name = 'WARE'
        build_swarm_no_foraging_stats(task_name=main_task_name,
                                      do_swarm_saving=False,
                                      feature_set_name=feature_set_name,
                                      feature_set_feature_list=Parser.
                                      read_features_set(feature_set_name=feature_set_name),
                                      experiments_number_down_sampling=40)
        build_foraging_swarm(feature_set_name=feature_set_name,
                             do_swarm_saving=False,
                             feature_set_feature_list=Parser.
                             read_features_set(feature_set_name=feature_set_name),
                             down_sampling=80)
