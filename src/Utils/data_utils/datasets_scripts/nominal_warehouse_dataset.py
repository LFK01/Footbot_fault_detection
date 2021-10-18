import os
import random
from os import chdir
from os.path import join
import subprocess
from xml.etree import ElementTree
from xml.etree.ElementTree import Element
from lxml import etree

from src.Utils.Parser import Parser
from src.data_writing import build_swarm_no_foraging_stats


def modify_warehouse_xsett_xml_file(par_element_tree: ElementTree.ElementTree,
                                    par_simulation_duration: int,
                                    par_seed: int):
    for element_iterator in par_element_tree.iter():
        if element_iterator.tag == 'Name':
            element_iterator.text = 'SimpleItem-Fill-' + str(par_simulation_duration)
        if element_iterator.tag == 'SimulationDuration':
            element_iterator.text = str(par_simulation_duration)
        if element_iterator.tag == 'Seed':
            element_iterator.text = str(par_seed)

    path = join('C:', os.sep, 'Users', 'Luciano', 'source', 'RawSim-O_conf_files', 'Execution_settings.xsett')
    par_element_tree.write(path)


def modify_warehouse_xlayo_xml_file(par_element_tree: ElementTree.ElementTree,
                                    par_bot_count: int,
                                    par_vertical_aisles: int,
                                    par_horizontal_aisles: int,
                                    par_pick_stations: int,
                                    par_replenishment_stations: int):
    for element_iterator in par_element_tree.iter():
        if element_iterator.tag == 'BotCount':
            element_iterator.text = str(par_bot_count)
        if element_iterator.tag == 'NrVerticalAisles':
            element_iterator.text = str(par_vertical_aisles)
        if element_iterator.tag == 'NrHorizontalAisles':
            element_iterator.text = str(par_horizontal_aisles)
        if element_iterator.tag == 'NPickStationEast':
            element_iterator.text = str(par_pick_stations)
        if element_iterator.tag == 'NReplenishmentStationWest':
            element_iterator.text = str(par_replenishment_stations)

    path = join('C:', os.sep, 'Users', 'Luciano', 'source', 'RawSim-O_conf_files', 'Execution_layout.xlayo')
    par_element_tree.write(path)


def modify_warehouse_xconf_xml_file(par_element_tree: ElementTree.ElementTree):
    path = join('C:', os.sep, 'Users', 'Luciano', 'source', 'RawSim-O_conf_files', 'Execution_configuration.xconf')
    par_element_tree.write(path)


def compute_parameters(par_vertical_aisles: int,
                       par_horizontal_aisles: int,
                       par_initial_aisles_number: int,
                       par_initial_bot_number: int,
                       par_initial_pick_number: int,
                       par_initial_replenishment_number: int, ):
    current_aisles_number = par_horizontal_aisles * par_vertical_aisles
    size_increase = current_aisles_number / par_initial_aisles_number
    current_bot_number = int(size_increase * par_initial_bot_number)
    current_pick_number = int(size_increase * par_initial_pick_number)
    current_replenishment_number = int(size_increase * par_initial_replenishment_number)

    return current_bot_number, current_pick_number, current_replenishment_number


def create_nominal_warehouse_dataset():
    # open xml file
    path = join('C:', os.sep, 'Users', 'Luciano', 'source', 'RawSim-O_conf_files', 'SimpleItem-Fill-200-200-4000.xsett')
    xsett_xml_file = ElementTree.parse(path)

    path = join('C:', os.sep, 'Users', 'Luciano', 'source', 'RawSim-O_conf_files', '1-4-4-32-0.85.xlayo')
    xlayo_xml_file = ElementTree.parse(path)

    path = join('C:', os.sep, 'Users', 'Luciano', 'source', 'RawSim-O_conf_files', 'PPFAR-TABalanced-SAActivateAll-ISEmptiest-'
                                                                           'PSNearest-RPDummy-OBPodMatching-RBSamePod-'
                                                                           'MMNoChange.xconf')
    xconf_xml_file = ElementTree.parse(path)

    xsett_filename = 'Execution_settings.xsett'
    xlayo_filename = 'Execution_layout.xlayo'
    xconf_filename = 'Execution_configuration.xconf'
    random.seed(Parser.read_seed())

    # variables to manage the simulation execution
    nominal_exp_repetitions = 4

    # variables to modify simulation parameters
    simulation_duration = 80000
    aisles_number = [5, 6, 7]
    initial_aisles_number = aisles_number[0] ** 2
    initial_bot_number = 32
    initial_pick_number = 4
    initial_replenishment_number = 4

    print('Doing Nominal')
    for horizontal_aisles_number in aisles_number:
        for vertical_aisles_number in aisles_number:
            print('Doing Arena {}x{}'.format(horizontal_aisles_number, vertical_aisles_number))

            current_bot_number, current_pick_number, current_replenishment_number = compute_parameters(
                par_horizontal_aisles=horizontal_aisles_number,
                par_vertical_aisles=vertical_aisles_number,
                par_initial_aisles_number=initial_aisles_number,
                par_initial_bot_number=initial_bot_number,
                par_initial_pick_number=initial_pick_number,
                par_initial_replenishment_number=initial_replenishment_number
            )

            seeds = random.sample(range(0, 1000), nominal_exp_repetitions)

            for i in range(nominal_exp_repetitions):
                print('Doing nominal rep: {} out of {}'.format(i + 1, nominal_exp_repetitions))
                modify_warehouse_xlayo_xml_file(par_element_tree=xlayo_xml_file,
                                                par_vertical_aisles=vertical_aisles_number,
                                                par_horizontal_aisles=horizontal_aisles_number,
                                                par_bot_count=current_bot_number,
                                                par_pick_stations=current_pick_number,
                                                par_replenishment_stations=current_replenishment_number)

                modify_warehouse_xsett_xml_file(par_element_tree=xsett_xml_file,
                                                par_simulation_duration=simulation_duration,
                                                par_seed=seeds[i])
                modify_warehouse_xconf_xml_file(par_element_tree=xconf_xml_file)
                path = join('C:', os.sep, 'Users', 'Luciano', 'source', 'RawSim-O_conf_files')
                chdir(path)
                exe_path = join('..', 'repos', 'RAWSim-O', 'RAWSimO.CLI', 'bin', 'x64',
                                'Debug', 'RAWSimO.CLI.exe')
                command_line = exe_path + ' ' \
                                        + xlayo_filename + ' ' \
                                        + xconf_filename + ' ' \
                                        + xsett_filename + ' ' \
                                        + 'StatsDir1 1'
                subproc = subprocess.Popen(command_line,
                                           stdin=subprocess.PIPE,
                                           shell=True,
                                           text=True)
                # nominal experiment?
                subproc.communicate(input='Y\n'
                                          '{}\n'.format(horizontal_aisles_number) +
                                          '{}\n'.format(vertical_aisles_number))


if __name__ == '__main__':
    create_nominal_warehouse_dataset()
