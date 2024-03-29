import random
from os import chdir, sep
from os.path import join
import subprocess
from xml.etree import ElementTree

from src.Utils.Parser import Parser
from src.Utils.data_utils.datasets_scripts.nominal_warehouse_dataset import compute_parameters


def modify_fault_warehouse_xsett_xml_file(par_element_tree: ElementTree.ElementTree,
                                          sett_file_id: str,
                                          par_simulation_duration: int,
                                          par_seed: int):
    for element_iterator in par_element_tree.iter():
        if element_iterator.tag == 'Name':
            element_iterator.text = 'SimpleItem-Fill-' + str(par_simulation_duration)
        if element_iterator.tag == 'SimulationDuration':
            element_iterator.text = str(par_simulation_duration)
        if element_iterator.tag == 'Seed':
            element_iterator.text = str(par_seed)

    path = join('C:', sep, 'Users', 'Luciano', 'source', 'repos', 'LFK01', 'Configuration_files',
                'Execution_fault_' + sett_file_id + '_settings.xsett')
    par_element_tree.write(path)


def modify_fault_warehouse_xlayo_xml_file(par_element_tree: ElementTree.ElementTree,
                                          layo_file_id: str,
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

    path = join('C:', sep, 'Users', 'Luciano', 'source', 'repos', 'LFK01', 'Configuration_files',
                'Execution_fault_' + layo_file_id + '_layout.xlayo')
    par_element_tree.write(path)


def modify_fault_warehouse_xconf_xml_file(par_element_tree: ElementTree.ElementTree,
                                          conf_file_id: str):
    path = join('C:', sep, 'Users', 'Luciano', 'source', 'repos', 'LFK01', 'Configuration_files',
                'Execution_fault_' + conf_file_id + '_configuration.xconf')
    par_element_tree.write(path)


def create_fault_warehouse_csv_logs(file_id: str,
                                    idx_begin: int,
                                    idx_end: int):
    # open xml file
    path = join('C:', sep, 'Users', 'Luciano', 'source', 'repos', 'LFK01', 'Configuration_files',
                'SimpleItem-Fill-200-200-8000.xsett')
    xsett_xml_file = ElementTree.parse(path)

    path = join('C:', sep, 'Users', 'Luciano', 'source', 'repos', 'LFK01', 'Configuration_files',
                '1-2-2-15-0.85.xlayo')
    xlayo_xml_file = ElementTree.parse(path)

    path = join('C:', sep, 'Users', 'Luciano', 'source', 'repos', 'LFK01', 'Configuration_files',
                'PPFAR-TABalanced-SAActivateAll-ISEmptiest-PSNearest-RPDummy-OBPodMatching-RBSamePod-MMNoChange.xconf')
    xconf_xml_file = ElementTree.parse(path)

    xsett_filename = 'Execution_fault_' + file_id + '_settings.xsett'
    xlayo_filename = 'Execution_fault_' + file_id + '_layout.xlayo'
    xconf_filename = 'Execution_fault_' + file_id + '_configuration.xconf'
    random.seed(Parser.read_seed())

    # variables to manage the simulation execution
    single_bot_fault_repetitions = 2

    # variables to modify simulation parameters
    simulation_duration = 80000
    aisles_number = [5, 6, 7, 8]
    initial_aisles_number = aisles_number[0] ** 2
    initial_bot_number = 15
    fault_modules = [20, 10, 5, 3]
    speed_operators = [2, 3]

    print('Doing Fault')
    for horizontal_aisles_number in aisles_number[idx_begin:idx_end]:
        for vertical_aisles_number in aisles_number:
            print('Doing Arena {}x{}'.format(horizontal_aisles_number, vertical_aisles_number))

            current_bot_number, current_pick_number, current_replenishment_number = compute_parameters(
                par_horizontal_aisles=horizontal_aisles_number,
                par_vertical_aisles=vertical_aisles_number,
                par_initial_aisles_number=initial_aisles_number,
                par_initial_bot_number=initial_bot_number,
            )

            seeds = random.sample(range(0, 1000), single_bot_fault_repetitions)

            for i in range(single_bot_fault_repetitions):
                for speed_operator in speed_operators:
                    print('Doing single bot fault rep: {} out of {}'.format(i + 1, single_bot_fault_repetitions))
                    modify_fault_warehouse_xlayo_xml_file(par_element_tree=xlayo_xml_file,
                                                          layo_file_id=file_id,
                                                          par_vertical_aisles=vertical_aisles_number,
                                                          par_horizontal_aisles=horizontal_aisles_number,
                                                          par_bot_count=current_bot_number,
                                                          par_pick_stations=current_pick_number,
                                                          par_replenishment_stations=current_replenishment_number)

                    modify_fault_warehouse_xsett_xml_file(par_element_tree=xsett_xml_file,
                                                          sett_file_id=file_id,
                                                          par_simulation_duration=simulation_duration,
                                                          par_seed=seeds[i])
                    modify_fault_warehouse_xconf_xml_file(par_element_tree=xconf_xml_file,
                                                          conf_file_id=file_id)
                    path = join('C:', sep, 'Users', 'Luciano', 'source', 'repos', 'LFK01', 'Configuration_files')
                    chdir(path)
                    exe_path = join('..', 'RAWSim-O', 'RAWSimO.CLI', 'bin', 'x64',
                                    'Debug', 'RAWSimO.CLI.exe')
                    command_line = exe_path + ' ' \
                                            + xlayo_filename + ' ' \
                                            + xsett_filename + ' ' \
                                            + xconf_filename + ' ' \
                                            + 'StatsDirFault' + file_id + '  1'
                    subproc = subprocess.Popen(command_line,
                                               stdin=subprocess.PIPE,
                                               shell=True,
                                               text=True)
                    module_offset = random.randint(0, 100)
                    subproc.communicate(input='N\n'
                                              '{}\n'.format(horizontal_aisles_number) +
                                              '{}\n'.format(vertical_aisles_number) +
                                              'N\n' +
                                              '{}\n'.format(speed_operator) +
                                              '{}\n'.format(current_bot_number) +
                                              '{}\n'.format(module_offset))

            seeds = random.sample(range(0, 1000), len(fault_modules)*len(speed_operators))
            counter = 0
            for fault_module in fault_modules:
                for speed_operator in speed_operators:
                    print('Doing fault module: {} Speed operator: {}'.format(fault_module, speed_operator))
                    modify_fault_warehouse_xlayo_xml_file(par_element_tree=xlayo_xml_file,
                                                          layo_file_id=file_id,
                                                          par_vertical_aisles=vertical_aisles_number,
                                                          par_horizontal_aisles=horizontal_aisles_number,
                                                          par_bot_count=current_bot_number,
                                                          par_pick_stations=current_pick_number,
                                                          par_replenishment_stations=current_replenishment_number)

                    modify_fault_warehouse_xsett_xml_file(par_element_tree=xsett_xml_file,
                                                          sett_file_id=file_id,
                                                          par_simulation_duration=simulation_duration,
                                                          par_seed=seeds[counter])
                    modify_fault_warehouse_xconf_xml_file(par_element_tree=xconf_xml_file,
                                                          conf_file_id=file_id)
                    path = join('C:', sep, 'Users', 'Luciano', 'source', 'repos', 'LFK01', 'Configuration_files')
                    chdir(path)
                    exe_path = join('..', 'RAWSim-O', 'RAWSimO.CLI', 'bin', 'x64',
                                    'Debug', 'RAWSimO.CLI.exe')
                    command_line = exe_path + ' ' \
                                            + xlayo_filename + ' ' \
                                            + xsett_filename + ' ' \
                                            + xconf_filename + ' ' \
                                            + 'StatsDirFault' + file_id + ' 1'
                    subproc = subprocess.Popen(command_line,
                                               stdin=subprocess.PIPE,
                                               shell=True,
                                               text=True)
                    # nominal experiment?
                    module_offset = random.randint(0, 100)
                    subproc.communicate(input='N\n'
                                              '{}\n'.format(horizontal_aisles_number) +
                                              '{}\n'.format(vertical_aisles_number) +
                                              'N\n' +
                                              '{}\n'.format(speed_operator) +
                                              '{}\n'.format(fault_module) +
                                              '{}\n'.format(module_offset))
                    counter += 1


if __name__ == '__main__':
    create_fault_warehouse_csv_logs(file_id='p0',
                                    idx_begin=0,
                                    idx_end=1)
