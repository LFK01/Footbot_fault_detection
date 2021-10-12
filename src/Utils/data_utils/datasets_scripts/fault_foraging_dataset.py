import os
import subprocess
import random
from xml.etree import ElementTree

from src.Utils.Parser import Parser

from nominal_foraging_dataset import modify_foraging_xml_file, compute_parameters_and_edit_files


def create_fault_foraging_dataset():
    # open xml file
    xml_file = ElementTree.parse('/Users/lucianofranchin/Documents/Github_repos/'
                                 'argos3-examples/experiments/foraging.argos')
    root = xml_file.getroot()

    random.seed(Parser.read_seed())

    # variables to manage the simulation execution
    single_bot_fault_repetitions = 2
    # integer numbers to compute the percentage of robots that are non-nominal
    fault_modules = [10, 5, 3]
    fault_timesteps = [0, 1000, 4000]

    # variables to modify simulation parameters
    arena_dimensions = [5, 6, 7, 8]
    initial_arena_size = arena_dimensions[0] ** 2
    initial_bot_number = 20
    initial_minimum_resting_time = 50
    initial_minimum_unsuccessful_explore_time = 600
    initial_minimum_search_for_place_in_nest_time = 50
    initial_items_number = 15
    initial_light_intensity = 3.0

    print('Doing Fault')
    for x_length in arena_dimensions:
        for y_length in arena_dimensions:
            print('Doing arena {}x{}'.format(x_length, y_length))
            current_bot_number, current_minimum_resting_time, current_minimum_unsuccessful_explore_time, \
                current_minimum_search_for_place_in_nest_time, current_items_number, \
                current_light_intensity = compute_parameters_and_edit_files(
                    par_x_length=x_length,
                    par_y_length=y_length,
                    par_initial_arena_size=initial_arena_size,
                    par_initial_bot_number=initial_bot_number,
                    par_initial_minimum_resting_time=initial_minimum_resting_time,
                    par_initial_minimum_unsuccessful_explore_time=initial_minimum_unsuccessful_explore_time,
                    par_initial_minimum_search_for_place_in_nest_time=
                    initial_minimum_search_for_place_in_nest_time,
                    par_initial_items_number=initial_items_number,
                    par_initial_light_intensity=initial_light_intensity
                )

            for fault_timestep in fault_timesteps:
                print('Doing fault timestep: ' + str(fault_timestep))
                seeds = random.sample(range(0, 1000), single_bot_fault_repetitions)
                for i in range(single_bot_fault_repetitions):
                    print('Doing rep: {} out of {}'.format(i, single_bot_fault_repetitions))
                    modify_foraging_xml_file(par_element_tree=xml_file,
                                             par_root=root,
                                             par_x_length=x_length,
                                             par_y_length=y_length,
                                             par_current_bot_number=current_bot_number,
                                             par_current_minimum_resting_time=current_minimum_resting_time,
                                             par_current_minimum_unsuccessful_explore_time=
                                             current_minimum_unsuccessful_explore_time,
                                             par_current_minimum_search_for_place_in_nest_time=
                                             current_minimum_search_for_place_in_nest_time,
                                             par_light_intensity=current_light_intensity,
                                             par_current_items_number=current_items_number,
                                             par_random_seed=seeds[i])

                    os.chdir('/Users/lucianofranchin/Documents/Github_repos/argos3-examples/')
                    s_proc = subprocess.Popen('argos3 -c experiments/foraging_execution.argos',
                                              stdin=subprocess.PIPE,
                                              shell=True,
                                              text=True)
                    # Nominal Experiment?
                    module_offset = random.randint(0, 100)
                    print('N\n'
                          + str(current_bot_number) + '\n'
                          + str(module_offset) + '\n'
                          + str(fault_timestep) + '\n')
                    s_proc.communicate(input='N\n'
                                             + str(current_bot_number) + '\n'
                                             + str(module_offset) + '\n'
                                             + str(fault_timestep) + '\n')

    seeds = random.sample(range(1000000), len(arena_dimensions)**2 * len(fault_modules) * len(fault_timesteps))
    counter = 0
    for x_length in arena_dimensions:
        for y_length in arena_dimensions:
            current_bot_number, current_minimum_resting_time, current_minimum_unsuccessful_explore_time, \
                current_minimum_search_for_place_in_nest_time, current_items_number, \
                current_light_intensity = compute_parameters_and_edit_files(
                    par_x_length=x_length,
                    par_y_length=y_length,
                    par_initial_arena_size=initial_arena_size,
                    par_initial_bot_number=initial_bot_number,
                    par_initial_minimum_resting_time=initial_minimum_resting_time,
                    par_initial_minimum_unsuccessful_explore_time=initial_minimum_unsuccessful_explore_time,
                    par_initial_minimum_search_for_place_in_nest_time=
                    initial_minimum_search_for_place_in_nest_time,
                    par_initial_items_number=initial_items_number,
                    par_initial_light_intensity=initial_light_intensity
                )
            for fault_module in fault_modules:
                for fault_timestep in fault_timesteps:
                    modify_foraging_xml_file(par_element_tree=xml_file,
                                             par_root=root,
                                             par_x_length=x_length,
                                             par_y_length=y_length,
                                             par_current_bot_number=current_bot_number,
                                             par_current_minimum_resting_time=current_minimum_resting_time,
                                             par_current_minimum_unsuccessful_explore_time=
                                             current_minimum_unsuccessful_explore_time,
                                             par_current_minimum_search_for_place_in_nest_time=
                                             current_minimum_search_for_place_in_nest_time,
                                             par_light_intensity=current_light_intensity,
                                             par_current_items_number=current_items_number,
                                             par_random_seed=seeds[counter])

                    os.chdir('/Users/lucianofranchin/Documents/Github_repos/argos3-examples/')
                    s_proc = subprocess.Popen('argos3 -c experiments/foraging_execution.argos',
                                              stdin=subprocess.PIPE,
                                              shell=True,
                                              text=True)
                    # Nominal Experiment?
                    module_offset = random.randint(0, current_bot_number)
                    s_proc.communicate(input='N\n'
                                             + str(fault_module) + '\n'
                                             + str(module_offset) + '\n'
                                             + str(fault_timestep) + '\n')
                    counter += 1


if __name__ == '__main__':
    create_fault_foraging_dataset()
