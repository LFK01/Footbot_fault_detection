import os
import subprocess
import random
from xml.etree import ElementTree

from src.Utils.Parser import Parser

from nominal_homing_dataset import modify_homing_xml_file, compute_parameters


def create_fault_homing_csv_logs(file_id: str,
                                 idx_begin: int,
                                 idx_end: int):
    # open xml file
    xml_file = ElementTree.parse('/Users/lucianofranchin/Documents/Github_repos/'
                                 'argos3-examples/experiments/footbot_homing.argos')
    root = xml_file.getroot()

    random.seed(Parser.read_seed())

    # variables to manage the simulation execution
    single_bot_fault_repetitions = 2
    # integer numbers to compute the percentage of robots that are non-nominal
    fault_modules = [20, 10, 5, 3]
    fault_timesteps = [0, 500, 1000, 2500]

    # variables to modify simulation parameters
    arena_dimensions = [10, 12, 14, 18]
    initial_arena_size = arena_dimensions[0] ** 2
    initial_bot_number = 36
    initial_light_intensity = 6.0
    initial_exp_length = 800

    print('Doing Fault')
    seeds = random.sample(range(1000000), (len(arena_dimensions) ** 2)
                          * len(fault_modules)
                          * len(fault_timesteps)
                          * single_bot_fault_repetitions)
    counter = 0
    for x_length in arena_dimensions[idx_begin:idx_end]:
        for y_length in arena_dimensions:
            print('Doing arena {}x{}'.format(x_length, y_length))
            current_bot_number, current_exp_length, current_light_intensity = compute_parameters(
                par_x_length=x_length,
                par_y_length=y_length,
                par_initial_arena_size=initial_arena_size,
                par_initial_bot_number=initial_bot_number,
                par_initial_exp_length=initial_exp_length,
                par_initial_light_intensity=initial_light_intensity)

            for fault_timestep in fault_timesteps:
                print('Doing fault timestep: ' + str(fault_timestep))
                for i in range(single_bot_fault_repetitions):
                    print('Doing rep: {} out of {}'.format(i+1, single_bot_fault_repetitions))
                    modify_homing_xml_file(file_id=file_id,
                                           par_element_tree=xml_file,
                                           par_root=root,
                                           par_x_length=x_length,
                                           par_y_length=y_length,
                                           par_current_bot_number=current_bot_number,
                                           par_exp_length=current_exp_length,
                                           par_light_intensity=current_light_intensity,
                                           par_random_seed=seeds[i])

                    os.chdir('/Users/lucianofranchin/Documents/Github_repos/argos3-examples/')
                    s_proc = subprocess.Popen('argos3 -c experiments/homing_execution' + file_id + '.argos',
                                              stdin=subprocess.PIPE,
                                              shell=True,
                                              text=True)
                    # Nominal Experiment?
                    module_offset = random.randint(0, 100)
                    s_proc.communicate(input='N\n'
                                             + '{}\n'.format(x_length)
                                             + '{}\n'.format(y_length)
                                             + '{}\n'.format(current_bot_number)
                                             + '{}\n'.format(current_bot_number)
                                             + '{}\n'.format(module_offset)
                                             + '{}\n'.format(fault_timestep))
                    counter += 1

    seeds = random.sample(range(1000000), (len(arena_dimensions) ** 2) * len(fault_modules) * len(fault_timesteps))
    counter = 0
    for x_length in arena_dimensions[idx_begin:idx_end]:
        for y_length in arena_dimensions:
            current_bot_number, current_exp_length, current_light_intensity = compute_parameters(
                par_x_length=x_length,
                par_y_length=y_length,
                par_initial_arena_size=initial_arena_size,
                par_initial_bot_number=initial_bot_number,
                par_initial_exp_length=initial_exp_length,
                par_initial_light_intensity=initial_light_intensity
            )
            for fault_module in fault_modules:
                for fault_timestep in fault_timesteps:
                    modify_homing_xml_file(file_id=file_id,
                                           par_element_tree=xml_file,
                                           par_root=root,
                                           par_x_length=x_length,
                                           par_y_length=y_length,
                                           par_current_bot_number=current_bot_number,
                                           par_exp_length=current_exp_length,
                                           par_light_intensity=current_light_intensity,
                                           par_random_seed=seeds[counter])

                    os.chdir('/Users/lucianofranchin/Documents/Github_repos/argos3-examples/')
                    s_proc = subprocess.Popen('argos3 -c experiments/homing_execution' + file_id + '.argos',
                                              stdin=subprocess.PIPE,
                                              shell=True,
                                              text=True)
                    # Nominal Experiment?
                    module_offset = random.randint(0, current_bot_number)
                    s_proc.communicate(input='N\n'
                                             + '{}\n'.format(x_length)
                                             + '{}\n'.format(y_length)
                                             + '{}\n'.format(current_bot_number)
                                             + '{}\n'.format(fault_module)
                                             + '{}\n'.format(module_offset)
                                             + '{}\n'.format(fault_timestep))
                    counter += 1


if __name__ == '__main__':
    create_fault_homing_csv_logs(file_id='p0',
                                 idx_begin=0,
                                 idx_end=1)
