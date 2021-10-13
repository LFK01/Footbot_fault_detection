import os
import subprocess
import random
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

from src.Utils.Parser import Parser
from src.Utils.data_utils.datasets_scripts.flocking_dataset import compile_repo


def modify_foraging_loop_func_cpp_file(par_x_length: int,
                                       par_y_length: int):
    filename = '/Users/lucianofranchin/Documents/Github_repos/argos3-examples' \
               '/loop_functions/foraging_loop_functions/foraging_loop_functions.cpp'
    file = open(filename, 'r')
    # nest limits definition and items placement area
    new_file_text = ''
    for file_line in file.readlines():
        if 'Real left_border = ' in file_line:
            left_border = -(par_x_length / 2)
            file_line = 'Real left_border = {:5.1f}f;\n'.format(left_border)
        if 'Real right_border = ' in file_line:
            right_border = (par_x_length / 2)
            file_line = 'Real right_border = {:5.1f}f;\n'.format(right_border)
        if 'Real top_border = ' in file_line:
            top_border = -(par_y_length * 0.20)
            file_line = 'Real top_border = {:5.1f}f;\n'.format(top_border)
        if 'Real bottom_border = ' in file_line:
            bottom_border = -(par_y_length / 2)
            file_line = 'Real bottom_border = {:5.1f}f;\n'.format(bottom_border)
        new_file_text = new_file_text + file_line

    file.close()

    output_file = open('/Users/lucianofranchin/Documents/Github_repos/argos3-examples'
                       '/loop_functions/foraging_loop_functions/foraging_loop_functions.cpp', 'w')
    output_file.write(new_file_text)
    output_file.close()


def modify_foraging_xml_file(par_element_tree: ElementTree.ElementTree,
                             par_root: Element,
                             par_x_length: int,
                             par_y_length: int,
                             par_current_bot_number: int,
                             par_current_minimum_resting_time: int,
                             par_current_minimum_unsuccessful_explore_time: int,
                             par_current_minimum_search_for_place_in_nest_time: int,
                             par_light_intensity: float,
                             par_current_items_number: int,
                             par_random_seed: int):
    for element_iterator in par_element_tree.iter():
        if element_iterator.tag == 'experiment':
            element_iterator.attrib['length'] = '800'
        if element_iterator.tag == 'experiment':
            element_iterator.attrib['random_seed'] = str(par_random_seed)
        if element_iterator.tag == 'state':
            element_iterator.attrib['minimum_resting_time'] = str(par_current_minimum_resting_time)
            element_iterator.attrib['minimum_unsuccessful_explore_time'] = str(
                par_current_minimum_unsuccessful_explore_time)
            element_iterator.attrib['minimum_search_for_place_in_nest_time'] = str(
                par_current_minimum_search_for_place_in_nest_time)
        if element_iterator.tag == 'foraging':
            element_iterator.attrib['items'] = str(par_current_items_number)
        if element_iterator.tag == 'arena':
            element_iterator.attrib['size'] = '{}, {}, 2'.format(par_y_length, par_x_length)
        if element_iterator.tag == 'box' and element_iterator.attrib['id'] == 'wall_north':
            element_iterator.attrib['size'] = '{},0.1,0.5'.format(par_y_length)
            for element_child in element_iterator.iter():
                if element_child.tag == 'body':
                    element_child.attrib['position'] = '0,{:5.1f},0'.format(par_x_length / 2)
        if element_iterator.tag == 'box' and element_iterator.attrib['id'] == 'wall_south':
            element_iterator.attrib['size'] = '{},0.1,0.5'.format(par_y_length)
            for element_child in element_iterator.iter():
                if element_child.tag == 'body':
                    element_child.attrib['position'] = '0,{:5.1f},0'.format(-par_x_length / 2)
        if element_iterator.tag == 'box' and element_iterator.attrib['id'] == 'wall_east':
            element_iterator.attrib['size'] = '0.1,{},0.5'.format(par_x_length)
            for element_child in element_iterator.iter():
                if element_child.tag == 'body':
                    element_child.attrib['position'] = '{:5.1f},0,0'.format(par_y_length / 2)
        if element_iterator.tag == 'box' and element_iterator.attrib['id'] == 'wall_west':
            element_iterator.attrib['size'] = '0.1,{},0.5'.format(par_x_length)
            for element_child in element_iterator.iter():
                if element_child.tag == 'body':
                    element_child.attrib['position'] = '{:5.1f},0,0'.format(-par_y_length / 2)
        if element_iterator.tag == 'light' and element_iterator.attrib['id'] == 'light_1':
            element_iterator.attrib['intensity'] = '{:5.1f}'.format(par_light_intensity)
            element_iterator.attrib['position'] = '{:5.1f},{:5.1f},1'.format(-par_y_length / 2 + 0.3,
                                                                             -par_x_length * 0.4)
        if element_iterator.tag == 'light' and element_iterator.attrib['id'] == 'light_2':
            element_iterator.attrib['intensity'] = '{:5.1f}'.format(par_light_intensity)
            element_iterator.attrib['position'] = '{:5.1f},{:5.1f},1'.format(-par_y_length / 2 + 0.3,
                                                                             -par_x_length * 0.2)
        if element_iterator.tag == 'light' and element_iterator.attrib['id'] == 'light_3':
            element_iterator.attrib['intensity'] = '{:5.1f}'.format(par_light_intensity)
            element_iterator.attrib['position'] = '{:5.1f},{:5.1f},1'.format(-par_y_length / 2 + 0.3,
                                                                             par_x_length * 0.2)
        if element_iterator.tag == 'light' and element_iterator.attrib['id'] == 'light_4':
            element_iterator.attrib['intensity'] = '{:5.1f}'.format(par_light_intensity)
            element_iterator.attrib['position'] = '{:5.1f},{:5.1f},1'.format(-par_y_length / 2 + 0.3,
                                                                             par_x_length * 0.4)
        if element_iterator.tag == 'position':
            element_iterator.attrib['min'] = '{:5.1f},{:5.1f},0'.format(-par_y_length / 2, -par_x_length / 2)
            element_iterator.attrib['max'] = '{:5.1f},{:5.1f},0'.format(-par_y_length * 0.25, par_x_length / 2)
        if element_iterator.tag == 'entity':
            element_iterator.attrib['quantity'] = str(par_current_bot_number)
        if element_iterator.tag == 'visualization':
            par_root.remove(element_iterator)

    par_element_tree.write('/Users/lucianofranchin/Documents/Github_repos/'
                           'argos3-examples/experiments/foraging_execution.argos')


def compute_parameters_and_edit_files(par_x_length: int,
                                      par_y_length: int,
                                      par_initial_arena_size: int,
                                      par_initial_bot_number: int,
                                      par_initial_minimum_resting_time: int,
                                      par_initial_minimum_unsuccessful_explore_time: int,
                                      par_initial_minimum_search_for_place_in_nest_time: int,
                                      par_initial_items_number: int,
                                      par_initial_light_intensity: float):
    current_arena_size = par_x_length * par_y_length
    size_increase = current_arena_size / par_initial_arena_size
    current_bot_number = int(size_increase * par_initial_bot_number)
    current_minimum_resting_time = int(size_increase * par_initial_minimum_resting_time)
    current_minimum_unsuccessful_explore_time = \
        int(size_increase * par_initial_minimum_unsuccessful_explore_time)
    current_minimum_search_for_place_in_nest_time = \
        int(size_increase * par_initial_minimum_search_for_place_in_nest_time)
    current_items_number = int(size_increase * par_initial_items_number)
    current_light_intensity = size_increase * par_initial_light_intensity

    modify_foraging_loop_func_cpp_file(par_x_length=par_x_length,
                                       par_y_length=par_y_length)
    compile_repo()

    return current_bot_number, current_minimum_resting_time, current_minimum_unsuccessful_explore_time, \
        current_minimum_search_for_place_in_nest_time, current_items_number, current_light_intensity


def create_nominal_foraging_dataset():
    # open xml file
    xml_file = ElementTree.parse('/Users/lucianofranchin/Documents/Github_repos/'
                                 'argos3-examples/experiments/foraging.argos')
    root = xml_file.getroot()

    random.seed(Parser.read_seed())

    # variables to manage the simulation execution
    nominal_exp_repetitions = 4

    # variables to modify simulation parameters
    arena_dimensions = [5, 6, 7, 8]
    initial_arena_size = arena_dimensions[0] ** 2
    initial_bot_number = 20
    initial_minimum_resting_time = 50
    initial_minimum_unsuccessful_explore_time = 600
    initial_minimum_search_for_place_in_nest_time = 50
    initial_items_number = 15
    initial_light_intensity = 3.0

    print('Doing Nominal')
    for x_length in arena_dimensions:
        for y_length in arena_dimensions:
            print('Doing Arena {}x{}'.format(x_length, y_length))

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

            seeds = random.sample(range(0, 1000), nominal_exp_repetitions)

            for i in range(nominal_exp_repetitions):
                print('Doing nominal rep: {} out of {}'.format(i, nominal_exp_repetitions))
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
                subproc = subprocess.Popen('argos3 -c experiments/foraging_execution.argos',
                                           stdin=subprocess.PIPE,
                                           shell=True)
                # nominal experiment?
                subproc.communicate(input=b'Y')


if __name__ == '__main__':
    pass
