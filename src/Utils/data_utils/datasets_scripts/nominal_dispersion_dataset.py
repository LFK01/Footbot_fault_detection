import random
import os
import subprocess
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

from src.Utils.Parser import Parser


def modify_dispersion_xml_file(par_element_tree: ElementTree.ElementTree,
                               par_root: Element,
                               par_x_length: int,
                               par_y_length: int,
                               par_current_bot_number: int,
                               par_light_intensity: float,
                               par_random_seed: int):
    for element_iterator in par_element_tree.iter():
        if element_iterator.tag == 'experiment':
            element_iterator.attrib['length'] = '500'
        if element_iterator.tag == 'experiment':
            element_iterator.attrib['random_seed'] = str(par_random_seed)
        if element_iterator.tag == 'arena':
            element_iterator.attrib['size'] = '{}, {}, 4'.format(par_y_length, par_x_length)
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
        if element_iterator.tag == 'light':
            element_iterator.attrib['intensity'] = '{:5.1f}'.format(par_light_intensity)
            element_iterator.attrib['position'] = '0,0,2'
        if element_iterator.tag == 'position':
            element_iterator.attrib['min'] = '{:5.1f},{:5.1f},0'.format((-par_y_length / 2)*0.2,
                                                                        (-par_x_length / 2)*0.2)
            element_iterator.attrib['max'] = '{:5.1f},{:5.1f},0'.format((par_y_length / 2)*0.2,
                                                                        (par_x_length / 2)*0.2)
        if element_iterator.tag == 'entity':
            element_iterator.attrib['quantity'] = str(par_current_bot_number)
        if element_iterator.tag == 'visualization':
            par_root.remove(element_iterator)

    par_element_tree.write('/Users/lucianofranchin/Documents/Github_repos/'
                           'argos3-examples/experiments/dispersion_execution.argos')


def compute_parameters(par_x_length: int,
                       par_y_length: int,
                       par_initial_arena_size: int,
                       par_initial_bot_number: int,
                       par_initial_light_intensity: float):
    current_arena_size = par_x_length * par_y_length
    size_increase = current_arena_size / par_initial_arena_size
    current_bot_number = int(size_increase * par_initial_bot_number)
    current_light_intensity = size_increase * par_initial_light_intensity

    return current_bot_number, current_light_intensity


def create_nominal_dispersion_dataset():
    # open xml file
    xml_file = ElementTree.parse('/Users/lucianofranchin/Documents/Github_repos/'
                                 'argos3-examples/experiments/footbot_dispersion.argos')
    root = xml_file.getroot()

    random.seed(Parser.read_seed())

    # variables to manage the simulation execution
    nominal_exp_repetitions = 4

    # variables to modify simulation parameters
    arena_dimensions = [10, 14, 18, 20]
    initial_arena_size = arena_dimensions[0] ** 2
    initial_bot_number = 35
    initial_light_intensity = 5.0

    print('Doing Nominal')
    for x_length in arena_dimensions:
        for y_length in arena_dimensions:
            print('Doing Arena {}x{}'.format(x_length, y_length))

            current_bot_number, current_light_intensity = compute_parameters(
                par_x_length=x_length,
                par_y_length=y_length,
                par_initial_arena_size=initial_arena_size,
                par_initial_bot_number=initial_bot_number,
                par_initial_light_intensity=initial_light_intensity)

            seeds = random.sample(range(0, 1000), nominal_exp_repetitions)

            for i in range(nominal_exp_repetitions):
                print('Doing nominal rep: {} out of {}'.format(i + 1, nominal_exp_repetitions))
                modify_dispersion_xml_file(par_element_tree=xml_file,
                                           par_root=root,
                                           par_x_length=x_length,
                                           par_y_length=y_length,
                                           par_current_bot_number=current_bot_number,
                                           par_light_intensity=current_light_intensity,
                                           par_random_seed=seeds[i])

                os.chdir('/Users/lucianofranchin/Documents/Github_repos/argos3-examples/')
                subproc = subprocess.Popen('argos3 -c experiments/dispersion_execution.argos',
                                           stdin=subprocess.PIPE,
                                           shell=True,
                                           text=True)
                # nominal experiment?
                subproc.communicate(input='Y\n'
                                          '{}\n'.format(x_length) +
                                          '{}\n'.format(y_length) +
                                          '{}\n'.format(current_bot_number))


if __name__ == '__main__':
    create_nominal_dispersion_dataset()
