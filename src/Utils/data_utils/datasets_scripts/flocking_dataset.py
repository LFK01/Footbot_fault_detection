import os
import random
import subprocess
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

from src.Utils.Parser import Parser

unique_identifiers = [988509, 798487, 431729, 390440, 180813, 271132, 385338, 233141, 988669, 138629, 119624, 553598,
                      656608, 810547, 536310, 217807, 65727, 801666, 382391, 70310, 424223, 774421, 448367, 741793,
                      65324, 918823, 341083, 110215, 124020, 600711, 475880, 301161, 565105, 222936, 539865, 933812,
                      451156, 70586, 728355, 108334, 502606, 530894, 940892, 631580, 235461, 901709, 444414, 143542,
                      431701, 726988, 205866, 872650, 438690, 930588, 31328, 404534, 134304, 82321, 209066, 911885,
                      643832, 176708, 367101, 383962, 885416, 13460, 866581, 392090, 738605, 159911, 155522, 172647,
                      761172, 883708, 940476, 390539, 469942, 644506, 656464, 752853, 544558, 72538, 250202, 338721,
                      938997, 996454, 370142, 852222, 782373, 404319, 311688, 275981, 410178, 337294, 146182, 90656,
                      890359, 980958, 317110, 149667, 671869, 361809, 523400, 522726, 339587, 148215, 188545, 479168,
                      897195, 656465, 26370, 875558, 769736, 383113, 970163, 849284, 42857, 148668, 132460, 373352,
                      271434, 911779, 868825, 345356, 771675, 89566, 982824, 568368, 131657, 173302, 77204, 806570,
                      387240, 583561, 291837, 996661, 990140, 572474, 637427, 808088, 651373, 200353, 20511, 759539]


def get_unique_identifiers():
    return unique_identifiers


def compile_repo():
    os.chdir('/Users/lucianofranchin/Documents/Github_repos/argos3-examples/build/')
    subprocess.call('make', shell=True)


def create_flocking_dataset():
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
    # boolean list to cycle through nominal and non-nominal executions
    nominal_experiments = [True, False]
    # integer numbers to compute the percentage of robots that are non-nominal
    fault_modules = [10, 5, 3]
    fault_timesteps: list[int] = [0, 500, 1500, 4000]

    # open xml file
    xml_file_element = ElementTree.parse('/Users/lucianofranchin/Documents/Github_repos/'
                                         'argos3-examples/experiments/flocking.argos')
    xml_root = xml_file_element.getroot()

    # cycle between nominal experiments and fault experiments
    for exp_is_nominal in nominal_experiments:
        if exp_is_nominal:
            # nominal experiments should be repeated more times with different seed in order to depict
            # a realistic scenario
            random_seeds: list[int] = random.sample(range(0, 1000), nominal_exp_repetitions)
            for i in range(nominal_exp_repetitions):
                for position in positions_dict.keys():
                    modify_flocking_cpp_file(par_quantity=experiment_bot_quantity,
                                             par_position=position)
                    modify_flocking_xml_file(par_element_tree=xml_file_element,
                                             par_root=xml_root,
                                             par_positions=positions_dict,
                                             par_position=position,
                                             par_bot_quantity=experiment_bot_quantity,
                                             par_random_seed=random_seeds[i])
                    compile_repo()

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
                for position in positions_dict.keys():
                    modify_flocking_cpp_file(par_quantity=experiment_bot_quantity,
                                             par_position=position)
                    compile_repo()
                    execute_flocking_simulation_command(par_xml_file=xml_file_element,
                                                        par_root=xml_root,
                                                        par_positions=positions_dict,
                                                        par_position=position,
                                                        par_random_seed=random_seeds[i],
                                                        par_fault_timesteps=fault_timesteps,
                                                        par_experiment_bot_quantity=experiment_bot_quantity,
                                                        par_fault_module=experiment_bot_quantity)

            for fault_module in fault_modules:
                random_seed = random.randint(0, 100)
                for position in positions_dict:
                    modify_flocking_cpp_file(par_quantity=experiment_bot_quantity,
                                             par_position=position)
                    execute_flocking_simulation_command(par_xml_file=xml_file_element,
                                                        par_root=xml_root,
                                                        par_positions=positions_dict,
                                                        par_position=position,
                                                        par_random_seed=random_seed,
                                                        par_fault_timesteps=fault_timesteps,
                                                        par_experiment_bot_quantity=experiment_bot_quantity,
                                                        par_fault_module=fault_module)


def modify_flocking_xml_file(par_element_tree: ElementTree.ElementTree,
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
                           'argos3-examples/experiments/flocking_execution.argos')


def modify_flocking_cpp_file(par_quantity: int,
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


def execute_flocking_simulation_command(par_xml_file: ElementTree,
                                        par_root: Element,
                                        par_positions: dict,
                                        par_position: str,
                                        par_experiment_bot_quantity: int,
                                        par_random_seed: int,
                                        par_fault_timesteps: list[int],
                                        par_fault_module: int):
    modify_flocking_xml_file(par_element_tree=par_xml_file,
                             par_root=par_root,
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
    pass
