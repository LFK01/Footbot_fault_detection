import math


class FootBot:
    def __init__(self, identifier):
        self.identifier = identifier
        self.positions = []
        self.speed_data = [0.0]

    def add_list_positions(self, positions_list):
        self.positions.extend(positions_list)
        self.compute_speed()

    def compute_speed(self):
        previous_position = self.positions[0]
        for current_position in self.positions[1:]:
            distance_x = previous_position[0] - current_position[0]
            distance_y = previous_position[1] - current_position[1]
            traversed_distance = math.sqrt(distance_x**2 + distance_y**2)
            self.speed_data.append(traversed_distance)
            previous_position = current_position
