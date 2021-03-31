class FootBot:
    def __init__(self, identifier):
        self.identifier = identifier
        self.positions = []

    def add_position(self, position):
        self.positions.append(position)

    def add_list_positions(self, positions_list):
        self.positions.extend(positions_list)
