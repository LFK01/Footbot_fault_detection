class AreaPartition:
    def __init__(self,
                 left_bound: float,
                 right_bound: float,
                 low_bound: float,
                 top_bound: float):
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.low_bound = low_bound
        self.top_bound = top_bound
        self.visited = False
