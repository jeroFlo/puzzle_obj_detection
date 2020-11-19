class Point2d:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.bag = []

    def __le__(self, other):
        if self.x == other.x:
            if self.y < other.y or self.y == other.y:
                return True
            return False
        if self.x < other.x:
            return True
        return False

    def __ge__(self, other):
        if self.x == other.x:
            if self.y > other.y or self.y == other.y:
                return True
            return False
        if self.x > other.x:
            return True
        return False

    def __lt__(self, other):
        if self.x < other.x and self.y < other.y:
            return True
        else:
            return False

    def __str__(self):
        return "({}, {})".format(self.x, self.y)
