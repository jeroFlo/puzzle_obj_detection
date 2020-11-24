

class Piece:

    def __init__(self, type, hist=0.0):
        """

        :param types: A vector with 4 boolean values, one for each side.
                        position 0: 0 degrees
                        position 1: 90 degrees rotated and so on
        """
        self.isEdge = type
        self.descriptor = hist
        self.position = (-1, -1)

    def setPosition(self, position):
        """
        Set the position that this piece deserves in the board
        :param position:
        :return:
        """
        self.position = position

    def setDescriptor(self, descriptor):
        self.descriptor = descriptor

    def getDescriptor(self):
        return self.descriptor

    def getType(self):
        return self.isEdge

    def __str__(self):
        return str(self.position)+": "+str( self.descriptor)


    def __gt__(self, other):
        return self.descriptor > other.descriptor
