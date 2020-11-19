import numpy as np
from Piece import Piece


class Board:
    def __init__(self, piece_size, board_size):
        """

        :param piece_size: in pixels
        :param board_size: in pixels
        """
        self.shape = self._calculeBoardSize(piece_size, board_size)
        self.histograms = np.zeros(self.shape)
        self.board = [[0] * self.shape[0]] * self.shape[1]  # Save Pieces objects in it

    def _calculeBoardSize(self, piece_size, board_size):
        """
        Calculates the shape for the board, numbers of columns and number of rows
        :param piece_size:
        :param board_size:
        :return:
        """
        rows = 6  # int(board_size[1]/piece_size[1])
        columns = 8  # int(board_size[0]/piece_size[0])
        return (columns, rows)

    def setHistogram(self, histograms):
        self.histograms = histograms

    def matchPiece(self, piece_object):

        return (0,0)

    def getShape(self):
        return self.shape