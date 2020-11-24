import numpy as np
from Piece import Piece
import cv2 as cv
import math
class Board:
    def __init__(self, piece_size, board_size):
        """

        :param piece_size: in pixels
        :param board_size: in pixels
        """
        self.shape = self._calculeBoardSize(piece_size, board_size)
        self.histograms = {}
        self.board = [[Piece(False) for i in range(self.shape[1])] for j in range(self.shape[0])] # [[Piece(False)] * self.shape[1]] * self.shape[0]  # Save Pieces objects in it


    def _calculeBoardSize(self, piece_size, board_size):
        """
        Calculates the shape for the board, numbers of columns and number of rows
        :param piece_size:
        :param board_size:
        :return:
        """
        rows = 6  # int(board_size[1]/piece_size[1])
        columns = 8  # int(board_size[0]/piece_size[0])
        return (rows, columns)

    def setHistogram(self, value, row, col):
        self.histograms[(row, col)] = value

    def matchPiece(self, piece, hist, clusters):
        piece_hist = hist
        distance_dict = {}
        print(piece.getType())
        rows_sum = {0:0,1:0, 2:0, 3:0, 4:0, 5:0}
        col_sum = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
        '''
        if piece.getType():
            print("I am an edge")
            rows = [0, self.shape[0]-1]
            columns= [0, self.shape[1]-1]
            for r in rows:
                rows_sum[r] = 0
                for c in range(self.shape[1]):
                    d = cv.compareHist(piece_hist, self.histograms[(r, c)], cv.HISTCMP_INTERSECT)
                    distance_dict[(r, c)] = d
                    rows_sum[r] += d
                    col_sum[c] += d

            for c in columns:
                for r in range(self.shape[0]):
                    d = cv.compareHist(piece_hist, self.histograms[(r, c)], cv.HISTCMP_INTERSECT)
                    distance_dict[(r, c)] = d
                    col_sum[c] += d
                    rows_sum[r] += d
        else:
        '''
        rows = range(0, self.shape[0]-1)
        columns = range(0, self.shape[1])
        for r in rows:
            rows_sum[r] = 0
            for c in columns:
                d = cv.compareHist(piece_hist, self.histograms[(r, c)], cv.HISTCMP_INTERSECT)
                distance_dict[(r, c)] = d
                rows_sum[r] += d
                col_sum[c] += d

        dict_sorted = sorted(distance_dict.items(), key=lambda item: item[1], reverse=True)
        rows_sum_sorted = sorted(rows_sum.items(), key=lambda item: item[1], reverse=True)
        cols_sum_sorted = sorted(col_sum.items(), key=lambda item: item[1], reverse=True)
        distance_dict.clear()
        if rows_sum_sorted[0][1] > cols_sum_sorted[0][1]:
            # hacer barrido en columnas
            centroid_y = rows_sum_sorted[0][0]* 40 + 20
            for c in columns:
                centroid_x = c * 40 + 20
                distance_dict[c] = 0
                for cluster in clusters:
                    distance_dict[c] += math.sqrt((cluster[0][0]-centroid_x)**2 + (cluster[0][1]-centroid_y)**2)

        else:
            # hacer barrido en filas
            centroid_x = cols_sum_sorted[0][0] * 40 + 20
            for r in rows:
                centroid_y = c * 40 + 20
                distance_dict[r] = 0
                for cluster in clusters:
                    distance_dict[r] += math.sqrt((cluster[0][0] - centroid_x) ** 2 + (cluster[0][1] - centroid_y) ** 2)
        print(distance_dict)
        print(dict_sorted)
        print(rows_sum_sorted)
        print(cols_sum_sorted)
        winner = dict_sorted[0]

        return 23


    def getShape(self):
        return self.shape

    def addPiece(self):
        pass