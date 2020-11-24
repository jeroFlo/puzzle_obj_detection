import os
import argparse
import cv2 as cv
import dlib
import numpy as np
from Point2D import Point2d
from Board import Board
from Piece import Piece
import math

parser = argparse.ArgumentParser(description='hog_svm')
parser.add_argument('-d', '--dataset', nargs=3,
                    help='dataset training xml path, test xml path, path/name for svm output file')
parser.add_argument('-w', '--hyperpameters', nargs=4, type=float, help='Hyperparameters "flip, cores, C, epsilon"',
                    default=[0, 2, 1, 0.01])
parser.add_argument('-e', '--detect',
                    help='Detection in image(s), folder image(s) path')
parser.add_argument('-v', '--video', help='Real time detection', default=False, action="store_true")


class Solver:
    def __init__(self):
        self.detector = "img3/detector.svm"
        self.base_img = cv.imread('resources/spiderman_3.jpeg')
        self.base_img_gray = cv.imread('resources/spiderman_3.jpeg', cv.IMREAD_GRAYSCALE)
        self.piece_size = (40, 40)  # this have to be calculated automatically
        self.board = Board(self.piece_size, self.base_img.shape)
        board_shape = self.board.getShape()
        self._setBoard(board_shape)

    def _setBoard(self, shape):
        for row in range(shape[0]):
            for col in range(shape[1]):
                image = self.base_img[row * self.piece_size[1]:row * self.piece_size[1] + self.piece_size[1],
                        col * self.piece_size[0]:col * self.piece_size[0] + self.piece_size[0]]
                #cv.imshow("({}, {})".format(row, col), image)
                hist = cv.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                                   [0, 256, 0, 256, 0, 256])
                hist = cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
                self.board.setHistogram(hist, row, col)

    def do_train(self, training_xml_path, testing_xml_path, name_svm, hyperparameters):
        options = dlib.simple_object_detector_training_options()

        options.detection_window_size = int(hyperparameters[0])
        # options.add_left_right_image_flips = True
        options.C = int(hyperparameters[2])
        options.num_threads = int(hyperparameters[1])
        options.epsilon = hyperparameters[3]
        options.be_verbose = True
        dlib.train_simple_object_detector(training_xml_path, name_svm, options)
        print("")  # Print blank line to create gap from previous output
        print("Training accuracy: {}".format(
            dlib.test_simple_object_detector(training_xml_path, name_svm)))
        print("Testing accuracy: {}".format(
            dlib.test_simple_object_detector(testing_xml_path, name_svm)))
        self.detector = name_svm

    def do_detect(self, d_img, video=True):
        #detector = dlib.simple_object_detector(self.detector)
        if video is not True:
            d_img = cv.imread(d_img)

        #dets = detector(d_img)
        d_img = cv.cvtColor(d_img, cv.COLOR_BGR2RGB)
        #print("Number of pieces detected: {}".format(len(dets)))
        '''
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()
            ))
            d_img_crop = d_img[d.top() - 20:d.bottom(), d.left() - 20:d.right()]
            # d_img_crop = do_segmentation(d_img_crop)
            self.do_match(d_img_crop)
            # cv.rectangle(d_img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)
        '''
        d_img_crop = d_img[60:180, 100:220]
        #d_img_crop = d_img[140:340,220:420]
        cv.imshow('frame', d_img_crop)
        self.do_match_2(d_img_crop)

    def do_match_2(self, img):
        # Using features descriptor
        orb = cv.BRISK_create()
        try:
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            kp1, des1 = orb.detectAndCompute(self.base_img_gray, None)
            kp2, des2 = orb.detectAndCompute(gray, None)
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
        except cv.error:
            return None
        coord2 = []
        points = []
        matches = sorted(matches, key=lambda x: x.distance)
        for m in matches[:50]:
            x, y = kp1[m.queryIdx].pt
            x, y = int(x / self.piece_size[0]), int(y / self.piece_size[1])
            p = Point2d(kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1])
            #points.append(p)
            # print("{},{},{}".format(kp1[m.queryIdx].pt, x, y))
            coord2.append((x, y))

        img3 = self.base_img.copy()
        best = self._get_best_options(coord2)
        for coord_keys in best.keys():
            y1 = coord_keys[1] * self.piece_size[1]
            y2 = y1 + self.piece_size[1]
            x1 = coord_keys[0] * self.piece_size[0]
            x2 = x1 + self.piece_size[0]
            cv.rectangle(img3, (x1, y1), (x2, y2), (0, 0, 255), 2)

        img4 = cv.drawMatches(self.base_img_gray, kp1, img, kp2, matches[:50], None,
                              flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv.imshow('Matching with rectangle', img3)
        cv.imshow('Matching with lines', img4)
        #cv.waitKey(0)


    def do_match(self, img):
        img_segment = self.do_segmentation(img)
        if img_segment is not None:

            type = self.sidesDetection(img_segment)
            new_piece = Piece(type)

            # Using features descriptor
            orb = cv.BRISK_create()
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            kp1, des1 = orb.detectAndCompute(self.base_img_gray, None)
            kp2, des2 = orb.detectAndCompute(gray, None)
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            coord2 = {}
            points = []
            matches = sorted(matches, key=lambda x: x.distance)
            for m in matches[:50]:
                 points.append(Point2d(kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]))

            self.quickSort(points, 0, len(points)-1)
            img3 = self.base_img.copy()
            clusters = {}
            i = 0
            while i < len(points):
                bound_box_x1 = points[i].x - (self.piece_size[0]-10)
                bound_box_x1 = bound_box_x1 if bound_box_x1 >= 0 else 0
                bound_box_x2 = points[i].x + (self.piece_size[0] -10)
                bound_box_x2 = bound_box_x2 if bound_box_x2 >= 0 else 0
                bound_box_y1 = points[i].y - (self.piece_size[1] -10)
                bound_box_y1 = bound_box_y1 if bound_box_y1 >= 0 else 0
                bound_box_y2 = points[i].y + (self.piece_size[1] -10)
                bound_box_y2 = bound_box_y2 if bound_box_y2 >= 0 else 0
                x1y1 = Point2d(bound_box_x1, bound_box_y1)
                x2y2 = Point2d(bound_box_x2, bound_box_y2)
                clusters[(bound_box_x1, bound_box_y1)] = 0
                i += 1
                if i >= len(points):
                    break

                while self._isInside(points[i], x1y1, x2y2):
                    clusters[(bound_box_x1, bound_box_y1)]+=1
                    i += 1

                    if i >= len(points):
                        break


            print(clusters)


            for cluster in clusters.keys():
                y1 = int(cluster[1])
                y2 = int(y1 + self.piece_size[1])
                x1 = int(cluster[0])
                x2 = int(x1 + self.piece_size[0])
                cv.rectangle(img3, (x1, y1), (x2, y2), (0, 0, 255), 2)


            img4 = cv.drawMatches(self.base_img_gray, kp1, img, kp2, matches[:50], None,
                                      flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            cv.imshow('Matching with rectangle', img3)
            cv.imshow('Matching with lines', img4)


            ones = np.where(img_segment == 255)
            util_area = img[ones[0].tolist(), ones[1].tolist()]
            side1 = int(math.sqrt(util_area.shape[0]))
            side2 = int(math.sqrt(util_area.shape[0]))
            if side1 * side2 != util_area.shape[0]:
                util_area = util_area[:side1*side2]
            util_area = util_area.reshape((side1,side2, 3))

            try:
                cv.imshow("util", util_area)

                hist = cv.calcHist([util_area], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist = cv.normalize(hist, hist,0,255,cv.NORM_MINMAX)
                clusters = sorted(clusters.items(), key=lambda item: item[1], reverse=True)
                best1= self.board.matchPiece(new_piece, hist, clusters[:4])

            except cv.error:
                print("trone!!! Auxilio")
                cv.waitKey(0)
                return None

            cv.waitKey(0)
            # cv.imshow('Matching with rectangle', img3)
            # cv.imshow('Matching with lines', img4)

    def _isInside(self, p, b1, b2):
        if p <= b2 and p >= b1:
            return True
        return False

    def do_segmentation(self, img):
        try:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        except cv.error:
            return None

        canny = cv.Canny(gray, 80, 220)
        cv.imshow('canny', canny)
        # _, canny = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        # canny = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,15,6)
        edges = self.dilation(canny, iter=2)
        cv.imshow('dilatation', edges)

        # In order to have a successful flood it is necessary remove borders with 255 value
        right_bottom_corner = (img.shape[0] - 1, img.shape[1] - 1)
        edges[:1, :] = 0
        edges[right_bottom_corner[1]:, :] = 0
        edges[:, :1] = 0
        edges[:, right_bottom_corner[0]:] = 0

        im_flood = edges.copy()
        h, w = edges.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv.floodFill(im_flood, mask, (0, 0), 255)
        im_flood_inv = cv.bitwise_not(im_flood)
        im_out = edges | im_flood_inv
        im_out = self.erosion(im_out, iter=2)

        img_segment = cv.bitwise_and(img, img, mask=im_out)

        cv.imshow("Cut Image", edges)
        cv.imshow("Flood Image", im_flood)
        cv.imshow("Inverted Flood Image", im_flood_inv)
        cv.imshow("Foreground", im_out)
        cv.imshow("Segmentation", img_segment)
        # cv.waitKey(0)
        return im_out

    def sidesDetection(self, img):
        """
        This method tells us if a piece has a line border
        :param img: this is a binary image
        :return: true if this piece is an edge inside the puzzle
        """
        edges = img.copy()
        line = []
        row_limit = 7
        size_percentage = 0.29
        size_limit = ((img.shape[0] + img.shape[1]) / 2) * size_percentage

        for i in range(4):
            ones = np.where(edges == 255)  # getting coordinates where there is a white pixel
            list_of_coord = list(zip(ones[0], ones[1]))  # concatenating coordinates (rows, columns)
            try:
                corner = list_of_coord[0]
                pre_coord_x = corner[1]  # coordinate in x (columns)
                pre_coord_y = corner[0]  # coordinate in y (rows)
            except IndexError:
                return False

            line.append(False)
            gaps = 0
            size = 0
            rows = 0
            sum_size = 0
            sum_gaps = 0
            for coord in list_of_coord[1:]:
                if coord[0] - pre_coord_y != 0:
                    rows += 1
                    sum_size += size
                    sum_gaps += gaps
                    size = 0
                    gaps = 0
                    pre_coord_x = coord[1] - 1
                    if rows == row_limit:
                        break
                if coord[1] - pre_coord_x != 1:
                    gaps += 1

                size += 1
                pre_coord_x = coord[1]
                pre_coord_y = coord[0]

            gaps = sum_gaps / row_limit
            size = sum_size / row_limit
            print("Size limit: {}".format(size_limit))
            print("Gaps: {}".format(sum_gaps))
            print("Gaps average: {}".format(gaps))
            print("Size average: {}".format(size))
            if gaps <= 0.7 and size > size_limit:
                line[i] = True
            edges = np.rot90(edges)

        print(line)

        #cv.imshow('hdj', edges)
        # cv.waitKey(0)
        return True in line

    def _partition(self, arr, low, high):
        i = (low - 1)  # index of smaller element
        pivot = arr[high]  # pivot
        for j in range(low, high):

            # If current element is smaller than or
            # equal to pivot
            if arr[j] <= pivot:
                # increment index of smaller element
                i = i + 1
                arr[i], arr[j] = arr[j], arr[i]

        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return (i + 1)

    def quickSort(self, arr, low, high):
        if len(arr) == 1:
            return arr
        if low < high:
            # pi is partitioning index, arr[p] is now
            # at right place
            pi = self._partition(arr, low, high)

            # Separately sort elements before
            # partition and after partition
            self.quickSort(arr, low, pi - 1)
            self.quickSort(arr, pi + 1, high)

    def _get_best_options(self, coord, limit=3, alls=False):
        repeated = {}
        key_best = {}
        for i, c in enumerate(coord):
            if c in repeated:
                repeated[c] += 1
            else:
                repeated[c] = 0
        print(repeated)
        copy = repeated.copy()
        for key, val in copy.items():
            if val < limit:
                repeated.pop(key)
        print(repeated)
        if alls == False:

            if len(repeated) != 0:
                max = 0
                for key, val in repeated.items():
                    if val > max:
                        max = val
                        keym = key

                key_best[keym] = 1
            return key_best
        return repeated

    def erosion(self, img, size=3, iter=1):
        kernel = np.ones((size, size), np.uint8)
        # The first parameter is the original image,
        # kernel is the matrix with which image is
        # convolved and third parameter is the number
        # of iterations, which will determine how much
        # you want to erode/dilate a given image.
        img_erosion = cv.erode(img, kernel, iterations=iter)
        return img_erosion

    def dilation(self, img, size=3, iter=1):
        kernel = np.ones((size, size), np.uint8)
        img_dilation = cv.dilate(img, kernel, iterations=iter)
        return img_dilation

    def closing(self, img, size=3, iter=1):
        for i_iter in range(iter):
            img = self.erosion(self.dilation(img, size), size)
        return img

    def opening(self, img, size=3, iter=1):
        for i_iter in range(iter):
            img = self.dilation(self.erosion(img, size), size)
        return img

    def do_streaming(self):
        cap = cv.VideoCapture(1)  # one as parameter to use a second camera
        print("{}, {}".format(cap.get(5), cap.get(21)))  # frames per second, buffersize
        # changing frames size 320x240
        cap.set(3, 320)
        cap.set(4, 240)
        while True:
            ret, frame = cap.read()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.do_detect(gray)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()


def main():
    args = parser.parse_args()
    dataset_xml_path = args.dataset
    hyper = args.hyperpameters
    imgs_path = args.detect
    streaming = args.video
    solver = Solver()

    print('{}, {}, {}, {}'.format(dataset_xml_path, hyper, imgs_path, streaming))
    solver.do_streaming()  # for debug porpuses
    if dataset_xml_path is not None:
        solver.do_train(dataset_xml_path[0], dataset_xml_path[1], dataset_xml_path[2], hyper)
    elif imgs_path is not None:
        solver.do_detect(imgs_path, False)
    elif streaming:
        solver.do_streaming()


if __name__ == '__main__':
    main()
