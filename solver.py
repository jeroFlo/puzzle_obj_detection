import os
import argparse
import cv2 as cv
import dlib
import numpy as np
from Point2D import Point2d

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
        self.base_img = cv.imread('spiderman_2.jpg')
        self.base_img_gray = cv.imread('spiderman_2.jpg', cv.IMREAD_GRAYSCALE)
        self.piece_size = (40, 40)

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
        detector = dlib.simple_object_detector(self.detector)
        if video is not True:
            d_img = cv.imread(d_img)

        dets = detector(d_img)
        d_img = cv.cvtColor(d_img, cv.COLOR_BGR2RGB)
        print("Number of pieces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()
            ))
            d_img_crop = d_img[d.top() - 10:d.bottom(), d.left() - 10:d.right()]
            # d_img_crop = do_segmentation(d_img_crop)
            self.do_match(d_img_crop)
            # cv.rectangle(d_img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)

        # cv.imshow('frame', d_img)

    def do_streaming(self):
        cap = cv.VideoCapture(0)  # one as parameter to use a second camera
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

    def do_match(self, img):
        if img.size != 0:
            limit = 3
            alls = False
            img_base_orig_gray = self.base_img_gray.copy()
            #if self.lineDetection(img):
                #img_base_orig_gray[40:200, 40:280]= 0
                #limit = 2
                #alls = True

            img2 = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  # query image
            # Initiate detector
            orb = cv.BRISK_create()
            # orb = cv.SIFT_create(sigma=1.3)

            # find the keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(img_base_orig_gray, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            # print("{}, {}".format(des1, des2))
            if des2 is not None:
                # create BFMatcher object
                # bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
                bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
                # Match descriptors.
                matches = bf.match(des1, des2)
                # Sort them in the order of their distance.
                coord2 = []
                #points = []
                matches = sorted(matches, key=lambda x: x.distance)
                for m in matches[:50]:
                    # print("{},{},{},{}".format(m.distance, m.trainIdx, m.queryIdx, m.imgIdx))
                    # print("{},{},{},{}".format(n.distance, n.trainIdx, n.queryIdx, n.imgIdx))
                    x, y = kp1[m.queryIdx].pt
                    x, y = int(x / self.piece_size[0]), int(y / self.piece_size[1])
                    p = Point2d(kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1])
                    #points.append(p)
                    # print("{},{},{}".format(kp1[m.queryIdx].pt, x, y))
                    coord2.append((x, y))

                img3 = self.base_img.copy()
                '''CLusters 
                self.quickSort(points, 0, len(points) - 1)
                clusters = {}
                i = 0
                while i < len(points):
                    bound_box_x1 = points[i].x - self.piece_size[0] / 2
                    bound_box_x1 = bound_box_x1 if bound_box_x1 >= 0 else 0
                    bound_box_x2 = points[i].x + self.piece_size[0] / 2
                    bound_box_x2 = bound_box_x2 if bound_box_x2 >= 0 else 0
                    bound_box_y1 = points[i].y - self.piece_size[1] / 2
                    bound_box_y1 = bound_box_y1 if bound_box_y1 >= 0 else 0
                    bound_box_y2 = points[i].y + self.piece_size[1] / 2
                    bound_box_y2 = bound_box_y2 if bound_box_y2 >= 0 else 0
                    x1y1 = Point2d(bound_box_x1, bound_box_y1)
                    x2y2 = Point2d(bound_box_x2, bound_box_y2)
                    clusters[(bound_box_x1, bound_box_y1)] = 0
                    i += 1
                    if i >= len(points):
                        break
                    while self._isInside(points[i], x1y1, x2y2):
                        clusters[(bound_box_x1, bound_box_y1)] += 1
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
                
                '''
                best = self._get_best_options(coord2, limit, alls)
                for coord_keys in best.keys():
                    y1 = coord_keys[1] * self.piece_size[1]
                    y2 = y1 + self.piece_size[1]
                    x1 = coord_keys[0] * self.piece_size[0]
                    x2 = x1 + self.piece_size[0]
                    cv.rectangle(img3, (x1, y1), (x2, y2), (0, 0, 255), 2)

                img4 = cv.drawMatches(self.base_img_gray, kp1, img2, kp2, matches[:50], None,
                                      flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                cv.imshow('Matching with rectangle', img3)
                cv.imshow('Matching with lines', img4)
                #cv.waitKey(0)

    def lineDetection(self, img):
        if img.size != 0:
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            # Apply edge detection method on the image
            edges = cv.Canny(gray, 100, 200)
            edges = self.dilation(edges, size=4, iter=2)
            edges = self.erosion(edges, size=3, iter=2)

            copy = edges.copy()

            line = []
            row_limit = 7
            size_limit = ((img.shape[0] + img.shape[1]) / 2) * 0.30
            for i in range(4):
                ones = np.where(edges == 255)
                if len(ones) == 0:
                    break
                list_of_coord = list(zip(ones[0], ones[1]))
                if len(list_of_coord) == 0:
                    break
                corner = list_of_coord[0]
                pre_coord_x = corner[1]  # coordinate in x (columns)
                pre_coord_y = corner[0]  # coordinate in y (rows)
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
                        pre_coord_x = coord[1]-1
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
                if gaps <= 1 and size > size_limit:
                    line[i] = True
                edges = np.rot90(edges)

            print(line)

            cv.imshow('hdj', copy)
            #cv.waitKey(0)
            return True in line

    def _isInside(self, p, b1, b2):
        if p <= b2 and p >= b1:
            return True
        return False

    def do_match_2(self, img):
        if img.size != 0:
            img2 = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  # trainImage
            sift = cv.SIFT_create(sigma=1.3)
            kp1, des1 = sift.detectAndCompute(self.base_img_gray, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            if des2 is not None:
                # BFMatcher with default params
                bf = cv.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                # Apply ratio test
                good = []
                coord = []
                coord2 = []
                print(len(matches))
                try:
                    for m, n in matches:
                        # print("{},{},{},{}".format(m.distance, m.trainIdx, m.queryIdx, m.imgIdx))
                        # print("{},{},{},{}".format(n.distance, n.trainIdx, n.queryIdx, n.imgIdx))
                        if m.distance < 0.75 * n.distance:
                            good.append([m])
                            x, y = kp1[m.queryIdx].pt
                            x, y = int(x / self.piece_size[0]), int(y / self.piece_size[1])
                            # coord.append(kp1[m.queryIdx].pt)  # Match coordinates for base image
                            coord2.append((x, y))
                            # print("{},{},{}".format(kp1[m.queryIdx].pt, x, y))
                except ValueError:
                    good = matches[:20]

                best_coord = self._get_best_options(coord2)
                img3 = self.base_img
                # if best_coord is not None:
                for coord_keys in best_coord.keys():
                    y1 = coord_keys[1] * self.piece_size[1]
                    y2 = y1 + self.piece_size[1]
                    x1 = coord_keys[0] * self.piece_size[0]
                    x2 = x1 + self.piece_size[0]
                    cv.rectangle(img3, (x1, y1), (x2, y2), (0, 0, 255), 2)
                '''
                hists = {}
                piece_hist = cv.calcHist(img[20:60, 20:60], [0, 1, 2], None, [8, 8, 8],
                                         [0, 256, 0, 256, 0, 256])
                piece_hist = cv.normalize(piece_hist, piece_hist).flatten()
                for coord_keys in best_coord.keys():
                    y1 = coord_keys[1] * self.piece_size[1]
                    y2 = y1 + self.piece_size[1]
                    x1 = coord_keys[0] * self.piece_size[0]
                    x2 = x1 + self.piece_size[0]
                    hist = cv.calcHist(self.base_img_gray[y1:y2, x1:x2], [0, 1, 2], None, [8, 8, 8],
                                       [0, 256, 0, 256, 0, 256])
                    hist = cv.normalize(hist, hist).flatten()
                    d = cv.compareHist(piece_hist, hist, cv.HISTCMP_BHATTACHARYYA)
                    hists[coord_keys] = d

                print(hists)
                '''
                img4 = cv.drawMatchesKnn(self.base_img_gray, kp1, img2, kp2, good, None,
                                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                cv.imshow('Matching', img3)
                cv.imshow('Matchinglines', img4)
                cv.waitKey(0)

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

    def _get_best_options(self, coord, limit=3, alls= False):
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
    '''
    def do_segmentation(self, img):
        # img = cv.imread(image_path)
        # binary = binarization(img, otsu=
        img2 = img  # cv.cvtColor(img, cv.COLOR_BGR2RGB)
        binary = cv.Canny(img, 100, 200)
        cv.imshow('binary', binary)

        binary = dilation(binary, size=3, iter=1)
        cv.imshow('dilatatio', binary)
        sure_bg = closing(binary, 4, iter=2)
        cv.imshow('clso', sure_bg)
        # binary = dilation(binary, size=4)
        # aux = np.dstack((sure_bg, sure_bg))
        new_img = cv.bitwise_and(img2, -1, sure_bg)
        # new_img= cv.cvtColor(new_img, cv.COLOR_GRAY2RGB)
        cv.imshow('dilatation/closing', new_img)
        return img2
       

    def do_resize(self, img, scale_percent=5):
        # print("Original dimension: {}".format(img.shape))
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # print("New dimension: {}".format(dim))
        return cv.resize(img, dim, interpolation=cv.INTER_AREA)

    def binarization(self, img, thres=127, otsu=False, inv=True):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # cv.imshow('gray', gray)
        if not otsu:
            if inv:
                ret1, thresh = cv.threshold(gray, thres, 255, cv.THRESH_BINARY_INV)
            else:
                ret1, thresh = cv.threshold(gray, thres, 255, cv.THRESH_BINARY)
            return thresh

        if inv:
            ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        else:
            ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return thresh

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
            img = erosion(dilation(img, size), size)
        return img

    def opening(self, img, size=3, iter=1):
        for i_iter in range(iter):
            img = dilation(erosion(img, size), size)
        return img
    '''


def main():
    args = parser.parse_args()
    dataset_xml_path = args.dataset
    hyper = args.hyperpameters
    imgs_path = args.detect
    streaming = args.video
    solver = Solver()

    print('{}, {}, {}, {}'.format(dataset_xml_path, hyper, imgs_path, streaming))
    solver.do_streaming()
    if dataset_xml_path is not None:
        solver.do_train(dataset_xml_path[0], dataset_xml_path[1], dataset_xml_path[2], hyper)
    elif imgs_path is not None:
        solver.do_detect(imgs_path, False)
    elif streaming:
        solver.do_streaming()


if __name__ == '__main__':
    main()
