import numpy
import cv2
import math
import matplotlib.pyplot as plt
from queue import *
from collections import namedtuple
import itertools
import random
import copy
import statistics
from collections import defaultdict
import scipy
import scipy.sparse, scipy.spatial


def grayScale(*image):
    gray_image = cv2.cvtColor(*image, cv2.COLOR_BGR2GRAY)
    return gray_image

def kmeansclustering(image) :
    Z = image.reshape((-1, 3))
    Z = numpy.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 6
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = numpy.uint8(center)
    clusteredImage = center[label.flatten()]
    clusteredImageFinal = clusteredImage.reshape((image.shape))

    cv2.imshow('Clustered Image', clusteredImageFinal)
    return clusteredImageFinal

def calculateGradient(image,edges):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    angles  = numpy.zeros(image.shape)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if(edges[row][col]>0):
                angles[row][col] = math.atan2(sobely[row][col], sobelx[row][col])


    return angles

def angleDifference(angle1, angle2):
    return math.atan2(math.sin(angle1-angle2), math.cos(angle1-angle2))

def rayLength(ray):
    return ((ray[0][0] - ray[-1][0]) ** 2 + (ray[0][1] - ray[-1][1]) ** 2) ** .5 #https://stackoverflow.com/questions/509211/understanding-pythons-slice-notation

def normalize(value, oldMin, oldMax, newMin, newMax):
  """ interpolation function from http://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio
  """
  return (((value - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin


def findRay(startPixel, angles, edgesSet, maxRayLength, direction):
    row,column = startPixel
    height,width = angles.shape
    rayLength = 1
    rayDirection = angles[row][column]
    rayValid = False
    ray = [(row,column)]
    while rayLength < maxRayLength:
        pixel = (int(row + math.sin(rayDirection) * rayLength * direction),
                 int(column + math.cos(rayDirection) * rayLength * direction))
        if pixel[0] >= height or pixel[0] < 0 or pixel[1] >= width or pixel[1] < 0:
            return None

        if not rayValid:
            rayValid = True
        ray.append(pixel)

        if pixel in edgesSet:
            oppositeDirection = angles[pixel[0]][pixel[1]]

            if angleDifference(rayDirection, oppositeDirection) > math.pi / 2:
                rayValid = False

            if rayValid:
                return ray
            else:
                return None

        rayLength += 1
    return None


def createRaysForEdges(edges, angles, direction, maxRayLength=100):

    swtPointsList = numpy.zeros((edges.shape[0],edges.shape[1]))
    swtPointsList.fill(255)
    rayList = []
    validEdges = edges.nonzero()
    edgesZip = zip(validEdges[0],validEdges[1])
    edgesSet = set(edgesZip)
    ray = []
    for(row,column) in edgesSet:
        ray = findRay((row,column), angles, edgesSet, maxRayLength, direction)
        if ray:
            if(len(ray) > 1 ):
                print("Ray found")
                rayList.append(ray)

    allRayLengths = list(map(lambda x: rayLength(x), filter(lambda x: x!=None, rayList)))
    print(len(allRayLengths))
    if len((allRayLengths)) == 0:
        return [swtPointsList, None]
    minL,maxL = min((allRayLengths)), max((allRayLengths))
    print(minL,maxL)
    i=0
    print(len(rayList))

    for ray in rayList:
        for pixel in ray:
            i+=1
            print("Plotting", i)
            swtPointsList[pixel[0],pixel[1]] = min(normalize(rayLength(ray),minL,maxL,0,255), swtPointsList[pixel[0], pixel[1]])


    return [swtPointsList,rayList]

def strokeWidthTransform(image,dark_on_light) :
    edges = edgeDetection(image)
    cv2.imshow('Edges',edges)
    cv2.imshow('Image',image)
    gradientAngles = calculateGradient(image,edges)
    cv2.imshow("Angles",gradientAngles)
    firstPass, rays = createRaysForEdges(edges, gradientAngles, dark_on_light)

    if rays == None:
        print("No rays found")
        return firstPass

    secondPass = refineRays(firstPass, rays)
    return secondPass

def gaussianThresholding(*image):
    thresholdImage = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    return thresholdImage

def edgeDetection(image):
    inputImage = numpy.zeros((image.shape[0],image.shape[1]))
    inputImage = numpy.uint8(inputImage)

    v = numpy.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(image, lower, upper)
    print(edges)
    return edges

def generateListOfAllPixels(rows, cols):
	all_pixels = []
	for i in range(rows):
		for j in range(cols):
			all_pixels.append((i,j))
	return all_pixels

def refineRays(swt, rays):

  for ray in rays:
    medianMap = map(lambda x: swt[x[0]][x[1]], ray)
    medianLength = numpy.min(list(medianMap)) # or trying median here
    print(medianLength)
    # medianLength = np.min(map(lambda x: swt[x[0]][x[1]], ray))
    for pixel in ray:
      if swt[pixel[0]][pixel[1]] > medianLength:
        swt[pixel[0]][pixel[1]] = medianLength
  return swt


def connectComponents(swt):
    class Label(object):
        def __init__(self, value):
            self.value = value
            self.parent = self
            self.rank = 0

        def __eq__(self, other):
            if type(other) is type(self):
                return self.value == other.value
            else:
                return False

        def __ne__(self, other):
            return not self.__eq__(other)

    ld = {}

    def MakeSet(x):
        try:
            return ld[x]
        except KeyError:
            item = Label(x)
            ld[x] = item
            return item

    def Find(item):
        # item = ld[x]
        if item.parent != item:
            item.parent = Find(item.parent)
        return item.parent

    def Union(x, y):
        x_root = Find(x)
        y_root = Find(y)
        if x_root == y_root:
            return x_root

        if x_root.rank < y_root.rank:
            x_root.parent = y_root
            return y_root
        elif x_root.rank > y_root.rank:
            y_root.parent = x_root
            return x_root
        else:
            y_root.parent = x_root
            x_root.rank += 1
            return x_root

    trees = {}

    label_map = numpy.zeros(shape=swt.shape, dtype=numpy.uint16)
    next_label = 1

    swt_ratio_threshold = 3.0
    for y in range(swt.shape[0]):
        for x in range(swt.shape[1]):
            sw_point = swt[y, x]
            if sw_point < numpy.Infinity and sw_point > 0:
                neighbors = [(y, x - 1),  # west
                             (y - 1, x - 1),  # northwest
                             (y - 1, x),  # north
                             (y - 1, x + 1)]  # northeast
                connected_neighbors = None
                neighborvals = []

                for neighbor in neighbors:
                    # west
                    try:
                        sw_n = swt[neighbor]
                        label_n = label_map[neighbor]
                    except IndexError:
                        continue
                    if label_n > 0 and sw_n / sw_point < swt_ratio_threshold and sw_point / sw_n < swt_ratio_threshold:
                        neighborvals.append(label_n)
                        if connected_neighbors:
                            connected_neighbors = Union(connected_neighbors, MakeSet(label_n))
                        else:
                            connected_neighbors = MakeSet(label_n)

                if not connected_neighbors:

                    trees[next_label] = (MakeSet(next_label))
                    label_map[y, x] = next_label
                    next_label += 1
                else:

                    label_map[y, x] = min(neighborvals)
                    trees[connected_neighbors.value] = Union(trees[connected_neighbors.value], connected_neighbors)

    # Second pass. re-base all labeling with representative label for each connected tree
    layers = {}
    contours = defaultdict(list)
    for x in range(swt.shape[1]):
        for y in range(swt.shape[0]):
            if label_map[y, x] > 0:
                item = ld[label_map[y, x]]
                common_label = Find(item).value
                label_map[y, x] = common_label
                contours[common_label].append([x, y])
                try:
                    layer = layers[common_label]
                except KeyError:
                    layers[common_label] = numpy.zeros(shape=swt.shape, dtype=numpy.uint16)
                    layer = layers[common_label]

                layer[y, x] = 1
    return layers



def findLetters(swt, shapes):

    swts = []
    heights = []
    widths = []
    topleft_pts = []
    images = []

    for label, layer in shapes.items():
        (nz_y, nz_x) = numpy.nonzero(layer)
        east, west, south, north = max(nz_x), min(nz_x), max(nz_y), min(nz_y)
        width, height = east - west, south - north

        if width < 8 or height < 8:
            continue

        if width / height > 10 or height / width > 10:
            continue

        diameter = math.sqrt(width * width + height * height)
        median_swt = numpy.median(swt[(nz_y, nz_x)])
        if diameter / median_swt > 10:
            continue

        if width / layer.shape[1] > 0.4 or height / layer.shape[0] > 0.4:
            continue

        swts.append([math.log(median_swt, 2)])
        heights.append([math.log(height, 2)])
        topleft_pts.append(numpy.asarray([north, west]))
        widths.append(width)
        images.append(layer)

    return swts, heights, widths, topleft_pts, images


def findWords(swts, heights, widths, topleft_pts, images):
    # Find all shape pairs that have similar median stroke widths
    swt_tree = scipy.spatial.KDTree(numpy.asarray(swts))
    stp = swt_tree.query_pairs(1)

    # Find all shape pairs that have similar heights
    height_tree = scipy.spatial.KDTree(numpy.asarray(heights))
    htp = height_tree.query_pairs(1)

    # Intersection of valid pairings
    isect = htp.intersection(stp)

    chains = []
    pairs = []
    pair_angles = []
    for pair in isect:
        left = pair[0]
        right = pair[1]
        widest = max(widths[left], widths[right])
        distance = numpy.linalg.norm(topleft_pts[left] - topleft_pts[right])
        if distance < widest * 3:
            delta_yx = topleft_pts[left] - topleft_pts[right]
            angle = numpy.arctan2(delta_yx[0], delta_yx[1])
            if angle < 0:
                angle += numpy.pi

            pairs.append(pair)
            pair_angles.append(numpy.asarray([angle]))

    angle_tree = scipy.spatial.KDTree(numpy.asarray(pair_angles))
    atp = angle_tree.query_pairs(numpy.pi / 12)

    for pair_idx in atp:
        pair_a = pairs[pair_idx[0]]
        pair_b = pairs[pair_idx[1]]
        left_a = pair_a[0]
        right_a = pair_a[1]
        left_b = pair_b[0]
        right_b = pair_b[1]

        added = False
        for chain in chains:
            if left_a in chain:
                chain.add(right_a)
                added = True
            elif right_a in chain:
                chain.add(left_a)
                added = True
        if not added:
            chains.append(set([left_a, right_a]))
        added = False
        for chain in chains:
            if left_b in chain:
                chain.add(right_b)
                added = True
            elif right_b in chain:
                chain.add(left_b)
                added = True
        if not added:
            chains.append(set([left_b, right_b]))

    word_images = []
    for chain in [c for c in chains if len(c) > 3]:
        for idx in chain:
            word_images.append(images[idx])

    return word_images
