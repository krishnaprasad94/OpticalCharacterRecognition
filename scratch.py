import numpy
import cv2
from Modules.PreProcessor import preProcessImage
from PIL import Image
import matplotlib.pyplot as plt

image_path = '../Resources/alphabets2.png'

img = cv2.imread(image_path)
B,G,R = cv2.split(cv2.imread(image_path))
img_color = cv2.merge((R,G,B))
imgProcessed = img
assert isinstance(imgProcessed, object)
imgProcessed = preProcessImage.kmeansclustering(img)
imgProcessed = preProcessImage.grayScale(imgProcessed)

cv2.imshow('GrayScaled', imgProcessed)
swtPositive = preProcessImage.strokeWidthTransform(imgProcessed,1)
cv2.imwrite('Positive_Image_StrokeWidth.jpg', swtPositive)
shapes  = preProcessImage.connectComponents(swtPositive)
swt, height, width, pts, images = preProcessImage.findLetters(swtPositive, shapes)
wordImages = preProcessImage.findWords(swt, height, width, pts, images)
finalPositiveImage = numpy.zeros(swtPositive.shape)
for word in wordImages:
    finalPositiveImage += word
cv2.imwrite('finalPositiveStrokWidthImage.jpg', finalPositiveImage * 255)

swtPositiveDilated = 255 - cv2.dilate(255 - swtPositive, kernel = numpy.ones((2,2),numpy.uint8), iterations = 2)
swtNegative = preProcessImage.strokeWidthTransform(imgProcessed, -1)
swtNegativeDilated = 255 - cv2.dilate(255 - swtNegative, kernel=numpy.ones((2, 2), numpy.uint8), iterations=2)

cv2.imwrite('Negative_SWT_Image.jpg',swtNegative)
plt.subplot(3,2,1)
plt.imshow(img_color, interpolation="none")
plt.title('original image')

plt.subplot(3,2,3)
plt.imshow(swtPositive, cmap="gray", interpolation="none")
plt.title('positive swt of image')

plt.subplot(3,2,4)
plt.imshow(swtPositiveDilated, cmap="gray", interpolation="none")
plt.title('dilated positive swt of image')

plt.subplot(3,2,5)
plt.title('negative swt of image')
plt.imshow(swtNegative, cmap="gray", interpolation="none")

plt.subplot(3,2,6)
plt.title('dilated negative swt of image')
plt.imshow(swtNegativeDilated, cmap="gray", interpolation="none")

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()