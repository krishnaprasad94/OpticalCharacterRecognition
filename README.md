# OpticalCharacterRecognition
This projects aims at locating text in image, subsequent isolation of text and classification of text. Epshtein, Ofek, and Wexler's Stroke Width Transform is primarily used to isolate text from image. The resulting image is input into an OCR engine (which is a work in progress).

Working
This is an ongoing project and text localization part of the project is the one currently developed and is a very basic prototype.

In this module, image is read from a source path on the device and is ran through PreProcessor Module, which performs SWT on the image.
The Stroke Width Transform is a technique used to extract text from a noisy image by isolating connected shapes that share a consistent stroke width. SWT works by traversing the image at each edge pixel, in the direction perpendicular to the edge, until another edge is found in the normal direction. Each pixel is identified to be a part of a stroke and the stroke width of different strokes in the image is calculated. Contiguous strokes having similar stroke widths ar identified and are isolated. This stems from the notion that a contiguous stroke typically represents a man-made character.

SWT logic :

Pre Process image :

Reducing the colors in the image using K-means clustering
Converting the reduced color image to grayscale image

SWT : 

Calculate the edge map of the image by using the Canny edge detector

Gradient direction(angle) for the edge pixels are  calculated using sobel operator.

We start from an edge pixel and move in its gradient direction till we reach some other edge pixel.

If other pixel with same gradient direction is not found, then the whole ray is discarded

Number of pixels in a valid stroke, is assigned as the stroke width to each pixel in that ray in the SWT dict. Valid stroke width is the one in which the end edge pixel has a gradient difference between –pi/2 and +pi/2 from start edge pixel The output of the previous steps is another image of the same size as the input image where each element contains the width of the stroke associated with the pixel.

Of the generated dataset, apply some intelligent filtering to the line sets; we should eliminate anything to small (width, height) to be a legible character, as well as anything too long or fat (width:height ratio), or too sparse (diameter:stroke width ratio) to realistically be a character.

Use a k-d tree to find pairings of similarly-stroked shapes (based on stroke width), and intersect this with pairings of similarly-sized shapes (based on height/width). Calculate the angle of the text between these two characters (sloping upwards, downwards?).

Use a k-d tree to find pairings of letter pairings with similar orientations. These groupings of letters likely form a word. Chain similar pairs together.

A final image with the words are produced and the image is dilated to correct intensity bumps generated during arithmetic operations.




