# HOG-feature-descriptor-implementation

A simple implementation of Histogram of oriented gradient (HOG) descriptor.

## Steps
* Get differential images from grayscale image.
* Compute gradient.
* Build the histogram of oriented gradients for all cells.
* Build normalized block descriptor from cells.

## Dependencies
* Numpy
* matplotlib
* OpenCV 3.4.2 (For image loading)
