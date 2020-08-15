# HOG-feature-descriptor-implementation

A simple implementation of Histogram of oriented gradient (HOG) descriptor.

<p float="left">
  <img src="/example/result_1.png" width="450" />
  <img src="/example/result_2.png" width="450" /> 
</p>


## Steps
* Get differential images from grayscale image.
* Compute gradient.
* Build the histogram of oriented gradients for all cells.
* Build normalized block descriptor from cells.

## Dependencies
* Numpy
* matplotlib
* OpenCV 3.4.2 (For image loading)

## Acknowledgment
Visualization code is from UMN Fall 2019 CSCI 5561 course material.
