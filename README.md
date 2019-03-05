# Cilia Segmentation
Cilia are micro-scopic hairlike structures that protrude from literally every cell in your body. They beat in regular, rhythmic patterns to perform myriad tasks, from moving nutrients in to moving irritants out to amplifying cell-cell signaling pathways to generating calcium fluid flow in early cell differentiation. Cilia, and their beating patterns, are increasingly being
implicated in a wide variety of syndromes that affected multiple organs.

# Approach

The training data consists of videos The first step would be to extract features from data.

As each sample is a video, we can make use of spatial as well as temporal features. 

The features we used are:
1) [Variance fluctuations](https://github.com/dsp-uga/Taylor-P2/wiki/Pixel-Variance)
2) [Optical Flow](https://github.com/dsp-uga/Taylor-P2/wiki/Optical-Flow) 
3) [Beat Frequency](https://github.com/dsp-uga/Taylor-P2/wiki/Beat-Frequency)

We pass the extracted features through a [U-NET](https://github.com/dsp-uga/Taylor-P2/wiki/Unet)

# Data
The data are all available on GCP: gs://uga-dsp/project2

* Data: Contains a bunch of folders (325 of them), named as hashes, each of which contains 100 consecutive frames of a gray scale video of cilia.

* Mask: Masks contains a number of PNG images (211 of them), named as hashes (cor-responding to the sub folders of data), that identify regions of the corresponding videos where cilia is.

# Prerequisites
* Python-3.6
* keras
* open-cv

# Scripts
## feature_engineering.py
* runs a folder of cilia videos in parallel extracting each feature as directed
### To run
$ python feature_engineering.py -i \<input directory\> -o \<output directory\> -f \<feature name\> [-g | if you want greyscale frame added]
"$ python feature_engineering.py -h" for help
## png_to_npy.py
* runs a folder of cilia video pngs converting them to npy files
### To run
$ python png_to_npy.py -i \<input directory\>  -o \<output directory\> [-s | to save while processing to conserve ram]
"$ python png_to_npy.py -h" for help


# Results

| Features Used                                       | IOU   |
|-----------------------------------------------------|-------|
| Beat Frequency + 1 frame                            | 15%   |
| Optical flow                                        | 24.5% |
| Variance Fluctuations, Beat Frequency, Optical Flow | 34.4% |


# Authors
* Sumer Singh
* Andrew Durden
* Anant Tripathi

# Contibution
There are no specific guidlines for contibuting. If you see something that could be improved, send a pull request! If you think something should be done differently (or is just-plain-broken), please create an issue.

# Reference
[1] https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html

[2] https://towardsdatascience.com/image-pre-processing-c1aec0be3edf

# License

This Project is under the MIT License. For more details visit License file here: https://github.com/dsp-uga/Taylor-P2/blob/master/LICENSE
