# Cilia Segmentation
Cilia are micro-scopic hairlike structures that protrude from literally every cell in your body. They beat in regular, rhythmic patterns to perform myriad tasks, from moving nutrients in to moving irritants out to amplifying cell-cell signaling pathways to generating calcium fluid flow in early cell differentiation. Cilia, and their beating patterns, are increasingly being
implicated in a wide variety of syndromes that affected multiple organs.

# Approach

# Data
The data are all available on GCP: gs://uga-dsp/project2

* Data: Contains a bunch of folders (325 of them), named as hashes, each of which contains 100 consecutive frames of a gray scale video of cilia.

* Mask: Masks contains a number of PNG images (211 of them), named as hashes (cor-responding to the sub folders of data), that identify regions of the corresponding videos where cilia is.

# Prerequisites
* Python-3.6
* keras
* open-cv

# Results


# Authors
* Sumer Singh
* Andrew Durden
* Anant Tripathi

# Contibution
There are no specific guidlines for contibuting. If you see something that could be improved, send a pull request! If you think something should be done differently (or is just-plain-broken), please create an issue.

# Reference
[1] https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html

[2] https://towardsdatascience.com/image-pre-processing-c1aec0be3edf

