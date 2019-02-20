import numpy as np


def video_raster_reshape(vid):
    """
    Reshape 3d video matrix into a 2d matrix where each row is a pixel and each
    column is a frame, the picel order remains that of a raster scan of the vid

    Parameters:
    vid, a matrix shape f,n,m where f is the number of frames size n by m

    Result:
    matrix, a matrix shape n*m,f representing the video in the format listed
        above
    """
    # reorder frame and spatial axes for ultimate reshape
    vid = np.swapaxes(vid, 0, 2)
    vid = np.swapaxes(vid, 0, 1)

    # determine new matrix shape
    n_pixels = vid.shape[0] * vid.shape[1]
    n_frames = vid.shape[2]

    # reshape vid to matrix
    matrix = vid.reshape((n_pixels, n_frames))
    return matrix


def get_variance_as_im(vid):
    """
    Take in a video and return a sigle image of the pixel-wise variance over
    all of the frames

    Parameters:
    vid, a matrix shape f,n,m where f is the number of frames size n by m

    Result:
    var_im, a matrix shape n,m where each pixel hold the variance of that
        pixel's data over the video
    """
    # convert to matrix
    matrix = video_to_matrix(vid)

    # compute variance for each row
    variances = np.var(matrix, axis=1)

    # reshape variances to original frame shape
    var_im = variances.reshape((vid.shape[1], vid.shape[2]))
    return var_im
