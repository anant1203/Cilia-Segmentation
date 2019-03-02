import numpy as np
import scipy.signal as signal
import cv2

import argparse
import glob
import os
import joblib


def video_raster_reshape(vid):
    """
    Reshape 3d video matrix into a 2d matrix where each row is a pixel and each
    column is a frame, the picel order remains that of a raster scan of the vid

    Parameters
    ----------
    vid, a matrix shape f,n,m where f is the number of frames size n by m

    Result
    ------
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

    Parameters
    ----------
    vid, a matrix shape f,n,m where f is the number of frames size n by m

    Result
    ------
    var_im, a matrix shape n,m where each pixel hold the variance of that
        pixel's data over the video
    """
    # convert to matrix
    matrix = video_raster_reshape(vid)

    # compute variance for each row
    variances = np.var(matrix, axis=1)

    # reshape variances to original frame shape
    var_im = variances.reshape((vid.shape[1], vid.shape[2]))
    return var_im


def get_beat_frequency(vid, f_size=15):
    """
    Perform fft on a video(100 frames) and extract the dominant frequency.
    Filtering is performed to reduce the effects of noise.

    Parameters
    ----------
    vid, a matrix shape f,n,m where f is the number of frames size n by m.
    f_size, the dimensions of the filter.

    Result
    ------
    Matrix of shape n, m consisting of each pixel's dominant frequency
    """

    # Number of frames
    N = vid.shape[0]

    # Mean centering
    combined_frames_0mean = vid - vid.mean(axis=0)

    # Fft of centered data
    combined_frames_fft = np.fft.fft(combined_frames_0mean, axis=0)

    # Frequency array
    freq = np.arange(N//2 + 1, dtype='uint8')

    # Saving only half the frequencies as fft gives symmetric results
    combined_frames_abs = np.absolute(combined_frames_fft[:N//2 + 1])

    # Getting dominant frequency's index
    max_freq_indices = combined_frames_abs.argmax(axis=0)

    # Filtering and storing results
    results = freq[max_freq_indices]
    results_filtered = signal.medfilt2d(results, f_size)

    return results_filtered


def get_optical_flow(video):
    """
    This is for computing the optical flow of a given video.

    Parameters
    ----------
    vid, a matrix shape f,n,m where f is the number of frames size n by m

    Return
    ------
    final: final aray matrix n x m
    """

    # storing the first frame of the video
    initial_frame = video[0]

    # converting it into proper format
    hsv = np.zeros_like(initial_frame)
    hsv = np.expand_dims(hsv, axis=2)
    rgb_frame = cv2.cvtColor(initial_frame, cv2.COLOR_GRAY2RGB)
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = 255
    final = 0
    # optical flow code reference:https://docs.opencv.org/3.4/d7/d8b/
    #   tutorial_py_lucas_kanade.html
    for next_frame in video[1:]:
        previous_frame = initial_frame
        flow = cv2.calcOpticalFlowFarneback(previous_frame, next_frame, None,
                                            0.5, 3, 5, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        gaussian_smoothing = cv2.blur(bgr, (5, 5))
        thresh = cv2.threshold(gaussian_smoothing, 66, 255,
                               cv2.THRESH_BINARY)[1]
        gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

        # storing the value convert value in final aray
        final += np.asarray(gray)
        previous_frame = next_frame

    return final


def stack_var_optic():
    """
    Stack the variance and optical flow into a 3D array

    Parameters
    ----------
    vid, a matrix shape f,n,m where f is the number of frames size n by m.
    f_size, the dimensions of the filter used in fft.

    Result
    ------
    Matrix of shape frame_width, frame_heigth, number of features
    """
    # Return both features as multichannel
    return np.dstack((get_variance_as_im(vid), get_optical_flow(vid)))


def stack_var_freq():
    """
    Stack the variance and frequency into a 3D array

    Parameters
    ----------
    vid, a matrix shape f,n,m where f is the number of frames size n by m.
    f_size, the dimensions of the filter used in fft.

    Result
    ------
    Matrix of shape frame_width, frame_heigth, number of features
    """
    # Return both features as multichannel
    return np.dstack((get_variance_as_im(vid), get_beat_frequency(vid)))


def stack_freq_opt():
    """
    Stack the frequency and optical flow into a 3D array

    Parameters
    ----------
    vid, a matrix shape f,n,m where f is the number of frames size n by m.
    f_size, the dimensions of the filter used in fft.

    Result
    ------
    Matrix of shape frame_width, frame_heigth, number of features
    """
    # Return both features as multichannel
    return np.dstack((get_beat_frequency(vid), get_optical_flow(vid)))


def get_features(vid, f_size=5):
    """
    Stack the features into a 3D array

    Parameters
    ----------
    vid, a matrix shape f,n,m where f is the number of frames size n by m.
    f_size, the dimensions of the filter used in fft.

    Result
    ------
    Matrix of shape frame_width, frame_heigth, number of features
    """
    # Return both features as multichannel
    return np.dstack((get_beat_frequency(vid), get_variance_as_im(vid),
                      get_optical_flow(vid)))


def save_helper(dispatch, filename, feature, outfile, add_greyscale):
    """
    saves feature map as determined by command line args

    Parameters
    ----------
    dispatch: the function to be called on numpy
    filename: location of the numpy file
    feature: name of feature being created
    outfile: directory to save to

    Return
    ------
    None
    """
    vid = np.load(filename)
    out = dispatch(vid)
    if(add_greyscale):
        out = np.dstack((out, vid[0]))
    key = filename.split(os.path.sep)[-1].split(".")[0]
    fname = "{}_{}.npy".format(key, feature)
    outfile = os.path.join(outfile, fname)
    np.save(outfile, out)
    return()


if __name__ == "__main__":
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(
        description=('Reads all npy files from video directory and ',
                     'creates an npy file for each containing a feature map'),
        add_help='How to use', prog='png_to_npy.py <args>')

    # Required arguments
    parser.add_argument("-i", "--input", required=True,
                        help=("path to the directory containing videos as npy",
                              " files"))

    # optional arguments
    parser.add_argument("-o", "--output", default=os.path.join(cwd, "videos"),
                        help=("The destination to store all of the npy files ",
                              "[default: cwd/videos]"))
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help=('degree of parallelism to use -1 will use all ',
                              'present cores[default: -1]'))
    parser.add_argument("--feature", "-f", default='variance',
                        help=('if set to "variance" it will compute variance.',
                              'if set to "frequency" it will comput beat.',
                              'frequencies. if set to "optic" it will create',
                              'optical flow features. if set to "var-freq" it',
                              'will create a feature map of variance and freq',
                              '. If set to "var-opt" it will create a feature',
                              ' map of variance and optical flow. If set to ',
                              '"freq-opt" it will create a feature map of ',
                              'freq and optical flow. If set to "all" it ',
                              'will create a multichannel feature map of all ',
                              'features.'))
    parser.add_argument("--add_greyscale", "-g", action='store_true',
                        help="if set the first frame will be added to output")

    # parse input and output arguments
    args = vars(parser.parse_args())
    if not os.path.exists(args['output']):
        os.mkdir(args['output'])

    file_list = glob.glob(args['input']+'*')

    dispatch = {'variance': get_variance_as_im,
                'frequency': get_beat_frequency,
                'optic': get_optical_flow,
                'var-opt': stack_var_optic,
                'var-freq': stack_var_freq,
                'freq-opt': stack_freq_opt,
                'all': get_features}
    # run over all input in parallel
    out = joblib.Parallel(n_jobs=args['n_jobs'], verbose=10,)(
          joblib.delayed(save_helper)(dispatch[args['feature']], f,
                                      args['feature'], args['output'],
                                      args['add_greyscale'])
          for f in file_list
    )
