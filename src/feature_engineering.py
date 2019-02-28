import numpy as np
import scipy.signal as signal

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
    return np.dstack((get_beat_frequency(vid), get_variance_as_im(vid)))


def save_helper(dispatch, filename, feature, outfile):
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
    key = filename.split(os.path.sep)[-1].split(".")[0]
    fname = "{}_{}.npy".format(key, feature)
    outfile = os.path.join(outfile, fname)
    np.save(outfile, out)
    return()


if __name__ == "__main__":
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(
        description=('Reads all png files from video subdirectories and ',
                     'creates an npy file for each subdirectory'),
        add_help='How to use', prog='png_to_npy.py <args>')

    # Required arguments
    parser.add_argument("-i", "--input", required=True,
                        help=("path to the directory containing video ",
                              "subdirectories"))

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
    parser.add_argument("-s", "--save_inproc", action="store_true",
                        help=("if true data will be written in time (good for",
                              " large datasets)"))

    # parse input and output arguments
    args = vars(parser.parse_args())
    if not os.path.exists(args['output']):
        os.mkdir(args['output'])

    file_list = glob.glob(args['input']+'*')

    dispatch = {'variance': get_variance_as_im,
                'frequency': get_beat_frequency,
                'optic': get_optical_flow,
                'var-opt': stack-var-optic,
                'var-freq': stack-var-freq,
                'freq-opt': stack-freq-opt,
                'all': get_features}
    # run over all input in parallel
    out = joblib.Parallel(n_jobs=args['n_jobs'], verbose=10,)(
          joblib.delayed(save_helper)(dispatch[args['feature']], f,
                                      args['feature'], args['output'])
          for f in file_list
    )
