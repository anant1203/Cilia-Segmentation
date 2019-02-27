import numpy as np
import numpy as np
import skimage.io as io
import scipy.signal as signal

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


def get_beat_frequency(vid, f_size=5):
    """Perform fft on a video(100 frames) and extract the dominant frequency.
    Filtering is performed to reduce the effects of noise.
    
    Parameters:
    vid, a matrix shape f,n,m where f is the number of frames size n by m.
    f_size, the dimensions of the filter.
    
    Result:
    Matrix of shape n, m consisting of each pixel's dominant frequency
    """
    
    
    #Number of frames
    N = vid.shape[0]

   
    
    #Mean centering
    combined_frames_0mean = vid - vid.mean(axis=0)
    
    #Fft of centered data
    combined_frames_fft = np.fft.fft(combined_frames_0mean, axis = 0)
    
    #Frequency array
    freq = np.arange(N//2 + 1, dtype = 'uint8')
    
    #Saving only half the frequencies as fft gives symmetric results 
    combined_frames_abs = np.absolute(combined_frames_fft[:N//2 + 1])
    
    #Getting dominant frequency's index
    max_freq_indices = combined_frames_abs.argmax(axis = 0)
    
    #Filtering and storing results
    results = freq[max_freq_indices]
    results_filtered = signal.medfilt2d(results, f_size)

    return results_filtered

# def get_op_flow(vid):

def get_features(vid, f_size=5):
    """Stack the features into a 3D array
    
    Parameters:
    vid, a matrix shape f,n,m where f is the number of frames size n by m.
    f_size, the dimensions of the filter used in fft.
    
    Result:
    Matrix of shape frame_width, frame_heigth, number of features
    """
    #return np.dstack((get_beat_frequency(vid)[0], get_variance_as_im(vid)[0], get_op_flow(vid)[0]))   
    return np.dstack((get_beat_frequency(vid)[0], get_variance_as_im(vid)[0]))
        
        
