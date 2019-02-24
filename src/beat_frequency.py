import numpy as np
import skimage.io as io
import scipy.signal as signal

def get_beat_frequency(vid, f_size=15):
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
    
