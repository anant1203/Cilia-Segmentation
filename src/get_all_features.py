import feature_engineering
import numpy as np

def read_file(file):
    
    """Read the train/test file names and store them in a list.
    
    Parameters:
    path to file
    
    Result:
    file names in a list
    """
    with open(file, "r") as ins:
        array = []
        for line in ins:
            array.append(line.rstrip('\n'))
    return array

def feature_extraction(data_path, file_name_path):
    
    """Loop over the train/test data and extract features
    
    Parameters:
    path to data folder, path to file names file (train.txt / test.txt)
    
    Result:
    Train / test features as a 4D numpy array of shape
    (num_samples, img_width, img_height, number_of_features)
    
    Note: If img_width and img_heigth varies, it will return a 1D numpy array of shape
    (num_samples, ) 
    """
    
    features = []
    file_names = read_file(file_name_path)
    for file in file_names:
        vid = np.load(data_path+file)
        features.append(feature_engineering.get_features(vid))
    return np.array(features)
