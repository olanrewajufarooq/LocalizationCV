import numpy as np
from scipy.io import loadmat, savemat
import PIL
import cv2
import argparse

# Class for Computing Homography
class Homography:
    """This is a class strictly for computing homoegraphy.
    
    It involves a function to compute homography between only two images.
    Either with or without RANSAC.    
    """
    
    def __init__(self):
        pass
    
    def compute_homography(self):
        pass
    
    def run_ransac(self):
        pass
    
# Class for Extracting Features in an Image
class FeatureExtraction:
    """This is a class to perform feature extraction on images
    """
    def __init__(self):
        pass

# Class for Running Solutions
class Solution:
    def __init__(self, config_path):
        self.config_path = config_path
        
        self.data_reading = {}
        self.read_config()
        
        self.config_dict = {}
        self.parse_config_data()
        
    def read_config(self):
        """Reading the config file using the code provided by instructor.
        """
        
        with open(self.config_path, 'r') as file:
            for line in file:
                line = line.strip()

                # Ignore comments
                if line.startswith('#') or len(line) == 0:
                    continue

                # Split the line into tokens
                tokens = line.split()
                
                print(tokens)

                # Extract parameter names and values
                param_name = tokens[0]
                param_values = [tokens[1:]]

                # Check if the token already exists in the dictionary
                if param_name in self.data_reading:
                    # Add new values to the existing token
                    self.data_reading[param_name].extend(param_values)
                else:
                    # Create a new entry in the dictionary
                    self.data_reading[param_name] = param_values
    
    def parse_config_data(self):
        # Parsing Video
        self.config_dict["videos"] = self.data_reading["videos"][0]

        # Parsing Points in Maps and Features
        self.config_dict["map_dtypes"] = []
        self.config_dict["frame_ids"] = []

        self.config_dict["map_points"] = []
        self.config_dict["frame_points"] = []

        assert len(self.data_reading["pts_in_map"]) == len(self.data_reading["pts_in_frame"]), f"Non-Equivalent Amount of Data for Map and Features. \nGiven: Map = {len(self.data_reading['pts_in_map'])}, Features = {len(self.data_reading['pts_in_frame'])}"

        for pt_in_map, pt_in_frame in zip(self.data_reading["pts_in_map"], self.data_reading["pts_in_frame"]):
            dtype = pt_in_map[0]
            frame_id = pt_in_frame[0]
            
            # assert (type(int(pt_in_frame[0])) == int) and (float(pt_in_frame[0]) == int(pt_in_frame[0])), f"The Frame ID should be an integer. Given: {pt_in_frame[0]}"
            
            map_points = []
            frame_points = []
            
            count = 0
            
            for p_map, p_frame in zip(pt_in_map[1:], pt_in_frame[1:]):
                # Implemeting the code for pixels
                if (count % 2 == 0) and (dtype == "pixel"):
                    x_map = int(p_map)
                    x_frame = int(p_frame)
                elif (count % 2 != 0) and (dtype == "pixel"):
                    y_map = int(p_map)
                    y_frame = int(p_frame)
                    
                    map_points.append( (x_map, y_map) )
                    frame_points.append( (x_frame, y_frame) )
                # TO DO: Codes for other data types can be implemented here later. For example: distances.
                
                count += 1
                
            assert len(map_points) == len(frame_points), "Amount of map points and amounts of frame points are inconsistent."
            
            self.config_dict["map_dtypes"].append(dtype)
            self.config_dict["frame_ids"].append(frame_id)

            self.config_dict["map_points"].append(map_points)
            self.config_dict["frame_points"].append(frame_points)

        # Parsing the Map Image
        self.config_dict["image_map"] = self.data_reading["image_map"][0][0]
        self.config_dict["keypoints_out"] = self.data_reading["keypoints_out"][0][0]
        self.config_dict["transforms_out"] = self.data_reading["transforms_out"][0][0]

        self.config_dict["transforms"] = self.data_reading["transforms"][0][1]


# Function to parse arguments from Command Line
def parse_args():
    """Reading arguments parsed on the command line.

    Returns:
        args: output the arguments parsed.
    """
    parser = argparse.ArgumentParser(
        description='Image or Video Processing.')
    
    parser.add_argument('config_file_path', type=str,
                        help='path of the configuration file')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args() #Read arguments passed on the command line
    
    sol = Solution(args.config_file_path)