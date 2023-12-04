import argparse
import pickle
import h5py
import numpy as np
import PIL
import cv2
from scipy.io import loadmat, savemat
from sklearn.neighbors import NearestNeighbors

# Class for Computing Homography
class Homography:
    """This is a class strictly for computing homoegraphy.
    
    It involves a function to compute homography between only two images.
    Either with or without RANSAC.
    """
    
    def __init__(self, transforms = "map", use_ransac = True, use_opencv=True, verbose=False, visualize=False):
        self.transforms = transforms
        self.use_ransac = use_ransac
        self.verbose = verbose
        self.use_opencv = use_opencv
        self.visualize = visualize
        
        if self.use_ransac:
            self.n_iterations = 35 # TODO: Implement the code to calculate the number of iterations.

    def visualize_homographies(self, map_image_path, video_path, homography_matrices, frame_ids):
        if self.visualize:
            print("Visualizing Homographies...", end=" ")

            # Load the map image
            map_image = cv2.imread(map_image_path)

            # Open the video file
            video = cv2.VideoCapture(video_path)
            assert video.isOpened(), f"Video not opened. Filepath: {video_path}"

            for frame_id in frame_ids:
                # Find the column in the homography matrix corresponding to this frame_id
                col_index = np.where(homography_matrices[1] == frame_id)[0][0]
                
                # Extract the 3x3 homography matrix
                H = homography_matrices[2:, col_index].reshape(3, 3)

                # Set the video to the specific frame
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                
                # Read the frame
                ret, frame_image = video.read()
                assert ret, f"Failed to read frame at ID {frame_id}"

                # Apply homography to transform the frame image
                transformed_image = cv2.warpPerspective(frame_image, H, (map_image.shape[1], map_image.shape[0]))

                # Create an image to draw on
                height = max(map_image.shape[0], transformed_image.shape[0])
                width = map_image.shape[1] + transformed_image.shape[1]
                output_image = np.zeros((height, width, 3), dtype=np.uint8)

                # Place the map and transformed frame side by side in the output image
                output_image[:map_image.shape[0], :map_image.shape[1]] = map_image
                output_image[:transformed_image.shape[0], map_image.shape[1]:] = transformed_image

                # Show the output image
                cv2.imshow(f"Homography - Frame {frame_id}", output_image)
                cv2.waitKey(0)

            video.release()
            cv2.destroyAllWindows()
            print("Done")
            
    def _compute_homography_without_cv(self, input_points, output_points):
        
        # Checking for consistency in the amount of points
        assert (len(input_points) >= 4) and (len(output_points) >= 4), "At least 4 points are required to compute homography."
        assert len(input_points) == len(output_points), "Point arrays must have the same number of points."

        # Build the matrix A for the homography calculation
        A = []
        
        for point1, point2 in zip(input_points, output_points):
            x_in, y_in = point1
            x_out, y_out = point2
            A.append([x_in , y_in, 1, 0, 0, 0, -x_out*x_in, -x_out*y_in, -x_out])
            A.append([0, 0, 0, x_in, y_in, 1, -y_out*x_in, -y_out*y_in, -y_out])
        
        A = np.array(A,dtype=np.double)
        eig_values , eig_vectors = np.linalg.eig(A.T @ A)

        min_eig_value = np.argmin(eig_values)
        min_eig_value_index = np.unravel_index(min_eig_value, eig_values.shape)
        H_array = eig_vectors[:,min_eig_value_index].reshape(-1,1)
        
        return H_array / H_array[-1]
    
    def _run_ransac(self, input_points, output_points):
        # Checking for consistency in the amount of points
        assert (len(input_points) >= 4) and (len(output_points) >= 4), "At least 4 points are required to compute homography."
        assert len(input_points) == len(output_points), "Point arrays must have the same number of points."
        
        n_inliers = []
        inliers_idx = []
        
        for _ in range(self.n_iterations):
            # Randomly select 4 points from the input and output points
            random_indices = np.random.choice(len(input_points), 4, replace=False)
            input_points_random = np.array(input_points, dtype=object)[random_indices].tolist()
            output_points_random = np.array(output_points, dtype=object)[random_indices].tolist()
            
            # Compute the homography using the 4 points
            H = self._compute_homography_without_cv(input_points_random, output_points_random)
            
            # Compute the number of inliers
            n_inlier = 0
            inlier_idx = []
            for i, (input_point, output_point) in enumerate(zip(input_points, output_points)):
                input_point = np.append(input_point, 1)
                output_point = np.append(output_point, 1)
                output_point_pred = H.reshape(3, 3) @ input_point
                output_point_pred /= output_point_pred[-1]
                
                if np.linalg.norm(output_point_pred - output_point) < 5:
                    n_inlier += 1
                    inlier_idx.append( i )
                
            n_inliers.append(n_inlier)
            inliers_idx.append(inlier_idx)
        
        # Select the best homography
        best_homo_index = np.argmax(n_inliers)
        best_points = inliers_idx[best_homo_index]
        
        best_inputs = np.array(input_points)[best_points].tolist()
        best_outputs = np.array(output_points)[best_points].tolist()
        
        best_H_array = self._compute_homography_without_cv(best_inputs, best_outputs)
        
        return best_H_array
    
    def _compute_homography_map(self, pts_in_map, pts_in_frame, frame_ids):
        
        H_output_arrays = []
        
        for output_points, input_points, frame_id in zip(pts_in_map, pts_in_frame, frame_ids):
            if self.use_opencv:
                # TODO: Implement OpenCV here.
                pass
            else:
                # Computing the homography matrix without using OpenCV
                if self.use_ransac:
                    H_array = self._run_ransac(input_points=input_points, output_points=output_points)
                else:
                    H_array = self._compute_homography_without_cv(input_points=input_points, output_points=output_points)
                
                assert H_array.shape[1] == 1, f"Output should be a column array. \nGotten: {H_array}"
                
                # Stacking the frame id to the homography array
                H_output_array = np.vstack( (np.array([0, frame_id]).reshape(-1, 1), H_array) )
                H_output_arrays.append(H_output_array)
        
        # Stacking the homography outputs.
        H_output = np.hstack(H_output_arrays)
        
        return H_output
    
    def compute_homography(self, pts_in_map, pts_in_frame, frame_ids):
        if self.verbose:
            print(f"Computing Homography. Use Ransac: {self.use_ransac}. Use OpenCV: {self.use_opencv}. Type: {self.transforms}...", end=" ")
        
        if self.transforms == "map":
            H_output = self._compute_homography_map(pts_in_map, pts_in_frame, frame_ids)
        elif self.transforms == "all":
            # TODO: Implement the code to compute homography for all frames.
            pass
        
        if self.verbose:
            print("Successful")
        
        return H_output


# Class for Extracting Features in an Image
class FeatureExtraction:
    """This is a class to perform feature extraction on images
    """
    def __init__(self, method = "SIFT", n_keypoints=0, verbose=False, visualize=False):
        self.method = method
        self.n_keypoints = n_keypoints
        self.verbose = verbose
        self.visualize = visualize
        
        if self.method == "SIFT":
            if self.verbose:
                print("Initializing SIFT Feature Detector... ", end=" ")
            
            # Initialize SIFT detector
            self.extractor = cv2.SIFT_create(
                nfeatures=n_keypoints,  # Set to 0 to disable limiting the number of keypoints
                # nOctaveLayers=3, # Number of layers in each octave of the image. Default: 3. (Automatically determined by image size)
                contrastThreshold=0.04, # If contrast of keypoint below this threshold, won't be detected. Default: 0.04
                edgeThreshold=10, #  If difference in intensity between keypoint and surrounding pixels below threshold, keypoint rejected. Default: 10
                sigma=1.6 # S.D. of Gaussian on first octave. Default: 1.6
            )
            
            if self.verbose:
                print("Successful")
    
    def _extract_features_sift(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints_main, descriptors = self.extractor.detectAndCompute(gray, None)
        
        # Extracting the X and Y points from the KeyPoint Object
        keypoints = np.array( [ (kp.pt[0], kp.pt[1]) for kp in keypoints_main ] )
        
        # Verification of Data
        assert keypoints.shape[0] == descriptors.shape[0], "Inconsistency. Amount of features must be equal to the amount of descriptors" + \
            f"\n Obtained: Keypoint Shape - {keypoints.shape}. Descriptors Shape - {descriptors.shape} "
        assert keypoints.shape[1] == 2, f"Inconsistency in the size of keypoints. Obtained: {keypoints.shape}"
        assert descriptors.shape[1] == 128, f"Inconsistency in the size of descriptors. Obtained: {descriptors.shape}"

        # #visualizing the features
        if self.visualize:
            image_with_keypoints = cv2.drawKeypoints(gray, keypoints_main, None, color=(0, 255, 0),)
            cv2.imshow('frame', image_with_keypoints)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return np.vstack((keypoints.T, descriptors.T))
    
    def extract_features(self, file_path, type="image"):
        
        if self.verbose:
            if type == "video":
                print(f"Loading video ({file_path}) to memory...", end=" ")
            elif type == "image":  
                print(f"Loading image ({file_path}) to memory...", end=" ")
        
        # Reading the file
        if type == "video":
            video = cv2.VideoCapture(file_path)
            assert video.isOpened(), f"Video not opened. Filepath: {file_path}"
            n_video_frames = int( video.get(cv2.CAP_PROP_FRAME_COUNT) )
            
            # Specify amount of desired frames
            n_desired_frames = 20
            
            # Calculating the divisor to obtain the desired amount of frames
            frame_ids = np.linspace(0, n_video_frames-1, n_desired_frames, dtype=int)
            frame = 0 # Initializing the frame counter
            
        elif type == "image":
            image = cv2.imread(file_path)
        
        if self.verbose:
            print("Successful")
            if type == "video":
                frame_rate = video.get(cv2.CAP_PROP_FPS)
                print(f"Number of Frames in Video: {n_video_frames}. Frame Rate: {frame_rate}")
            print("Extracting features from frames...", end=" ")
        
        features = []
        
        while True:
            
            # If the file is a video, break the loop after all frames are read
            if (type == "video"):
                
                if (frame == n_desired_frames) :
                    break
                
                # Setting the frame to the next desired frame
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_ids[frame])
                frame += 1
                
                # Reading the frame
                ret, image = video.read()
                if (not ret):
                    break
            
            # Performing Feature Extraction
            if self.method == "SIFT":
                features.append( self._extract_features_sift(image) )
            
            # If the file is an image, break the loop after one iteration
            if type == "image":
                break
            
            # Press 'q' to exit the loop
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        
        # Checking for consistency in the amount of features obtained
        if type == "video":
            assert len(features) == n_desired_frames, f"Inconsistency: Amount of features obtained ({len(features)}) does not equate the amount of desired frames from video ({n_desired_frames})."
            
        if self.verbose:
            print("Successful")

        # Closing opened files
        if type == "video":
            video.release()
        
        cv2.destroyAllWindows()
        
        if type == "video":
            return features, frame_ids
        elif type == "image":
            return features


# Class for Feature Matching
class FeatureMatching:
    def __init__(self, lib="opencv", verbose=False, visualize=False):
        self.lib = lib # Options: "opencv", "sklearn"
        self.verbose = verbose
        self.visualize = visualize

    def _visualize_matching(self, map_image_path, video_path, pts_in_map, pts_in_frame):
        if self.visualize:
            print("Visualizing the matches...", end=" ")

            # Load the map image
            map_image = cv2.imread(map_image_path)

            # Open the video file
            video = cv2.VideoCapture(video_path)
            assert video.isOpened(), f"Video not opened. Filepath: {video_path}"

            n_video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            n_desired_frames = 20
            frame_ids = np.linspace(0, n_video_frames - 1, n_desired_frames, dtype=int)

            for frame_id in frame_ids:
                # Set the video to the specific frame
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                
                # Read the frame
                ret, frame_image = video.read()
                assert ret, f"Failed to read frame at ID {frame_id}"

                # Create an image to draw on
                height = max(map_image.shape[0], frame_image.shape[0])
                width = map_image.shape[1] + frame_image.shape[1]
                output_image = np.zeros((height, width, 3), dtype=np.uint8)

                # Place the map and frame side by side in the output image
                output_image[:map_image.shape[0], :map_image.shape[1]] = map_image
                output_image[:frame_image.shape[0], map_image.shape[1]:] = frame_image

                # Draw lines between matched points for the current frame
                for pt_map, pt_frame in zip(pts_in_map[frame_id % n_desired_frames], pts_in_frame[frame_id % n_desired_frames]):
                    pt_map = (int(pt_map[0]), int(pt_map[1]))
                    pt_frame = (int(pt_frame[0] + map_image.shape[1]), int(pt_frame[1]))
                    cv2.line(output_image, pt_map, pt_frame, color=(0, 255, 0), thickness=1)

                # Show the output image
                cv2.imshow(f"Matches - Frame {frame_id}", output_image)
                cv2.waitKey(0)

            video.release()
            cv2.destroyAllWindows()
            print("Done")

    def match_features(self, features, map="first", match_threshold=0.75):
        if self.verbose:
            print("Matching features...", end=" ")
        
        # Extracting the features of the map and frames
        if map == "first":
            features_map = features[0]
            features_frames = features[1:]
        elif map == "last":
            features_map = features[-1]
            features_frames = features[:-1]
        
        if self.lib == "opencv":
            pts_in_map, pts_in_frame, features_match_objects = self._cv_feature_matching(features_map, features_frames, match_threshold=match_threshold)
        elif self.lib == "sklearn":
            pts_in_map, pts_in_frame = self._knn_feature_matching(features_map, features_frames, match_threshold=match_threshold)
        
        assert len(pts_in_map) == len(pts_in_frame), f"Inconsistency in matching algorithm. Expected length equal length \nMap: {len(pts_in_map)}. Frame: {len(pts_in_frame)}"

        if self.verbose:
            print("Successful")
 
        return pts_in_map, pts_in_frame
    
    def _knn_feature_matching(self, features_map, features_frames, match_threshold=0.75):
        map_descriptor = features_map[2:].T.astype(np.uint8)
        
        pts_in_map = []
        pts_in_frame = []
        
        knn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(map_descriptor)
            
        for feature in features_frames:
                
                # Extracting the descriptors of the frame
                frame_descriptor = feature[2:].T.astype(np.uint8)
                
                # Checking for consistencies in the descriptors
                assert map_descriptor.shape[1] == frame_descriptor.shape[1], f"Inconsistency in the size of descriptor. \nMap: {map_descriptor.shape[1]}. Frame: {frame_descriptor.shape[1]}"
                
                distances, indices = knn.kneighbors(frame_descriptor)
                
                accepted_pt_in_map = []
                accepted_pt_in_frame = []
                
                for i, d in enumerate(distances):
                    if d[0] < match_threshold * d[1]:
                        accepted_pt_in_map.append( tuple(features_map[:2, indices[i][0]]) ) # indices[i][0] is the index of the best match for the i-th feature in the frame
                        accepted_pt_in_frame.append( tuple(feature[:2, i]) ) # i is the index of the i-th feature in the frame
                
                pts_in_map.append( accepted_pt_in_map )
                pts_in_frame.append( accepted_pt_in_frame )
        
        return pts_in_map, pts_in_frame
    
    def _cv_feature_matching(self, features_map, features_frames, match_threshold=0.75):
        # Extracting the keypoints and descriptors of the map
        map_descriptor = features_map[2:].T.astype(np.uint8)
        
        pts_in_map = []
        pts_in_frame = []
        
        bf = cv2.BFMatcher() # Brute Force Matcher
        
        features_match_objects = [ ]
            
        for feature in features_frames:
            
            # Extracting the descriptors of the frame
            frame_descriptor = feature[2:].T.astype(np.uint8)
            
            # Checking for consistencies in the descriptors
            assert map_descriptor.shape[1] == frame_descriptor.shape[1], f"Inconsistency in the size of descriptor. \nMap: {map_descriptor.shape[1]}. Frame: {frame_descriptor.shape[1]}"
            
            matches = bf.knnMatch(map_descriptor, frame_descriptor, k=2) # map_descriptor is the query descriptor and frame_descriptor is the train descriptor.
            # The knnMatch function returns the two best matches for each descriptor.
            # Hence, later we will need to filter the matches to only keep the best ones.
            
            accepted_pt_in_map = []
            accepted_pt_in_frame = []
            
            accepted_matches = []
            
            for m, n in matches: # m and n are the two best matches from the frame descriptor
                if m.distance < match_threshold * n.distance:
                    accepted_pt_in_map.append( tuple(features_map[:2, m.queryIdx]) )
                    accepted_pt_in_frame.append( tuple(feature[:2, m.trainIdx]) )
                    accepted_matches.append(m)
            
            pts_in_map.append( accepted_pt_in_map )
            pts_in_frame.append( accepted_pt_in_frame )
            features_match_objects.append(accepted_matches)
        
        return pts_in_map, pts_in_frame, features_match_objects


# Class for Running Solutions and Parsing Data
class ConfigParser:
    def __init__(self, config_path, verbose=False):
        self.config_path = config_path
        self.verbose = verbose
        
        if self.verbose:
            print("Reading config file...", end=" ")
        
        self.data_reading = {}
        self.read_config()
        
        self.config_dict = {}
        self.parse_config_data()
        
        if self.verbose:
            print("Successful",)
    
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
        """Parsing the read data to a format usable for the project.
        """
        # Parsing Video Filenames
        self.config_dict["videos"] = self.data_reading["videos"][0][0]

        # Parsing Points in Maps and Features
        self.config_dict["map_dtypes"] = []
        self.config_dict["frame_ids"] = []

        self.config_dict["map_points"] = []
        self.config_dict["frame_points"] = []

        assert len(self.data_reading["pts_in_map"]) == len(self.data_reading["pts_in_frame"]), f"Non-Equivalent Amount of Data for Map and Features. \nGiven: Map = {len(self.data_reading['pts_in_map'])}, Features = {len(self.data_reading['pts_in_frame'])}"

        for pt_in_map, pt_in_frame in zip(self.data_reading["pts_in_map"], self.data_reading["pts_in_frame"]):
            dtype = pt_in_map[0]
            frame_id = pt_in_frame[0]
            
            assert (type(int(pt_in_frame[0])) == int) and (float(pt_in_frame[0]) == int(pt_in_frame[0])), f"The Frame ID should be an integer. Given: {pt_in_frame[0]}"
            
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
                # TODO: Codes for other data types can be implemented here later. For example: distances.
                
                count += 1
                
            assert len(map_points) == len(frame_points), "Amount of map points and amounts of frame points are inconsistent."
            
            self.config_dict["map_dtypes"].append(dtype)
            self.config_dict["frame_ids"].append(frame_id)

            self.config_dict["map_points"].append(map_points)
            self.config_dict["frame_points"].append(frame_points)

        # Parsing Other Parameters
        self.config_dict["image_map"] = self.data_reading["image_map"][0][0]
        
        # Parsing the Keypoint Output Path and Extension
        keypoints_out= self.data_reading["keypoints_out"][0][0].split(".")
        self.config_dict["keypoints_out_path"] = keypoints_out[0]
        self.config_dict["keypoints_out_ext"] = keypoints_out[1]
        
        # Parsing the Transformation Output Path and Extension
        transforms_out= self.data_reading["transforms_out"][0][0].split(".")
        self.config_dict["transforms_out_path"] = transforms_out[0]
        self.config_dict["transforms_out_ext"] = transforms_out[1]

        # Saving the specified Transformation Method
        self.config_dict["transforms"] = self.data_reading["transforms"][0][1]
    
    def save_features(self, features):
        
        if self.verbose:
            print(f"Saving features as .{self.config_dict['keypoints_out_ext']} file...", end=" ")
            
        # Saving as MATLAB file
        if self.config_dict["keypoints_out_ext"] == "mat":
            savemat( f"{self.config_dict['keypoints_out_path']}.mat", {"features": features})
        
        # Saving as HDF5 file
        elif self.config_dict["keypoints_out_ext"] == "h5":
            with h5py.File(f"{self.config_dict['keypoints_out_path']}.h5", 'w') as file:
                for frame, feature in enumerate(features):
                    file.create_dataset(f"{frame}", data=feature)
        
        # Saving as Pickle file
        elif self.config_dict["keypoints_out_ext"] == "pkl":
            with open(f"{self.config_dict['keypoints_out_path']}.pkl", 'wb') as file:
                pickle.dump(features, file)
        
        if self.verbose:
            print("Successful")
            
    def load_features(self):
            
        if self.verbose:
            print(f"Loading features from .{self.config_dict['keypoints_out_ext']} file...", end=" ")
        
        # Loading from MATLAB file
        if self.config_dict["keypoints_out_ext"] == "mat":
            feats = loadmat( self.config_dict['keypoints_out_path'] )
        
        # Loading from HDF5 file
        elif self.config_dict["keypoints_out_ext"] == "h5":
            with h5py.File(f"{self.config_dict['keypoints_out_path']}.h5", 'r') as file:
                feats = []
                for frame in file:
                    feats.append( file[frame][()] )
        
        # Loading from Pickle file
        elif self.config_dict["keypoints_out_ext"] == "pkl":
            with open(f"{self.config_dict['keypoints_out_path']}.pkl", 'rb') as file:
                feats = pickle.load(file)
        
        if self.verbose:
            print("Successful")
        
        return feats
    
    def save_homography_output(self, homo_matrix):
        
        if self.verbose:
            print(f"Saving homography matrices as .{self.config_dict['transforms_out_ext']} file...", end=" ")
            
        # Saving as MATLAB file
        if self.config_dict["transforms_out_ext"] == "mat":
            savemat( f"{self.config_dict['transforms_out_path']}.mat", {"H": homo_matrix} )
        
        # Saving as HDF5 file
        elif self.config_dict["transforms_out_ext"] == "h5":
            with h5py.File(f"{self.config_dict['transforms_out_path']}.h5", 'w') as file:
                file.create_dataset("H", data=homo_matrix)
        
        # Saving as Pickle file
        elif self.config_dict["transforms_out_ext"] == "pkl":
            with open(f"{self.config_dict['transforms_out_path']}.pkl", 'wb') as file:
                pickle.dump(homo_matrix, file)
        
        if self.verbose:
            print("Successful")
        


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
    
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                        help='enable verbose mode')
    
    parser.add_argument('--visualize', '-o', action='store_true', default=False,
                        help='enable plotting mode')
    
    return parser.parse_args()

if __name__ == '__main__':
    cmd_args = parse_args() #Read arguments passed on the command line
    parser = ConfigParser(config_path = cmd_args.config_file_path, verbose=cmd_args.verbose)

    # Initializing the Feature Extraction
    feat_extract = FeatureExtraction(method="SIFT", verbose=cmd_args.verbose, visualize=cmd_args.visualize)
    
    # Extracting Features from Image Map
    features = feat_extract.extract_features(parser.config_dict["image_map"], type="image")
    
    # Extracting Features from video files (Only One Video Accepted)
    features_video, frame_ids = feat_extract.extract_features(parser.config_dict["videos"], type="video")
    features.extend(features_video)
    frame_ids = frame_ids + 1 # Adding 1 to the frame ids to account for the map frame
    
    parser.save_features(features)
    # parser.load_features()

    # Matching Features
    feat_match = FeatureMatching(lib="sklearn", verbose=cmd_args.verbose, visualize=cmd_args.visualize)
    pts_in_map, pts_in_frame = feat_match.match_features(features)

    #Visualize  the matches
    feat_match._visualize_matching(parser.config_dict["image_map"], parser.config_dict["videos"], pts_in_map, pts_in_frame)

    # Computing Homography
    homography = Homography(transforms="map", use_ransac=True, use_opencv=False, verbose=cmd_args.verbose, visualize=cmd_args.visualize)
    homo_output_matrix = homography.compute_homography( pts_in_map, pts_in_frame, frame_ids )
    parser.save_homography_output(homo_output_matrix)

    #visualize homography
    homography.visualize_homographies(parser.config_dict["image_map"], parser.config_dict["videos"], homo_output_matrix, frame_ids)