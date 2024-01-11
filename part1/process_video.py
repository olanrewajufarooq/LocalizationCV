from src_file import parse_args, ConfigParser, FeatureExtraction

# verbose = True
# visualize = False
# config_path = "myconfig.cfg"

# parser = ConfigParser(config_path = config_path, verbose=verbose)

cmd_args = parse_args() #Read arguments passed on the command line
parser = ConfigParser(config_path = cmd_args.config_file_path, verbose=cmd_args.verbose)

# Feature Extraction
feat_extract = FeatureExtraction(method="SIFT", blur=True, verbose=cmd_args.verbose, visualize=cmd_args.visualize)
features, frame_ids = feat_extract.extract_features(parser.config_dict["videos"], # Extracting Features from video files (Only One Video Accepted)
                                                    type="video",
                                                    n_desired_frames=0, # Number of desired frames to extract features from, 0 for all frames
                                                    visualization_delay=0.0001, # Visualization delay in seconds
                                                    )

parser.save_features(features, frame_ids)
