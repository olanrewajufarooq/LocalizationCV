from src_file import parse_args, ConfigParser, FeatureMatching, Homography

# verbose = True
# visualize = False
# config_path = "myconfig.cfg"

# parser = ConfigParser(config_path = config_path, verbose=verbose)

cmd_args = parse_args() #Read arguments passed on the command line
parser = ConfigParser(config_path = cmd_args.config_file_path, verbose=cmd_args.verbose)

# Loading Extracted Features
features, frame_ids = parser.load_features()

# Matching Features
feat_match = FeatureMatching(lib="sklearn", verbose=cmd_args.verbose, visualize=cmd_args.visualize)
corr_ref_ids, pts_in_ref, pts_in_frame = feat_match.match_features(features,
                                                                    frame_ids=frame_ids,
                                                                    ref_frame_ids=parser.config_dict["frame_ids"], # Frame IDs use 1-indexing
                                                                    video_path=parser.config_dict["videos"],
                                                                    # map_image_path=parser.config_dict["image_map"],
                                                                    visualization_delay=1)

# Computing Homography Matrices
homography = Homography(transforms=parser.config_dict["transforms"], use_ransac=True, use_opencv=False, verbose=cmd_args.verbose, visualize=cmd_args.visualize)
homo_output_matrix = homography.compute_homography( pts_in_ref, pts_in_frame, frame_ids, corr_ref_ids, 
                                                   parserObject=parser, 
                                                   # map_image_path=parser.config_dict['image_map'], 
                                                   video_path=parser.config_dict['videos'],
                                                   visualization_delay=0.1)

parser.save_homography_output(homo_output_matrix)