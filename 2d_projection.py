import argparse
import glob
import os
import numpy as np
from pcdet.utils import common_utils, box_utils, calibration_kitti
from skimage import io

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()
    return args

def load_proposals(filepath):
    proposals = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # Ignore the first string and convert the remaining parts to float
            proposals.append([float(x) for x in parts[1:]])
    return np.array(proposals)

def is_box_inside_image(box, image_shape):
    """Check if the bounding box is inside the image boundaries."""
    h, w = image_shape
    x_min, y_min, x_max, y_max = box
    return (x_min > 0 and y_min > 0 and x_max < w-1 and y_max < h-1)

def main():
    base_dir = "/home/saeed/3D_Obj_Det/OpenPCDet/data/kitti/training/"
    proposals_dir = "/home/saeed/3D_Obj_Det/Hungarian_Matching/3d_proposals/"
    output_dir = "/home/saeed/3D_Obj_Det/Hungarian_Matching/2d_projected/"
    calib_dir = os.path.join(base_dir, 'calib')

    args = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    proposal_files = glob.glob(os.path.join(proposals_dir, "3D_*.txt"))
    proposal_files.sort()

    for proposal_file in proposal_files:
        # Extract frame_id from the filename
        frame_id = os.path.basename(proposal_file).replace("3D_", "").replace(".txt", "")
        if not frame_id.isdigit():
            continue  # Skip if frame_id is not a valid number

        frame_id = frame_id.zfill(6)  # Ensure frame_id is zero-padded
        img_file = os.path.join(base_dir, 'image_2', f"{frame_id}.png")
        calib_file = os.path.join(calib_dir, f"{frame_id}.txt")

        # Debug print statements
        print(f"Processing frame: {frame_id}")
        print(f"Image file path: {img_file}")
        print(f"Calibration file path: {calib_file}")

        if not os.path.isfile(img_file):
            print(f"Image file not found: {img_file}")
            continue

        if not os.path.isfile(calib_file):
            print(f"Calibration file not found: {calib_file}")
            continue

        proposals = load_proposals(proposal_file)
        #print("Loaded 3D proposals:", proposals)

        clb = calibration_kitti.Calibration(calib_file)

        output_lines = []

        image_shape = np.array(io.imread(img_file).shape[:2], dtype=np.int32)  # Get image shape
        
        #print('proposals ',proposals[0])
        #import time


        for idx, proposal in enumerate(proposals, start=1):  # Start index at 1
            #print(proposals[0])
            #time.sleep(10)
            proposal = proposal[np.newaxis, :]  # Reshape to (1, -1)
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(proposal, clb)
            #print("3D box in camera coordinates:", pred_boxes_camera)

            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(pred_boxes_camera, clb, image_shape=image_shape)
            #print("2D box in image coordinates:", pred_boxes_img)

            # Save each projected box coordinates along with the original 3D proposal if it is inside the image
            for box in pred_boxes_img:
                if is_box_inside_image(box, image_shape):
                    original_3d_info = " ".join(map(str, proposal[0]))
                    output_lines.append(f"{idx} {box[0]} {box[1]} {box[2]} {box[3]} {original_3d_info}")

        output_file = os.path.join(output_dir, f"2D_Proj_{frame_id}.txt")
        with open(output_file, 'w') as f:
            f.write("\n".join(output_lines))
        print(f"Saved 2D projections to {output_file}")

    logger.info('Demo done.')

if __name__ == '__main__':
    main()

