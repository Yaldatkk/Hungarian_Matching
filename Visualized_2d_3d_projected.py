import argparse
import glob
from pathlib import Path
import numpy as np
import os
from pcdet.utils import common_utils
from skimage import io
import cv2


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()
    return args


def load_proposals(filepath, has_obj_name=True):
    proposals = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if has_obj_name:
                parts = parts[1:]  # Ignore the first category name if present
            proposals.append([float(x) for x in parts])
    return np.array(proposals)


def draw_boxes_on_image(image, boxes, color):
    for box in boxes:
        top_left = (int(box[0]), int(box[1]))
        bottom_right = (int(box[2]), int(box[3]))
        cv2.rectangle(image, top_left, bottom_right, color=color, thickness=2)
    return image


def main():
    base_dir = "/home/saeed/3D_Obj_Det/OpenPCDet/data/kitti/testing/"
    proposals_dir = "/home/saeed/3D_Obj_Det/Hungarian_Matching/2d_proposals/"
    projected_dir = "/home/saeed/3D_Obj_Det/Hungarian_Matching/2d_projected/"
    output_dir = "/home/saeed/3D_Obj_Det/Hungarian_Matching/visualized_images/"
    calib = 'calib'

    args = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    proposal_files = glob.glob(os.path.join(proposals_dir, "2D_*.txt"))
    proposal_files.sort()

    for proposal_file in proposal_files:
        frame_id = os.path.basename(proposal_file).replace("2D_", "").replace(".txt", "")
        frame_id = os.path.basename(frame_id).replace("_pred", "").replace(".txt", "")
        print(frame_id)
        img_file = os.path.join(base_dir, 'image_2', f"{frame_id}.png")
        projected_file = os.path.join(projected_dir, f"2D_Proj_{frame_id}.txt")

        print(f"Processing frame: {frame_id}")

        # Load proposals and projected boxes
        proposals = load_proposals(proposal_file, has_obj_name=True)
        projected_boxes = load_proposals(projected_file, has_obj_name=False)

        print("Loaded 2D proposals:", proposals)
        print("Loaded 2D projected boxes:", projected_boxes)

        # Load image
        image = cv2.imread(img_file)
        
        # Draw 2D proposals on the image
        image_with_proposals = draw_boxes_on_image(image.copy(), proposals, color=(0, 255, 0))
        
        # Draw 2D projected boxes on the same image
        image_with_proposals_and_projected = draw_boxes_on_image(image_with_proposals, projected_boxes, color=(255, 0, 0))
        
        # Save the image
        output_image_path = os.path.join(output_dir, f"Visualized_{frame_id}.png")
        cv2.imwrite(output_image_path, image_with_proposals_and_projected)
        print(f"Saved visualized image to {output_image_path}")

    logger.info('Demo done.')


if __name__ == '__main__':
    main()

