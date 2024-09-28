import os
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import clip
from PIL import Image

base_dir = "/home/saeed/3D_Obj_Det/OpenPCDet/data/kitti/KITTI/object/testing/"
proposals_dir = "/home/saeed/3D_Obj_Det/Hungarian_Matching/2d_proposals/"
projected_dir = "/home/saeed/3D_Obj_Det/Hungarian_Matching/2d_projected/"
output_dir = "/home/saeed/3D_Obj_Det/Hungarian_Matching/visualized_images/"
calib = 'calib'
image_folder = 'image_2'
hungarian_results_dir = "/home/saeed/3D_Obj_Det/Hungarian_Matching/Hungarian_Results"
hungarian_visualized_dir = "/home/saeed/3D_Obj_Det/Hungarian_Matching/Hungarian_Matching_Visualized"
hungarian_consistency_visualized_dir = "/home/saeed/3D_Obj_Det/Hungarian_Matching/Hungarian_Matching_Visualized_With_Consistency"

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def load_proposals(file_path, has_obj_name=True):
    proposals = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if has_obj_name:
                parts = parts[1:]
            proposals.append([float(x) for x in parts])
    return proposals

def draw_boxes_on_image(image, boxes, color, thickness=2):
    for box in boxes:
        top_left = (int(box[0]), int(box[1]))
        bottom_right = (int(box[2]), int(box[3]))
        cv2.rectangle(image, top_left, bottom_right, color, thickness)
    return image

def extract_patch(image, box):
    h, w, _ = image.shape
    top_left = (max(0, int(box[0])), max(0, int(box[1])))
    bottom_right = (min(w, int(box[2])), min(h, int(box[3])))
    patch = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    return patch

def compute_clip_similarity(patch1, patch2):
    if patch1.size == 0 or patch2.size == 0:
        return 0  # If the patch is empty, return 0 similarity
    image1 = preprocess(Image.fromarray(patch1)).unsqueeze(0).to(device)
    image2 = preprocess(Image.fromarray(patch2)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features1 = clip_model.encode_image(image1)
        image_features2 = clip_model.encode_image(image2)
    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    image_features2 /= image_features2.norm(dim=-1, keepdim=True)
    similarity = (image_features1 @ image_features2.T).item()
    return similarity

def hungarian_matching_with_consistency(proposals, projected_boxes, image):
    num_proposals = len(proposals)
    num_projected_boxes = len(projected_boxes)
    cost_matrix = np.zeros((num_proposals, num_projected_boxes))

    for i, prop in enumerate(proposals):
        for j, proj in enumerate(projected_boxes):
            iou_cost = 1 - compute_iou(prop, proj)
            prop_center = [(prop[0] + prop[2]) / 2, (prop[1] + prop[3]) / 2]
            proj_center = [(proj[0] + proj[2]) / 2, (proj[1] + proj[3]) / 2]
            center_distance_cost = np.linalg.norm(np.array(prop_center) - np.array(proj_center))
            prop_patch = extract_patch(image, prop)
            proj_patch = extract_patch(image, proj)
            consistency_cost = 1 - compute_clip_similarity(prop_patch, proj_patch)
            total_cost = iou_cost + center_distance_cost + consistency_cost
            cost_matrix[i, j] = total_cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind, cost_matrix

def compute_iou(box1, box2):
    """
    Compute IoU between two bounding boxes.
    Each box is represented as [x1, y1, x2, y2].
    """
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    inter_area = max(0, x2_min - x1_max + 1) * max(0, y2_min - y1_max + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def save_matching_results(proposals, projected_boxes, row_ind, col_ind, output_path):
    with open(output_path, 'w') as f:
        for i, j in zip(row_ind, col_ind):
            f.write(f"{proposals[i]} matched with {projected_boxes[j]}\n")

def main():
    if not os.path.exists(hungarian_results_dir):
        os.makedirs(hungarian_results_dir)
    if not os.path.exists(hungarian_consistency_visualized_dir):
        os.makedirs(hungarian_consistency_visualized_dir)

    proposal_files = sorted([os.path.join(proposals_dir, f) for f in os.listdir(proposals_dir) if f.endswith('.txt')])

    for proposal_file in proposal_files:
        frame_id = os.path.basename(proposal_file).replace("2D_", "").replace(".txt", "")
        img_file = os.path.join(base_dir, image_folder, f"{frame_id}.png")
        projected_file = os.path.join(projected_dir, f"2D_Proj_{frame_id}.txt")
        output_file = os.path.join(hungarian_results_dir, f"Hungarian_Results_{frame_id}.txt")
        visualized_output_file = os.path.join(hungarian_consistency_visualized_dir, f"Hungarian_Matching_Visualized_{frame_id}.png")

        print(f"Processing frame: {frame_id}")

        # Load proposals and projected boxes
        proposals = load_proposals(proposal_file, has_obj_name=True)
        projected_boxes = load_proposals(projected_file, has_obj_name=False)

        print("Loaded 2D proposals:", proposals)
        print("Loaded 2D projected boxes:", projected_boxes)

        # Load image
        image = cv2.imread(img_file)

        # Perform Hungarian matching with consistency loss
        row_ind, col_ind, cost_matrix = hungarian_matching_with_consistency(proposals, projected_boxes, image)

        # Save matching results
        save_matching_results(proposals, projected_boxes, row_ind, col_ind, output_file)

        # Visualize and save the matched proposals
        matched_image = image.copy()
        for i, j in zip(row_ind, col_ind):
            matched_image = draw_boxes_on_image(matched_image, [proposals[i]], color=(0, 255, 0))  # Green for proposals
            matched_image = draw_boxes_on_image(matched_image, [projected_boxes[j]], color=(255, 0, 0))  # Red for projected

        cv2.imwrite(visualized_output_file, matched_image)
        print(f"Saved visualized matching results to: {visualized_output_file}")

if __name__ == "__main__":
    main()

