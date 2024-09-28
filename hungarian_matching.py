import os
import numpy as np
import cv2
import json
from scipy.optimize import linear_sum_assignment

def load_proposals(file_path):
    """
    Load proposals from a file, along with their labels.
    Extract elements from positions 1:5 as proposals and labels from position 0.
    If element 5 exists, extract elements from 5:12 as rpn_3d.
    """
    proposals = []
    labels = []
    rpn_3d_boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            
            # Extract label from the last part
            label = parts[-1]
            labels.append(label)
            
            # Extract the elements from positions 1:5 as proposals
            proposal_box = list(map(float, parts[1:5]))  # Ensure the elements are floats
            proposals.append(proposal_box)
            
            # If element 5 exists, extract elements from 5:12 as rpn_3d
            if len(parts) > 5:
                rpn_3d = list(map(float, parts[5:12]))  # Ensure the elements are floats
                rpn_3d_boxes.append(rpn_3d)
                
    return proposals, labels, rpn_3d_boxes


def draw_boxes_on_image(image, boxes, labels=None, color=(0, 255, 0)):
    """
    Draw bounding boxes on an image with optional labels.
    """
    if not boxes:
        print("No boxes to draw.")
        return image
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw the label if provided
        if labels:
            label = labels[i]
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image


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

def compute_center_distance(box1, box2):
    """
    Compute Euclidean distance between the centers of two bounding boxes.
    Each box is represented as [x1, y1, x2, y2].
    """
    center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance

def compute_cost_matrix(rpn_boxes, projected_boxes, iou_weight=0.5, distance_weight=0.5):
    """
    Compute the cost matrix based on both IoU and center distance.
    """
    cost_matrix = np.zeros((len(rpn_boxes), len(projected_boxes)))
    for i, rpn_box in enumerate(rpn_boxes):
        for j, projected_box in enumerate(projected_boxes):
            iou_cost = -compute_iou(rpn_box, projected_box)  # We use negative IoU as cost
            distance_cost = compute_center_distance(rpn_box, projected_box)
            combined_cost = iou_weight * iou_cost + distance_weight * distance_cost
            cost_matrix[i, j] = combined_cost
    return cost_matrix

def match_rpns_and_projected(rpn_boxes, projected_boxes, iou_weight=0.5, distance_weight=0.5):
    """
    Match RPNs to projected RPNs using the Hungarian algorithm.
    """
    cost_matrix = compute_cost_matrix(rpn_boxes, projected_boxes, iou_weight, distance_weight)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = list(zip(row_ind, col_ind))
    return matches

base_dir = "/home/saeed/3D_Obj_Det/OpenPCDet/data/kitti/training/"
proposals_dir = "/home/saeed/3D_Obj_Det/Hungarian_Matching/2d_proposals/"
projected_dir = "/home/saeed/3D_Obj_Det/Hungarian_Matching/2d_projected/"
output_dir = "/home/saeed/3D_Obj_Det/Hungarian_Matching/Hungarian_Matching_Visualized/"
results_dir = "/home/saeed/3D_Obj_Det/Hungarian_Matching/Hungarian_Results/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

proposal_files = [os.path.join(proposals_dir, f) for f in os.listdir(proposals_dir) if f.endswith('.txt')]

for proposal_file in proposal_files:
    frame_id = os.path.basename(proposal_file).replace("2D_", "").replace(".txt", "")
    frame_id = os.path.basename(frame_id).replace("_pred", "").replace(".txt", "")
    
    img_file = os.path.join(base_dir, 'image_2', f"{frame_id}.png")
    projected_file = os.path.join(projected_dir, f"2D_Proj_{frame_id}.txt")

    print(f"Processing frame: {frame_id}")
    
    # Load proposals and projected boxes along with YOLO labels
    proposals, yolo_labels, _ = load_proposals(proposal_file)
    projected_boxes, _, rpn_3d_boxes = load_proposals(projected_file)

    # Load image
    image = cv2.imread(img_file)
    
    # Draw 2D proposals on the image
    image_with_proposals = draw_boxes_on_image(image.copy(), proposals, color=(0, 255, 0))
    
    # Draw 2D projected boxes on the same image
    image_with_proposals_and_projected = draw_boxes_on_image(image_with_proposals, projected_boxes, color=(255, 0, 0))

    # Match proposals and projected boxes
    matches = match_rpns_and_projected(proposals, projected_boxes, iou_weight=0.5, distance_weight=0.5)
    print("Matches:", matches)

    # Draw matched boxes with different colors and save YOLO labels
    for idx, (rpn_index, projected_index) in enumerate(matches):
        rpn_box = proposals[rpn_index]
        projected_box = projected_boxes[projected_index]
        
        # Draw RPN box in green
        cv2.rectangle(image, (int(rpn_box[0]), int(rpn_box[1])), (int(rpn_box[2]), int(rpn_box[3])), (0, 255, 0), 2)
        
        # Draw projected box in blue
        cv2.rectangle(image, (int(projected_box[0]), int(projected_box[1])), (int(projected_box[2]), int(projected_box[3])), (255, 0, 0), 2)
        
        # Annotate the index (as the ID) of the match
        cv2.putText(image, f"ID {idx}", (int(rpn_box[0]), int(rpn_box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, f"ID {idx}", (int(projected_box[0]), int(projected_box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Save the output image with matched boxes
    output_file = os.path.join(output_dir, f"{frame_id}_matched.png")
    cv2.imwrite(output_file, image)
    print(f"Saved matched visualization to {output_file}")

    # Save the Hungarian results, including 3D RPN and YOLO labels in JSON format with IDs
    results = {
        "frame_id": frame_id,
        "matches": [
            {
                "id": idx,  # Unique ID for the match
                "rpn_2d": proposals[rpn_index],
                "rpn_3d": rpn_3d_boxes[projected_index] if rpn_3d_boxes else None,
                "projected_2d": projected_boxes[projected_index],
                "yolo_label": yolo_labels[rpn_index]  # Assume YOLO label corresponds to RPN index
            }
            for idx, (rpn_index, projected_index) in enumerate(matches)
        ]
    }
    
    results_file = os.path.join(results_dir, f"{frame_id}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved matching results to {results_file}")

