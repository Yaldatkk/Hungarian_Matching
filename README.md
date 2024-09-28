For each KITTI scene, we have 3D points and their corresponding 2D images. The objective is to project the 3D bounding boxes onto the relevant 2D image. This projection method adheres to the original OpenPCDet format, with the implementation found in 2d_projection.py. Once the 2D projected bounding boxes are generated, we utilize Hungarian Matching to identify the optimal matches between the 2D projected boxes and the 2D RPNs obtained from any 2D detection method. The results of the best matching bounding boxes between the 2D proposals and the 2D projected bounding boxes can be found in the Hungarian_Results folder.


The matching format is structured as follows:

{
    "frame_id": "000000",
    "matches": [
        {
            "id": 0,
            "rpn_2d": [
                716.948914,
                144.885239,
                803.560608,
                316.844757
            ],
            "rpn_3d": [
                8.673374,
                -1.665905,
                -0.655717,
                1.12611,
                0.740124,
                1.915942,
                -1.677326
            ],
            "projected_2d": [
                700.2764282226562,
                142.16505432128906,
                803.5546264648438,
                313.12353515625
            ],
            "yolo_label": "1"
        }
    ]
}

Each frame_id can have a varying number of matches, with each match assigned a unique id. If you adopt the YOLO format for the 2D RPNs, you can include the yolo_label for additional data validation. The rpn_3d field contains the seven-value RPN format used in OpenPCDet, while projected_2d represents its corresponding projected 2D bounding box.
You can locate the visualized matching results in the Hungarian_Matching_Visualized folder.

![000066_matched](https://github.com/user-attachments/assets/1948e825-3797-4d39-8b97-4bcfb50c1c0e)
