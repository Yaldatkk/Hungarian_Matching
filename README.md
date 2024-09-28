For each KITTI scene, we have 3D points and their corresponding 2D images. The objective is to project the 3D bounding boxes onto the relevant 2D image. This projection method adheres to the original OpenPCDet format, with the implementation found in 2d_projection.py. Once the 2D projected bounding boxes are generated, we utilize Hungarian Matching to identify the optimal matches between the 2D projected boxes and the 2D RPNs obtained from any 2D detection method. The results of the best matching bounding boxes between the 2D proposals and the 2D projected bounding boxes can be found in the Hungarian_Results folder.
