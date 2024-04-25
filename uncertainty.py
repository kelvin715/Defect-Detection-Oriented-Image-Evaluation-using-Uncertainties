from typing import List, Optional, Tuple
from matplotlib import pyplot
import numpy as np
from PIL import Image
import os
from sklearn.cluster import KMeans
from copy import deepcopy
import sys

sys.path.append('/Data4/student_zhihan_data/source_code/yolo/ultralytics')
from ultralytics.utils.plotting import Annotator, Colors
from ultralytics import YOLO

'''this file is used to calculate the entropy of predicted bounding boxes to quantify uncertainty'''

def cal_entropy(prob):
    """
    Calculate the Shannon entropy of a probability distribution.

    Args:
        prob (array-like): Array representing a probability distribution.

    Returns:
        float: The entropy of the probability distribution.
    """
    # Calculate the entropy using the formula: -∑(p * log2(p))
    entropy = -1 * np.sum(prob * np.log2(prob))
    return entropy


def cal_entropy_one_image(img_path, model, times):
    """
    Calculate the mean Shannon entropy of a probability distribution
    for a given image.

    Args:
        img_path (str): Path to the image.
        model (YOLO): YOLO model.
        times (int): Number of times to infer the image.

    Returns:
        tuple: A tuple containing the mean entropy and the model's results.
    """
    # Create a list of img_path repeated times
    img_path_list = [img_path] * times

    # Infer the image multiple times
    results = model(img_path_list)

    # Calculate the entropy for each inference result
    entropy = []
    for re in results:
        cls_all = re.cls_all 
        if len(cls_all) != 0:
            cls_all = np.array(cls_all.cpu())
            entropy_sum = 0
            for i in range(len(cls_all)):
                entropy_sum += cal_entropy(cls_all[i])
            entropy.append(entropy_sum / len(cls_all)) 
        
    # Calculate the mean entropy
    entropy_mean = np.mean(np.array(entropy))
    
    # Return the mean entropy and the model's results
    return entropy_mean, results


def cluster_bounding_boxes(bounding_boxes: np.ndarray,
                           n_clusters: int = 3,
                           confs: Optional[np.ndarray] = None,
                           threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, List[float], float]:
    """Cluster the bounding boxes based on their coordinates.

    Args:
        bounding_boxes (np.ndarray): Array of shape (n_boxes, 4) representing the bounding boxes.
        n_clusters (int, optional): Number of clusters to form. Defaults to 3.
        confs (Optional[np.ndarray], optional): Array of confidence scores for each bounding box. Defaults to None.
        threshold (float, optional): Confidence threshold for selecting bounding boxes. Defaults to 0.5.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[float], float]: 
            selected_data (np.ndarray): Array of selected bounding boxes.
            labels (np.ndarray): Array of labels for each bounding box.
            variances (List[float]): List of variances for each cluster.
            weighted_variance_sum (float): Weighted sum of variances.
    """

    # Select bounding boxes above the threshold
    if confs is None:
        selected_data = bounding_boxes[...]
    else:
        selected_data = bounding_boxes[confs > threshold]

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=1000).fit(selected_data)
    labels = kmeans.labels_

    # Calculate variance for each cluster
    variances = []
    for i in range(n_clusters):
        cluster_data = selected_data[labels == i]
        variances.append(np.var(cluster_data, axis=0))

    # Calculate weighted variance
    weighted_variance = np.array(variances) * np.bincount(labels,
                                                          minlength=n_clusters).reshape(-1, 1) / np.sum(
        np.bincount(labels))

    # Calculate weighted variance sum
    weighted_variance_sum = np.sum(np.mean(weighted_variance, axis=1))

    return selected_data, labels, variances, weighted_variance_sum

def visualize_cluster_results(results, selected_boxes, labels, specify):
    """
    Visualizes the results of clustering by displaying an annotated image of 
    selected bounding boxes.

    Args:
        results (List[ObjectDetectionResult]): List of object detection results.
        selected_boxes (np.ndarray): Array of selected bounding boxes.
        labels (np.ndarray): Array of labels for each bounding box.
        specify (int): Index of the cluster to visualize.

    Returns:
        None
    """
    
    # Get a deep copy of the original image
    background = results[0].orig_img
    # Create an annotator object to draw on the image
    annotator = Annotator(
        deepcopy(background),
    )
    
    # Get the colors object to get colors for labels
    colors = Colors()
    # Get the names of classes from the first result in the list
    names = results[0].names
    # Loop through each selected box
    for id, re in enumerate(selected_boxes):
        # If the label of the box is not the specified label, skip to the next box
        if labels[id] != specify:
            continue
        
        # Loop through each detection in the box in reverse order
        for d in reversed(re):
            # Get the class index, confidence, and id of the detection
            c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
            # Create the label string
            name = ("" if id is None else f"id:{id} ") + names[c]
            label = (f"{name} {conf:.2f}" if conf else name)
            # Get the coordinates of the box
            box = d.xyxy.squeeze()
            # Draw the box and label on the image
            annotator.box_label(box, label, color=colors(c, True), rotated=False)    
    
    # Convert the annotated image to a PIL image and show it
    im = Image.fromarray(annotator.result()[..., ::-1])  # RGB PIL image
    pyplot.imshow(im)
    pyplot.show()

def cal_uncertainty(dirs, times, model):
    """
    Calculate uncertainty metrics for a given set of directories and a model.

    Args:
        dirs (list): List of directories containing test images and labels.
        times (int): Number of times to run inference for each image.
        model (YOLO): YOLO model instance.

    Returns:
        tuple: A tuple containing objectness uncertainty, weighted variance sum, and weighted entropy.
    """
    # Loop through each directory
    for j in dirs:
        # Loop through each image in the directory
        for i in range(len(os.listdir(os.path.join(j, 'test', 'images')))):
            # Get image and label paths
            img = os.listdir(os.path.join(j, "test", "images"))[i]
            label = os.listdir(os.path.join(j, "test", "labels"))[i]
            img_path = os.path.join(j, "test", "images", img)
            label_path = os.path.join(j, "test", "labels", label)

            # Run inference
            results = model([img_path]*times, verbose=False)

            # Initialize lists to store values
            boundingboxes = []
            boxes = []
            conf = []
            cls_conf = []

            # Loop through each result and extract values
            for re in results:
                conf.extend(re.boxes.conf.cpu())
                tmp = re.boxes.xywhn
                boxes.extend(re.boxes.cpu())
                boundingboxes.extend(tmp.cpu())
                cls_conf.extend(re.cls_all.cpu())

            # Calculate cluster variables
            cluster_num = len(np.loadtxt(label_path))
            boundingboxes = np.array(boundingboxes)
            conf = np.array(conf)
            cls_conf = np.array(cls_conf)
            selected_data, labels, variances, weighted_variance_sum = cluster_bounding_boxes(boundingboxes, n_clusters=cluster_num, confs=np.array(conf), threshold=0.5)

            # Calculate uncertainty metrics
            objectness_uncertainty = np.var(conf[conf > 0.5])
            entropy_cluster = []
            for n in range(cluster_num):
                cluster = cls_conf[conf > 0.5][labels == n]
                entropy = np.apply_along_axis(lambda x: -1 * np.sum(x * np.log2(x)), 1, cluster)
                entropy_cluster.append(np.mean(entropy))
            weighted_entropy = np.mean(np.array(entropy_cluster) * np.bincount(labels, minlength=cluster_num).reshape(-1, 1) / np.sum(np.bincount(labels)))

            # Print the metrics
            print(f'{j}:objectness_uncertainty: {objectness_uncertainty}, weighted_variance_sum: {weighted_variance_sum}, weighted_entropy: {weighted_entropy}')

            # Return the metrics
            return objectness_uncertainty, weighted_variance_sum, weighted_entropy


if __name__ == '__main__':
    # Load the model
    # Please note that the model imported here is not the same as the one used in the original repository because I modified the *****non-maxima suppression function******* in the /Data4/student_zhihan_data/source_code/yolo/ultralytics/ultralytics/utils/ops.py file  
    model = YOLO('/Data4/student_zhihan_data/source_code/yolo/ultralytics/runs/detect/GC10-DET_brightness_0 detect by yolov8n with dropout(p=0.1)/weights/best.pt')    # Get the directories to test
    img_path = '/Data4/student_zhihan_data/data/GC10-DET_brightness_10/test/images/img_08_425391700_00198_jpg.rf.79baed4ea8e426615cf676e94acf6292.jpg'
    results = model([img_path]*400)
