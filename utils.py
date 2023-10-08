import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates the Intersection over Union (IoU) between predicted bounding boxes and ground truth bounding boxes.
    
    Args:
        boxes_preds (tensor): Predicted bounding boxes, shape (N, 4).
        boxes_labels (tensor): Ground truth bounding boxes, shape (N, 4).
        box_format (str): Format of the bounding boxes. Can be either "midpoint" (default) or "corners".
        
    Returns:
        tensor: Intersection over Union (IoU) for each pair of boxes, shape (N,).
    """
    
    # Convert box format if necessary
    if box_format == "midpoint":
        # [midpoint_x, midpoint_y, width, heigth]
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    elif box_format == "corners":
        # [top_x, top_y, bottom_x, bottom_y]
        box1_x1, box1_y1, box1_x2, box1_y2 = boxes_preds[..., 0:1], boxes_preds[..., 1:2], boxes_preds[..., 2:3], boxes_preds[..., 3:4]
        box2_x1, box2_y1, box2_x2, box2_y2 = boxes_labels[..., 0:1], boxes_labels[..., 1:2], boxes_labels[..., 2:3], boxes_labels[..., 3:4]
    else:
        raise ValueError("Invalid box format. Supported formats are 'midpoint' and 'corners'.")
    
    # Calculate the coordinates of the intersection rectangle
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    # Intersection area
    # torch.clamp is used to limit or "clamp" the values of a tensor within a specified range.
    # e.g. torch.clamp(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32), min=2, max=4)
    # O/P  tensor([2., 2., 3., 4., 4.])
    intersection_area = torch.clamp(x2 - x1, min=1e-6) * torch.clamp(y2 - y1, min=1e-6) 
    
    
    # Calculate the area of each bounding box
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
    # Calculate the Union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / (union_area + 1e-6)  # Adding epsilon to avoid division by zero
    print(f"Intersection Area: {intersection_area} \n Box1 Area: {box1_area} \n Box2 Area: {box2_area} \n Union Area: {union_area}")
    return iou

def non_max_suppression(boxes, iou_threshold, threshold, box_format='corners'):
    """
    Calculates the non-max suppresion for predicted bounding boxes

    Parameters
    ----------
    boxes : List
        List of predicted bounding boxes [ [class, probability, x1, y1, x2, y2], ... ].
    iou_threshold : float
        intersection of union threshold
    threshold : float
        threshold for predicted probability
    box_format : str, optional
        Format of the bounding boxes. This can be either 'midpoint' or 'corners'. The default is "corners".

    Returns
    -------
    boxes_after_nms : list
        List of boxes choosen after non-max suppression

    """
    assert type(boxes) == list

    boxes = [box for box in boxes if box[1]>threshold]
    # Sort the bounding boxes by their prob scores in descending order
    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
    
    # Initialize a list to store the indices of the selected boxes
    boxes_after_nms = []
    
    while boxes:
        # Select the box with the highest score
        choosen_box = boxes.pop(0)
        
        boxes = [
           box
           for box in boxes
           # if it isn't of the same class we don't want to compare them
           if box[0] != choosen_box[0]
           or
           # if iou is less that threshhold then we want to keep it
           # i.e. only when they are not overlapping
           intersection_over_union(
                torch.tensor(choosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format
           ) < iou_threshold
       ]
        boxes_after_nms.append(choosen_box)
    return boxes_after_nms

def mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=0.5,
        box_format="corners",
        num_classes=20
    ):
    """
    Calculates the mean average precision

    Parameters
    ----------
    pred_boxes : list
        List of predicted bounding boxes [ [train_idx, class_pred, probability, x1, y1, x2, y2], ... ].
    true_boxes : list
        List of true bounding boxes [ [train_idx, class_pred, probability, x1, y1, x2, y2], ... ].
    iou_threshold : TYPE, optional
        intersection of union threshold. The default is 0.5.
    box_format : TYPE, optional
        Format of the bounding boxes. This can be either 'midpoint' or 'corners'. The default is "corners".
    num_classes : TYPE, optional
        number of classes. The default is 20.

    Returns
    -------
    Mean average precision value.

    """
    average_precisions = []
    epsilon = 1e-6
    
    # Iterating over each class
    for c in range(num_classes):
        detections = []
        ground_truth = []
        
        # filtering out the detected bbox of current class
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        # filtering out the true bbox of current class
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truth.append(true_box)
                
        # img 0 has 3 bboxes
        # img 1 has 5 bboxes
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truth])
        
        # amount_bboxes = {0:torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}
        for key, value in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(value)
        # sorting the bounding boxes by their prob scores in descending order
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_boxes = len(ground_truth)
        
        for detection_idx, detection in enumerate(detections):
            # filtering out only those bbox which has same train_idx as detected bbox train_idx 
            # or of that particular image
            ground_truth_img = [ 
                bbox
                for bbox in ground_truth 
                if bbox[0] == detection[0]
            ]
            num_gts = len(ground_truth_img)
            best_iou = 0
            # getiing the 1 best detected bbox of an image for a class
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(detection[3:], gt[3:], box_format=box_format)
                if iou > best_iou:
                    best_iou = iou
                    best_iou_idx = idx
            
            # if best bbox iou is greater than threshold and it is not already covered then it is TP else FP
            if best_iou > iou_threshold:
                # checking if this ground truth image is already covered by checking from counter
                if amount_bboxes[detection[0]][best_iou_idx] == 0:
                    TP[detection_idx] = 1
                    # updating that it is covered
                    amount_bboxes[detection[0]][best_iou_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        # torch.sum returns the cumulative sum of elements
        # e.g. [1,1,0,1,0] -> [1,2,2,3,3]
        TP_cummulative_sum = torch.cumsum(TP, dim=1)
        FP_cummulative_sum = torch.cumsum(FP, dim=0)
        recalls = TP_cummulative_sum / (total_true_boxes + epsilon)
        precisions = torch.divide(TP_cummulative_sum, (TP_cummulative_sum + FP_cummulative_sum + epsilon))
        # Adding this for numerical integration
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # Numerical integration to get the area under the curve
        average_precisions.append(torch.trapz(precisions, recalls))
        
    return sum(average_precisions)/len(average_precisions)

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )


            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes



def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
        
       
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    