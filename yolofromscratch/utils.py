import torch
import numpy as np
import cv2


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def gen_dim(pred):
    pred = pred.detach().reshape(-1, 7, 7, 1 + 2 * 5)
    first_bbox = pred[..., 1:2]
    second_bbox = pred[..., 6:7]
    both_box_prob = torch.cat((first_bbox.unsqueeze(0), second_bbox.unsqueeze(0)), 0)
    val, ind = torch.max(both_box_prob, 0)
    box_pred = (ind * pred[..., 6:11] + (1 - ind) * (pred[..., 1:6]))

    return box_pred


def conv_box(box_pred):
    conv_box = []
    red_dim = np.array(box_pred.squeeze().to('cpu'))
    for i, box in enumerate(red_dim):
        for j, grid in enumerate(box):
            if grid[0] < 0.1:
                continue
            conf = grid[0]
            col = (grid[1] + i) / red_dim.shape[0]
            row = (grid[2] + j) / red_dim.shape[0]
            w = grid[3] / red_dim.shape[0]
            h = grid[4] / red_dim.shape[0]

            conv_box.append([conf, col, row, w, h])
    return conv_box


def nms(bboxes, iou_threshold=0.2, nms_threshold=0.3):
    bboxes = [box for box in bboxes if box[0] > nms_threshold]
    # Sort the bounding boxes based on confidence score (in descending order)
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)

    # Initialize an empty list to store the selected bounding boxes
    selected_bboxes = []

    while bboxes:
        # Select the bounding box with the highest confidence score
        best_bbox = bboxes[0]
        selected_bboxes.append(best_bbox)

        # Calculate the IOU between the best_bbox and all other remaining bounding boxes
        remaining_bboxes = []
        for bbox in bboxes[1:]:
            if intersection_over_union(torch.tensor(best_bbox[1:]), torch.tensor(bbox[1:])) < iou_threshold:
                remaining_bboxes.append(bbox)

        # Update the list of bounding boxes to consider for the next iteration
        bboxes = remaining_bboxes

    return selected_bboxes


def plot(boxes, gt):
    coord = [np.array(box[:]) for box in boxes]
    img = (np.array(gt) * 255).astype('uint8')
    for c in coord:
        if c[0] > 0.1:
            start_x = int((c[1] - c[3] / 2) * 448)
            end_x = int((c[1] + c[3] / 2) * 448)
            start_y = int((c[2] - c[4] / 2) * 448)
            end_y = int((c[2] + c[4] / 2) * 448)
            #     break
            #     print(start_x,end_x,start_y,end_y)

            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    #     gt_label = np.array([0.4720, 0.3110, 5.2710, 3.7100])
    #     col = int((gt_label[0]+3)/7 *448)
    #     row = int((gt_label[1]+3)/7 *448)
    #     w = int(gt_label[2]/7 *448)
    #     h = int(gt_label[3]/7 *448)

    #     start_x = int(col-w/2)
    #     end_x = start_x + w

    #     start_y = int(row-h/2)
    #     end_y = start_y + h
    #     cv2.rectangle(img, (start_x,start_y),(end_x,end_y), (255,255,0),2)

    cv2.imshow('s', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()