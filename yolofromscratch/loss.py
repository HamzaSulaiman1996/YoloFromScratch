import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=1):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        iou_b1 = intersection_over_union(predictions[..., 2:6], target[..., 2:6])
        iou_b2 = intersection_over_union(predictions[..., 7:11], target[..., 2:6])

        #         print(iou_b1,iou_b2)

        #         print(iou_b1.shape,iou_b2.shape)
        # #         print(iou_b1)

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        ioumax, bestbox_ind = torch.max(ious,
                                        dim=0)  ##bestbox_ind tells which box produced highest iou, ioumax tells the value of that iou
        existobj = target[..., 1:2]
        #         print(existobj,existobj.shape) #if object exisits

        box_predictions = existobj * (bestbox_ind * predictions[..., 7:11] + (1 - bestbox_ind) * predictions[..., 2:6])

        target_bbox = existobj * target[..., 2:6]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )

        target_bbox[..., 2:4] = torch.sqrt(target_bbox[..., 2:4])

        #         print(target_bbox.shape,target_bbox)
        #         print(box_predictions,box_predictions.shape)

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(target_bbox, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
                bestbox_ind * predictions[..., 6:7] + (1 - bestbox_ind) * predictions[..., 1:2]
        )

        object_loss = self.mse(
            torch.flatten(existobj * pred_box),
            torch.flatten(existobj * target[..., 1:2]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.mse(
            torch.flatten((1 - existobj) * predictions[..., 1:2], start_dim=1),
            torch.flatten((1 - existobj) * target[..., 1:2], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - existobj) * predictions[..., 6:7], start_dim=1),
            torch.flatten((1 - existobj) * target[..., 6:7], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(existobj * predictions[..., :1], end_dim=-2, ),
            torch.flatten(existobj * target[..., :1], end_dim=-2, ),
        )

        loss = (
                5 * box_loss  # first two rows in paper
                + object_loss  # third row in paper
                + class_loss  # fifth row
                + 0.5 * no_object_loss
        )

        return class_loss, object_loss, box_loss, no_object_loss, loss
