import torch
import torch.nn as nn
from default_boxes import SSDDefaultBoxes
from utils import transform_cwh_to_xyxy, transform_xyxy_to_cwh

class GenerateGroundTruth(nn.Module):
    """
    This class is made to use as a transform to generate ground truth for an image.
    """
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu')

        self.dboxes = SSDDefaultBoxes().to(self.device)
        n_dbox_per_map = self.dboxes.n_dbox_per_pixel * self.dboxes.feature_map_sizes**2
        self.register_buffer("n_dbox_per_map", n_dbox_per_map)

    def _jaccard_dboxes(self, bboxes, dboxes):
        """
        Function is made to find jaccard between all default boxes and bboxes in an image.
        Boxes in xyxy format are expected.
        jaccard_dboxes((N, 4), (8732, 4)) -> (N, 8732)
        """
        xmin1, ymin1, xmax1, ymax1 = bboxes.T.unsqueeze(2)
        xmin2, ymin2, xmax2, ymax2 = dboxes.T

        inter_xmin = torch.max(xmin1, xmin2)
        inter_ymin = torch.max(ymin1, ymin2)
        inter_xmax = torch.min(xmax1, xmax2)
        inter_ymax = torch.min(ymax1, ymax2)

        inter_w = torch.clamp(inter_xmax-inter_xmin, min=0)
        inter_h = torch.clamp(inter_ymax-inter_ymin, min=0)

        intersection = inter_w*inter_h
        union = (xmax1-xmin1)*(ymax1-ymin1) + (xmax2-xmin2)*(ymax2-ymin2) - intersection

        return intersection/union

    def _match_labels(self, overlaps, iou_threshold=0.5):
        """
        Takes in overlaps: (N, 8732), Returns tensor: (8732)
        Returned values range from [-1, N), where -1 represents background, and [0, N) corresponds to the index of the matched ground truth label.
        """
        matched_labels = torch.full((8732,), -1, dtype=torch.int64, device=self.device)

        # Criteria 2
        max_iou_per_dbox = torch.max(overlaps, dim=0)
        mask2 = max_iou_per_dbox[0]>=iou_threshold
        idx2 = max_iou_per_dbox[1][mask2]
        matched_labels[mask2] = idx2

        # Criteria 1
        max_iou_per_label = torch.max(overlaps, dim=1)
        mask1 = max_iou_per_label[0]>0
        idx1 = max_iou_per_label[1][mask1]
        matched_labels[idx1] = torch.arange(overlaps.shape[0], dtype=torch.int64)[mask1]

        return matched_labels

    def _generate_offsets(self, matched_labels, bboxes_cwh, dboxes_cwh):
        """Generate offsets for all the matched default boxes."""

        offsets = torch.zeros((8732, 4), device=self.device)
        not_bg_mask = matched_labels != -1
        gcx, gcy, gw, gh = bboxes_cwh[matched_labels[not_bg_mask]].T
        dcx, dcy, dw, dh = dboxes_cwh[not_bg_mask].T

        # All formulas are from the paper.
        off_cx = (gcx-dcx)/dw
        off_cy = (gcy-dcy)/dh
        off_w = torch.log(gw/dw)
        off_h = torch.log(gh/dh)

        offsets[not_bg_mask] = torch.stack([off_cx,off_cy,off_w,off_h], dim=1)

        return offsets

    def forward(self, img, bboxes, labels):
        """
        Generate ground truth for an imgae.
        Ground Truth: Offsets + Labels
        Returns (8732, 4), (8732) shape tensors.
        """
        if bboxes.shape[0] == 0:
            offsets = torch.zeros((8732, 4), device=self.device)
            new_labels = torch.zeros(8732, dtype=torch.int64, device=self.device)

            split_offsets = torch.split(offsets, self.n_dbox_per_map.tolist())
            split_labels = torch.split(new_labels, self.n_dbox_per_map.tolist())

            return split_offsets, split_labels

        dboxes_xyxy = transform_cwh_to_xyxy(self.dboxes.dboxes, denorm=True)
        bboxes_cwh = transform_xyxy_to_cwh(bboxes, norm=True)

        # Generate offsets
        overlaps = self._jaccard_dboxes(bboxes, dboxes_xyxy)
        matched_labels = self._match_labels(overlaps)
        offsets = self._generate_offsets(matched_labels, bboxes_cwh, self.dboxes.dboxes)

        # Generate new labels
        not_bg_mask = matched_labels != -1
        new_labels = torch.zeros(8732, dtype=torch.int64, device=self.device)
        new_labels[not_bg_mask] = labels[matched_labels[not_bg_mask]]

        # Split offsets and labels feature map wise.
        split_offsets = torch.split(offsets, self.n_dbox_per_map.tolist())
        split_labels = torch.split(new_labels, self.n_dbox_per_map.tolist())

        return split_offsets, split_labels

    def to(self, device):
        """Override to() for proper device handling."""
        self.device = device
        return super().to(device)
