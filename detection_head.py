import torch
import torch.nn as nn

class SSDDetectionHead(nn.Module):
    """Defines the conv layers responsible for predicting box offsets and class scores."""

    def __init__(self, in_channels_per_map, n_dbox_per_map, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.reg_heads = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels_per_map[i], out_channels=4*n_dbox_per_map[i], kernel_size=1)
            for i in range(len(in_channels_per_map))])

        self.cls_heads = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels_per_map[i], out_channels=n_classes*n_dbox_per_map[i], kernel_size=1)
            for i in range(len(in_channels_per_map))])

        for reg_head, cls_head in zip(self.reg_heads, self.cls_heads):
            nn.init.xavier_uniform_(reg_head.weight)
            nn.init.xavier_uniform_(cls_head.weight)

            nn.init.zeros_(reg_head.bias)
            nn.init.zeros_(cls_head.bias)

    def forward(self, maps):
        """
        Takes in feature maps extracted from VGG Feature Extractor and predicts offsets
        and class scores of the default boxes.

        Output dim: ((B, 8732, 4), (B, 8732, n_classes))
        """

        batch_size = next(iter(maps)).shape[0]

        det_outputs = torch.cat([reg_head(map).view(batch_size, -1, 4)
                        for map, reg_head in zip(maps, self.reg_heads)], dim=1)

        cls_outputs = torch.cat([cls_head(map).view(batch_size, -1, self.n_classes)
                        for map, cls_head in zip(maps, self.cls_heads)], dim=1)

        return det_outputs, cls_outputs