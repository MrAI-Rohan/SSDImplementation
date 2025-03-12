import torch
import torch.nn as nn


class SSDDefaultBoxes(nn.Module):
    """
    Stores default boxes and metadata related to them.
    This class needs to be intialized just once.
    """
    def __init__(self):
        super().__init__()

        self.n_maps = 6
        self.register_buffer("n_dbox_per_pixel", torch.tensor([4, 6, 6, 6, 4, 4]))
        self.register_buffer("aspect_ratios", torch.tensor([1, 2, 1/2, 1/3, 3]))
        self.register_buffer("feature_map_sizes", torch.tensor([38, 19, 10, 5, 3, 1]))

        scales = [self._calculate_scale(k, self.n_maps,) for k in range(1, self.n_maps+1)]
        self.register_buffer("scales", torch.tensor(scales))
        dboxes_per_map = self._create_default_boxes(self.feature_map_sizes,
                                                 self.n_dbox_per_pixel,
                                                 self.scales, self.aspect_ratios)
        self.register_buffer("dboxes", torch.cat(dboxes_per_map))


    def _create_default_boxes(self, feature_map_sizes, n_dbox_per_pixel, scales, aspect_ratios):
        dboxes_per_map = []
        for size, n_dbox, scale in zip(feature_map_sizes, n_dbox_per_pixel, scales):
            dboxes = []
            for i in range(size):
                for j in range(size):
                    center_x = (i+0.5)/size
                    center_y = (j+0.5)/size

                    for ratio in aspect_ratios[:n_dbox-1]:
                        height = scale[0]/torch.sqrt(ratio)
                        width = scale[0]*torch.sqrt(ratio)
                        dboxes.append([center_x, center_y, width, height])

                    # Extra box for ascpect ratio 1
                    height = scale[1]
                    width = scale[1]
                    dboxes.append([center_x, center_y, width, height])
            dboxes_per_map.append(torch.tensor(dboxes))

        return dboxes_per_map


    def _calculate_scale(self, k, n_maps, s_min=0.2, s_max=0.9):
        # k is the nth feature map
        f = lambda k: (s_min + ((s_max-s_min)/(n_maps-1))*(k-1))

        s1 = torch.tensor(0.1 if k==1 else f(k))
        s2 = torch.sqrt(s1*f(k+1))

        return s1, torch.min(s2, torch.tensor(s_max))