class SSDLoss:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction="none")
        self.cross_entropy = nn.CrossEntropyLoss()

    def get_hard_negatives(self, pred_scores, positive_box_mask, n_positives):
        """
        Returns indices of hard negatives.
        Input:
            pred_scores: (8732, n_classes)
            positive_box_mask: (8732)
        Output:
            hard_negative_idxs: (3*n_positives)
        """
        with torch.no_grad():
            negative_mask = ~positive_box_mask
            negative_indices = torch.nonzero(negative_mask).flatten()

            negative_box_scores = pred_scores[negative_mask]

            class_probs = F.softmax(negative_box_scores, dim=-1)

            misclassified_negative_mask = torch.max(class_probs, dim=1)[1] != 0
            misclassified_probs = class_probs[misclassified_negative_mask]
            misclassified_indices = torch.nonzero(misclassified_negative_mask).flatten()

            hard_negative_idxs = misclassified_probs[:, 0].sort()[1][:3*n_positives]

            # After finding hard negatives in misclassfied probabilites, the indices need to be mapped to the original data
            final_idxs = negative_indices[misclassified_indices[hard_negative_idxs]]

        return final_idxs

    def __call__(self, preds, gt):
        gt_bboxes, gt_labels = gt
        pred_bboxes, pred_scores = preds

        batch_size = gt_bboxes.shape[0]

        positive_bbox_mask = gt_labels != 0
        n_positives_per_image = positive_bbox_mask.sum(dim=-1)

        # Smooth L1 Loss
        smooth_l1_per_image = self.smooth_l1_loss(pred_bboxes, gt_bboxes).mean(dim=(1, 2))

        # Cross Entropy Loss
        cross_entropy_per_image = torch.zeros(batch_size, device=gt_bboxes.device)
        for i in range(batch_size):
            if n_positives_per_image[i] == 0:
                continue

            hard_negative_idxs = self.get_hard_negatives(pred_scores[i],
                                                        positive_bbox_mask[i],
                                                        n_positives_per_image[i])

            gt_labels_final = torch.cat((gt_labels[i][positive_bbox_mask[i]],
                                        gt_labels[i][hard_negative_idxs]))

            pred_scores_final = torch.cat((pred_scores[i][positive_bbox_mask[i]],
                                        pred_scores[i][hard_negative_idxs]))

            cross_entropy_per_image[i] = self.cross_entropy(pred_scores_final, gt_labels_final)

        # Final Loss
        loss_per_image = (smooth_l1_per_image + self.alpha * cross_entropy_per_image)
        loss_per_image[n_positives_per_image==0] = 0

        return loss_per_image.mean()