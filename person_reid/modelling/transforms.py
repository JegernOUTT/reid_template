import torch.nn

__all__ = ['MaskKeypointsConcat']


class MaskKeypointsConcat(torch.nn.Module):
    @torch.no_grad()
    def forward(self, images, masks, keypoints):
        bs, _, h, w = images.shape

        keypoints = keypoints.round().long()
        keypoints[..., [0]] = keypoints[..., [0]].clip(0, w)
        keypoints[..., [1]] = keypoints[..., [1]].clip(0, h)
        keypoints_indices = keypoints[:, :, [1]] * w + keypoints[:, :, [0]]
        keypoints_heatmap = torch.zeros((bs, 17, h * w), dtype=images.dtype, device=images.device)
        keypoints_heatmap = keypoints_heatmap.scatter(2, keypoints_indices, 1.0).view(bs, -1, h, w)

        masks = masks.view(bs, 1, h, w) / 255.

        return torch.cat([images, masks, keypoints_heatmap], dim=1)
