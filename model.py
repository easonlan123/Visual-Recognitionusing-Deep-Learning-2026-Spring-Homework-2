"""
NYCU Computer Vision 2026 HW2 - Model definition and training logic.
Implements a DETR-based digit detector using ResNet-50.
"""

import os

# Fix for Windows OpenMP duplicate runtime warnings.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import batched_nms, generalized_box_iou
from tqdm import tqdm


def box_cxcywh_to_xyxy(x_input):
    """Convert boxes from cx-cy-w-h to xmin-ymin-xmax-ymax."""
    cx, cy, w, h = x_input.unbind(-1)
    return torch.stack(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h],
        dim=-1,
    )


def box_xyxy_to_xywh(x_input):
    """Convert boxes from xmin-ymin-xmax-ymax to xmin-ymin-width-height."""
    xmin, ymin, xmax, ymax = x_input.unbind(-1)
    return torch.stack([xmin, ymin, xmax - xmin, ymax - ymin], dim=-1)


def box_ciou(boxes1, boxes2, eps=1e-7):
    """Calculate pairwise CIoU between two sets of xyxy boxes."""
    b1 = boxes1[:, None, :]
    b2 = boxes2[None, :, :]

    inter_x1 = torch.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = torch.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = torch.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = torch.minimum(b1[..., 3], b2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0.0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0.0)
    )[:, None]
    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0.0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0.0)
    )[None, :]
    union = (area1 + area2 - inter_area).clamp(min=eps)
    iou = inter_area / union

    c1x = (boxes1[:, 0] + boxes1[:, 2]) * 0.5
    c1y = (boxes1[:, 1] + boxes1[:, 3]) * 0.5
    c2x = (boxes2[:, 0] + boxes2[:, 2]) * 0.5
    c2y = (boxes2[:, 1] + boxes2[:, 3]) * 0.5
    rho2 = (c1x[:, None] - c2x[None, :]).pow(2) + (c1y[:, None] - c2y[None, :]).pow(2)

    enclose_x1 = torch.minimum(b1[..., 0], b2[..., 0])
    enclose_y1 = torch.minimum(b1[..., 1], b2[..., 1])
    enclose_x2 = torch.maximum(b1[..., 2], b2[..., 2])
    enclose_y2 = torch.maximum(b1[..., 3], b2[..., 3])
    diag_c2 = (enclose_x2 - enclose_x1).pow(2) + (enclose_y2 - enclose_y1).pow(2) + eps

    w1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=eps)
    h1 = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=eps)
    w2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=eps)
    h2 = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=eps)

    v_val = (4.0 / (torch.pi**2)) * (
        torch.atan(w2[None, :] / h2[None, :]) - torch.atan(w1[:, None] / h1[:, None])
    ).pow(2)
    alpha = v_val / (1.0 - iou + v_val + eps)

    return iou - (rho2 / diag_c2) - alpha * v_val


class TrainTransform:
    """Apply augmentation and normalization for training."""

    def __init__(self, image_size):
        h_val, w_val = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.resize = T.Resize((h_val, w_val))
        self.color_jitter = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __call__(self, image, target):
        boxes = target["boxes"].clone()
        labels = target["labels"]

        if random.random() < 0.5:
            image = TF.hflip(image)
            if boxes.numel() > 0:
                boxes[:, 0] = 1.0 - boxes[:, 0]

        if random.random() < 0.8:
            image = self.color_jitter(image)

        image = self.resize(image)
        image = self.normalize(self.to_tensor(image))
        return image, {"boxes": boxes, "labels": labels}


class EvalTransform:
    """Apply deterministic resize and normalization for evaluation."""

    def __init__(self, image_size):
        h_val, w_val = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.resize = T.Resize((h_val, w_val))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __call__(self, image, target):
        image = self.resize(image)
        image = self.normalize(self.to_tensor(image))
        return image, target


class DigitDataset(CocoDetection):
    """COCO dataset wrapper returning normalized boxes and labels."""

    def __getitem__(self, idx):
        image, annotations = super().__getitem__(idx)
        image_id = self.ids[idx]
        width, height = image.size

        boxes = []
        labels = []
        for obj in annotations:
            x_min, y_min, box_w, box_h = obj["bbox"]
            if box_w <= 0 or box_h <= 0:
                continue
            boxes.append(
                [
                    (x_min + box_w / 2.0) / width,
                    (y_min + box_h / 2.0) / height,
                    box_w / width,
                    box_h / height,
                ]
            )
            labels.append(obj["category_id"] - 1)

        if boxes:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.long),
            }

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target, image_id


def collate_fn(batch):
    """Custom collate for images, detection targets, and image IDs."""
    images, targets, image_ids = zip(*batch)
    return torch.stack(images), list(targets), list(image_ids)


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers))

    def forward(self, x_input):
        for layer in self.layers[:-1]:
            x_input = F.relu(layer(x_input))
        return self.layers[-1](x_input)


class DigitDETR(nn.Module):
    """DETR implementation for digit detection."""

    def __init__(self, num_classes=10, num_queries=20, num_decoder_layers=6):
        super().__init__()
        self.num_classes = num_classes
        self.num_decoder_layers = num_decoder_layers

        backbone_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(backbone_model.children())[:-2])
        self.conv = nn.Conv2d(2048, 256, kernel_size=1)

        self.row_embed = nn.Parameter(torch.rand(64, 128))
        self.col_embed = nn.Parameter(torch.rand(64, 128))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=256,
                    nhead=8,
                    dim_feedforward=1024,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.class_embed = nn.Linear(256, num_classes + 1)
        self.bbox_embed = MLP(256, 256, 4, 3)
        self.query_embed = nn.Embedding(num_queries, 256)
        self.label_query_embed = nn.Embedding(num_classes, 256)
        self.box_query_embed = MLP(4, 256, 256, 3)

    def _build_dn_queries(self, targets, batch_size, device, dn_args):
        """Build denoising queries for DN-DETR style training."""
        dn_num = dn_args.get("dn_number", 5)
        max_gt = max((target["labels"].numel() for target in targets), default=0)
        if max_gt == 0:
            return None, None

        pad_size = max_gt * dn_num
        dn_query = torch.zeros((batch_size, pad_size, 256), device=device)
        known_labels = torch.full((batch_size, pad_size), -1, dtype=torch.long, device=device)
        known_boxes = torch.zeros((batch_size, pad_size, 4), dtype=torch.float32, device=device)
        valid_mask = torch.zeros((batch_size, pad_size), dtype=torch.bool, device=device)

        for b_idx, target in enumerate(targets):
            labels = target["labels"]
            boxes = target["boxes"]
            if labels.numel() == 0:
                continue

            rep_labels = labels.repeat(dn_num)
            rep_boxes = boxes.repeat(dn_num, 1)
            count = rep_labels.shape[0]

            noisy_labels = rep_labels.clone()
            label_noise = torch.rand(count, device=device) < dn_args.get("label_noise_ratio", 0.2)
            noisy_count = int(label_noise.sum().item())
            if noisy_count > 0:
                noisy_labels[label_noise] = torch.randint(
                    0,
                    self.num_classes,
                    (noisy_count,),
                    device=device,
                )

            scale = dn_args.get("box_noise_scale", 0.25)
            noisy_boxes = rep_boxes.clone()
            center_noise = (
                (torch.rand_like(noisy_boxes[:, :2]) * 2 - 1)
                * scale
                * noisy_boxes[:, 2:]
            )
            size_noise = (
                (torch.rand_like(noisy_boxes[:, 2:]) * 2 - 1)
                * scale
                * noisy_boxes[:, 2:]
            )
            noisy_boxes[:, :2] = (noisy_boxes[:, :2] + center_noise).clamp(0.0, 1.0)
            noisy_boxes[:, 2:] = (noisy_boxes[:, 2:] + size_noise).clamp(1e-4, 1.0)

            query_embed = self.label_query_embed(noisy_labels) + self.box_query_embed(noisy_boxes)
            dn_query[b_idx, :count] = query_embed
            known_labels[b_idx, :count] = rep_labels
            known_boxes[b_idx, :count] = rep_boxes
            valid_mask[b_idx, :count] = True

        meta = {
            "pad_size": pad_size,
            "known_labels": known_labels,
            "known_boxes": known_boxes,
            "valid_mask": valid_mask,
        }
        return dn_query, meta

    def forward(self, images, targets=None, dn_args=None):
        """Run forward pass."""
        features = self.backbone(images)
        projected = self.conv(features)

        batch_size, _, feat_h, feat_w = projected.shape
        pos = torch.cat(
            [
                self.col_embed[:feat_w].unsqueeze(0).repeat(feat_h, 1, 1),
                self.row_embed[:feat_h].unsqueeze(1).repeat(1, feat_w, 1),
            ],
            dim=-1,
        ).flatten(0, 1).unsqueeze(0)

        src = projected.flatten(2).permute(0, 2, 1)
        query = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        dn_meta = None
        if self.training and targets is not None:
            cfg = dn_args or {}
            if cfg.get("enabled", False):
                dn_query, dn_meta = self._build_dn_queries(targets, batch_size, images.device, cfg)
                if dn_query is not None:
                    query = torch.cat([dn_query, query], dim=1)

        memory = self.encoder(src + pos)
        tgt = torch.zeros_like(query)

        hs_list = []
        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(tgt + query, memory)
            hs_list.append(tgt)

        hs = torch.stack(hs_list)
        logits = self.class_embed(hs)
        boxes = self.bbox_embed(hs).sigmoid()

        dn_pad = dn_meta["pad_size"] if dn_meta is not None else 0
        output = {
            "pred_logits": logits[-1][:, dn_pad:, :],
            "pred_boxes": boxes[-1][:, dn_pad:, :],
            "aux_outputs": [
                {
                    "pred_logits": logits[i][:, dn_pad:, :],
                    "pred_boxes": boxes[i][:, dn_pad:, :],
                }
                for i in range(self.num_decoder_layers - 1)
            ],
        }

        if dn_meta is not None:
            output["dn_outputs"] = {
                "pred_logits": logits[:, :, :dn_pad, :],
                "pred_boxes": boxes[:, :, :dn_pad, :],
                "meta": dn_meta,
            }

        return output


class HungarianMatcher(nn.Module):
    """Assign predictions to ground truth via bipartite matching."""

    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0, use_ciou_cost=False):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_ciou_cost = use_ciou_cost

    @torch.no_grad()
    def forward(self, outputs, targets):
        indices = []
        batch_size = outputs["pred_logits"].shape[0]

        for b_idx in range(batch_size):
            tgt_labels = targets[b_idx]["labels"]
            tgt_boxes = targets[b_idx]["boxes"]
            if tgt_labels.numel() == 0:
                indices.append(
                    (
                        torch.as_tensor([], dtype=torch.long),
                        torch.as_tensor([], dtype=torch.long),
                    )
                )
                continue

            probabilities = outputs["pred_logits"][b_idx].softmax(-1)
            pred_boxes = outputs["pred_boxes"][b_idx]

            cost_class = -probabilities[:, tgt_labels]
            cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)

            out_xyxy = box_cxcywh_to_xyxy(pred_boxes)
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
            if self.use_ciou_cost:
                cost_giou = -box_ciou(out_xyxy, tgt_xyxy)
            else:
                cost_giou = -generalized_box_iou(out_xyxy, tgt_xyxy)

            cost = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )

            src_idx, tgt_idx = linear_sum_assignment(cost.detach().cpu())
            indices.append(
                (
                    torch.as_tensor(src_idx, dtype=torch.long),
                    torch.as_tensor(tgt_idx, dtype=torch.long),
                )
            )

        return indices


class SetCriterion(nn.Module):
    """Compute classification and box regression losses."""

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        no_obj_weight=0.1,
        use_ciou_loss=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.use_ciou_loss = use_ciou_loss

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = no_obj_weight
        self.register_buffer("empty_weight", empty_weight)

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _dn_loss(self, dn_outputs):
        """Compute denoising losses averaged over decoder layers."""
        meta = dn_outputs["meta"]
        valid_mask = meta["valid_mask"]
        known_labels = meta["known_labels"]
        known_boxes = meta["known_boxes"]

        if not valid_mask.any():
            zero = dn_outputs["pred_logits"].sum() * 0.0
            return {
                "loss_dn_ce": zero,
                "loss_dn_bbox": zero,
                "loss_dn_giou": zero,
            }

        pred_logits = dn_outputs["pred_logits"]
        pred_boxes = dn_outputs["pred_boxes"]

        valid_count = valid_mask.sum().clamp(min=1).float()
        ce_total = pred_logits.sum() * 0.0
        bbox_total = pred_logits.sum() * 0.0
        giou_total = pred_logits.sum() * 0.0

        num_layers = pred_logits.shape[0]
        for layer_idx in range(num_layers):
            layer_logits = pred_logits[layer_idx]
            layer_boxes = pred_boxes[layer_idx]

            selected_logits = layer_logits[valid_mask]
            selected_labels = known_labels[valid_mask]
            ce_total = ce_total + F.cross_entropy(selected_logits, selected_labels)

            selected_boxes = layer_boxes[valid_mask]
            selected_targets = known_boxes[valid_mask]
            bbox_total = bbox_total + (
                F.l1_loss(selected_boxes, selected_targets, reduction="none").sum() / valid_count
            )

            out_xyxy = box_cxcywh_to_xyxy(selected_boxes)
            tgt_xyxy = box_cxcywh_to_xyxy(selected_targets)
            overlap = box_ciou(out_xyxy, tgt_xyxy) if self.use_ciou_loss else generalized_box_iou(
                out_xyxy,
                tgt_xyxy,
            )
            giou_total = giou_total + ((1.0 - torch.diag(overlap)).sum() / valid_count)

        denom = float(num_layers)
        return {
            "loss_dn_ce": ce_total / denom,
            "loss_dn_bbox": bbox_total / denom,
            "loss_dn_giou": giou_total / denom,
        }

    def _loss_for_single_output(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(target["labels"]) for target in targets)
        num_boxes = torch.as_tensor(
            [max(num_boxes, 1)],
            dtype=torch.float32,
            device=outputs["pred_logits"].device,
        )

        logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(
            logits.shape[:2],
            self.num_classes,
            dtype=torch.long,
            device=logits.device,
        )

        if len(idx[0]) > 0:
            target_classes_o = torch.cat(
                [target["labels"][j] for target, (_, j) in zip(targets, indices)],
            )
            target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(logits.transpose(1, 2), target_classes, self.empty_weight)
        loss_dict = {"loss_ce": loss_ce}

        if len(idx[0]) > 0:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat(
                [target["boxes"][j] for target, (_, j) in zip(targets, indices)],
                dim=0,
            )

            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / num_boxes

            out_xyxy = box_cxcywh_to_xyxy(src_boxes)
            tgt_xyxy = box_cxcywh_to_xyxy(target_boxes)
            overlap = box_ciou(out_xyxy, tgt_xyxy) if self.use_ciou_loss else generalized_box_iou(
                out_xyxy,
                tgt_xyxy,
            )
            loss_giou = (1.0 - torch.diag(overlap)).sum() / num_boxes
            loss_dict.update({"loss_bbox": loss_bbox, "loss_giou": loss_giou})

        return loss_dict

    def forward(self, outputs, targets):
        losses = self._loss_for_single_output(outputs, targets)

        for layer_idx, aux_outputs in enumerate(outputs.get("aux_outputs", [])):
            aux_losses = self._loss_for_single_output(aux_outputs, targets)
            for loss_name, value in aux_losses.items():
                losses[f"{loss_name}_{layer_idx}"] = value

        if "dn_outputs" in outputs:
            losses.update(self._dn_loss(outputs["dn_outputs"]))

        return losses


def prepare_for_coco(outputs, image_ids, coco_gt, score_threshold=0.3, nms_iou=0.4, max_dets=10):
    """Convert model outputs to COCO json-format detections."""
    probabilities = outputs["pred_logits"].softmax(-1)[..., :-1]
    boxes = outputs["pred_boxes"]

    all_results = []
    for batch_idx, image_id in enumerate(image_ids):
        image_id = int(image_id)
        image_info = coco_gt.imgs[image_id]
        width = image_info["width"]
        height = image_info["height"]

        image_probs = probabilities[batch_idx]
        image_boxes = boxes[batch_idx]

        scores, labels = image_probs.max(-1)
        keep = scores > score_threshold
        if keep.sum() == 0:
            continue

        scores = scores[keep]
        labels = labels[keep]
        image_boxes = image_boxes[keep]

        boxes_xyxy = box_cxcywh_to_xyxy(image_boxes)
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] * width).clamp(0, width)
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] * height).clamp(0, height)

        keep_nms = batched_nms(boxes_xyxy, scores, labels, iou_threshold=nms_iou)
        boxes_xyxy = boxes_xyxy[keep_nms]
        scores = scores[keep_nms]
        labels = labels[keep_nms]

        keep_topk = torch.argsort(scores, descending=True)[:max_dets]
        scores = scores[keep_topk]
        labels = labels[keep_topk]
        boxes_xywh = box_xyxy_to_xywh(boxes_xyxy[keep_topk])

        for score, label, box in zip(scores, labels, boxes_xywh):
            all_results.append(
                {
                    "image_id": image_id,
                    "bbox": [float(val) for val in box.cpu().tolist()],
                    "score": float(score.item()),
                    "category_id": int(label.item() + 1),
                }
            )

    return all_results


def summarize_detection_results(results):
    """Print quick summary stats for validation detections."""
    if not results:
        print("Validation detections summary: no detections exported.")
        return

    per_image_counts = {}
    scores = []
    category_counter = {}

    for det in results:
        image_id = det["image_id"]
        category_id = det["category_id"]

        per_image_counts[image_id] = per_image_counts.get(image_id, 0) + 1
        scores.append(det["score"])
        category_counter[category_id] = category_counter.get(category_id, 0) + 1

    count_values = list(per_image_counts.values())
    sorted_categories = sorted(category_counter.items(), key=lambda item: item[1], reverse=True)

    print(
        "Validation detections summary: "
        f"images_with_dets={len(per_image_counts)} "
        f"total_dets={len(results)} "
        f"avg_dets_per_image={sum(count_values) / len(count_values):.2f} "
        f"score_range=({min(scores):.4f}, {max(scores):.4f})"
    )
    print(f"  most_common_categories={sorted_categories[:5]}")


def main():
    """Main training entrypoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    finetune_checkpoint = None  # Example: "best_model_3.pth"
    finetune_strict = True

    dn_enabled = True
    dn_extra_epochs_after_unfreeze = -1
    dn_number = 2
    dn_label_noise_ratio = 0.2
    dn_box_noise_scale = 0.25

    image_size = (264, 440)
    num_epochs = 40
    batch_size = 8
    freeze_backbone_epochs = 3
    phase2_start_epoch = 15

    head_lr = 2e-4
    backbone_lr = 7e-6
    phase2_head_lr = 5e-5
    phase2_backbone_lr = 2e-6

    class_cost = 1.0
    no_object_weight = 0.1
    use_ciou_cost = False
    use_ciou_loss = True

    ce_weight = 1.0
    bbox_weight = 6.5
    giou_weight = 3.0

    phase2_class_cost = 1.5
    phase2_ce_weight = 1.5
    phase2_bbox_weight = 5.0
    phase2_giou_weight = 2.0

    val_score_threshold = 0.3
    val_nms_iou = 0.4
    val_max_dets = 10

    save_checkpoint = "best_model_3.pth"
    dn_disable_epoch = freeze_backbone_epochs + dn_extra_epochs_after_unfreeze

    print(
        f"Training setup: finetune_checkpoint={finetune_checkpoint} strict={finetune_strict} "
        f"dn_enabled={dn_enabled} dn_disable_epoch={dn_disable_epoch} "
        f"ciou_cost={use_ciou_cost} ciou_loss={use_ciou_loss} "
        f"phase2_start_epoch={phase2_start_epoch} "
        f"val_cfg=(thr={val_score_threshold}, nms={val_nms_iou}, max_dets={val_max_dets})"
    )

    train_dataset = DigitDataset("nycu-hw2-data/train", "nycu-hw2-data/train.json", transforms=None)
    train_dataset.transform = TrainTransform(image_size)

    val_dataset = DigitDataset("nycu-hw2-data/valid", "nycu-hw2-data/valid.json", transforms=None)
    val_dataset.transform = EvalTransform(image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    model = DigitDETR(num_classes=10, num_queries=10, num_decoder_layers=6).to(device)

    if finetune_checkpoint:
        if not os.path.isfile(finetune_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {finetune_checkpoint}")

        checkpoint = torch.load(finetune_checkpoint, map_location=device)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        incompatible = model.load_state_dict(state_dict, strict=finetune_strict)
        print(
            "Loaded checkpoint "
            f"{finetune_checkpoint}. "
            f"missing_keys={len(incompatible.missing_keys)}, "
            f"unexpected_keys={len(incompatible.unexpected_keys)}"
        )

    for param in model.backbone.parameters():
        param.requires_grad = False

    matcher = HungarianMatcher(
        cost_class=class_cost,
        cost_bbox=bbox_weight,
        cost_giou=giou_weight,
        use_ciou_cost=use_ciou_cost,
    )

    weight_dict = {
        "loss_ce": ce_weight,
        "loss_bbox": bbox_weight,
        "loss_giou": giou_weight,
    }
    for layer_idx in range(model.num_decoder_layers - 1):
        weight_dict[f"loss_ce_{layer_idx}"] = ce_weight
        weight_dict[f"loss_bbox_{layer_idx}"] = bbox_weight
        weight_dict[f"loss_giou_{layer_idx}"] = giou_weight

    weight_dict["loss_dn_ce"] = ce_weight
    weight_dict["loss_dn_bbox"] = bbox_weight
    weight_dict["loss_dn_giou"] = giou_weight

    criterion = SetCriterion(
        num_classes=10,
        matcher=matcher,
        weight_dict=weight_dict,
        no_obj_weight=no_object_weight,
        use_ciou_loss=use_ciou_loss,
    ).to(device)

    param_dicts = [
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if "backbone" not in name and param.requires_grad
            ]
        },
        {
            "params": [param for name, param in model.named_parameters() if "backbone" in name],
            "lr": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=head_lr * 0.5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6,
    )

    def apply_phase_settings(active_class_cost, active_ce, active_bbox, active_giou):
        matcher.cost_class = active_class_cost
        matcher.cost_bbox = active_bbox
        matcher.cost_giou = active_giou

        criterion.weight_dict["loss_ce"] = active_ce
        criterion.weight_dict["loss_bbox"] = active_bbox
        criterion.weight_dict["loss_giou"] = active_giou

        for layer_idx in range(model.num_decoder_layers - 1):
            criterion.weight_dict[f"loss_ce_{layer_idx}"] = active_ce
            criterion.weight_dict[f"loss_bbox_{layer_idx}"] = active_bbox
            criterion.weight_dict[f"loss_giou_{layer_idx}"] = active_giou

        criterion.weight_dict["loss_dn_ce"] = active_ce
        criterion.weight_dict["loss_dn_bbox"] = active_bbox
        criterion.weight_dict["loss_dn_giou"] = active_giou

    apply_phase_settings(class_cost, ce_weight, bbox_weight, giou_weight)
    in_phase2 = False
    best_map = 0.0

    for epoch in range(num_epochs):
        current_dn_enabled = dn_enabled and (epoch < dn_disable_epoch)

        if epoch == freeze_backbone_epochs:
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer.param_groups[1]["lr"] = backbone_lr

        if (not in_phase2) and epoch >= phase2_start_epoch:
            in_phase2 = True
            apply_phase_settings(
                phase2_class_cost,
                phase2_ce_weight,
                phase2_bbox_weight,
                phase2_giou_weight,
            )
            optimizer.param_groups[0]["lr"] = phase2_head_lr
            optimizer.param_groups[1]["lr"] = phase2_backbone_lr
            print(
                f"Switching to phase 2 at epoch {epoch + 1}: "
                f"class_cost={phase2_class_cost}, ce={phase2_ce_weight}, "
                f"bbox={phase2_bbox_weight}, giou={phase2_giou_weight}, "
                f"head_lr={phase2_head_lr}, backbone_lr={phase2_backbone_lr}"
            )

        model.train()
        if epoch < freeze_backbone_epochs:
            model.backbone.eval()

        processed_batches = 0
        running_total = 0.0
        running_ce = 0.0
        running_bbox = 0.0
        running_giou = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")
        for images, targets, _ in train_bar:
            images = images.to(device)
            targets = [{key: value.to(device) for key, value in tgt.items()} for tgt in targets]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(
                    images,
                    targets=targets,
                    dn_args={
                        "enabled": current_dn_enabled,
                        "dn_number": dn_number,
                        "label_noise_ratio": dn_label_noise_ratio,
                        "box_noise_scale": dn_box_noise_scale,
                    },
                )
                loss_dict = criterion(outputs, targets)
                loss = sum(
                    loss_dict[key] * criterion.weight_dict[key]
                    for key in loss_dict
                    if key in criterion.weight_dict
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
            scaler.step(optimizer)
            scaler.update()

            processed_batches += 1
            running_total += loss.item()
            running_ce += loss_dict.get("loss_ce", torch.tensor(0.0, device=device)).item()
            running_bbox += loss_dict.get("loss_bbox", torch.tensor(0.0, device=device)).item()
            running_giou += loss_dict.get("loss_giou", torch.tensor(0.0, device=device)).item()

            train_bar.set_postfix(
                total=f"{running_total / processed_batches:.3f}",
                ce=f"{running_ce / processed_batches:.3f}",
                bbox=f"{running_bbox / processed_batches:.3f}",
                giou=f"{running_giou / processed_batches:.3f}",
                dn=("on" if current_dn_enabled else "off"),
                phase=("p2" if in_phase2 else "p1"),
            )

        model.eval()
        val_results = []
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]")
            for images, _, image_ids in val_bar:
                images = images.to(device)
                outputs = model(images)
                val_results.extend(
                    prepare_for_coco(
                        outputs,
                        image_ids,
                        val_dataset.coco,
                        score_threshold=val_score_threshold,
                        nms_iou=val_nms_iou,
                        max_dets=val_max_dets,
                    )
                )

        scheduler.step()
        summarize_detection_results(val_results)

        if not val_results:
            print(f"Epoch {epoch + 1}: no detections above threshold.")
            continue

        coco_gt = val_dataset.coco
        coco_dt = coco_gt.loadRes(val_results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        current_map = coco_eval.stats[0]
        print(f"Epoch {epoch + 1} mAP: {current_map:.4f}")

        if current_map > best_map:
            best_map = current_map
            torch.save(model.state_dict(), save_checkpoint)
            print(f"New best mAP: {best_map:.4f}. Saved {save_checkpoint}")


if __name__ == "__main__":
    main()
