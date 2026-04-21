"""
Inference script for NYCU HW2.
Loads a trained DigitDETR model and saves results to pred.json.
"""

import json
import os
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.ops import batched_nms
from tqdm import tqdm

from model import DigitDETR, box_cxcywh_to_xyxy

# Configuration Constants
SCORE_THRESHOLD = 0.001
MAX_DETS_PER_IMAGE = 8
WEIGHT_PATH = "best_model.pth"
TEST_DIR = "nycu-hw2-data/test"


def box_xyxy_to_xywh(x_input):
    """Converts xmin-ymin-xmax-ymax to xmin-ymin-width-height."""
    xmin, ymin, xmax, ymax = x_input.unbind(-1)
    return torch.stack([xmin, ymin, xmax - xmin, ymax - ymin], dim=-1)


def run_inference():
    """Performs model inference on test images and exports a JSON result."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitDETR(num_classes=10, num_queries=20).to(device)
    
    if not os.path.exists(WEIGHT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {WEIGHT_PATH}")
        
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device, weights_only=True))
    model.eval()

    image_size = (192, 320)
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_names = sorted(
        [f for f in os.listdir(TEST_DIR) if f.endswith(".png")],
        key=lambda x: int(x[:-4])
    )

    predictions = []
    for image_name in tqdm(image_names, desc="Inference"):
        image_id = int(image_name[:-4])
        image_path = os.path.join(TEST_DIR, image_name)

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)

        probs = output["pred_logits"].softmax(-1)[0, :, :-1]
        scores, labels = probs.max(-1)
        boxes = output["pred_boxes"][0]

        keep = scores > SCORE_THRESHOLD
        if keep.sum() == 0:
            continue

        s_filtered = scores[keep]
        l_filtered = labels[keep]
        b_filtered = boxes[keep]

        boxes_xyxy = box_cxcywh_to_xyxy(b_filtered)
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] * img_w).clamp(0, img_w)
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] * img_h).clamp(0, img_h)

        # NMS
        keep_nms = batched_nms(boxes_xyxy, s_filtered, l_filtered, iou_threshold=0.9)
        boxes_xyxy = boxes_xyxy[keep_nms]
        scores_nms = s_filtered[keep_nms]
        labels_nms = l_filtered[keep_nms]

        # Top-K
        keep_topk = torch.argsort(scores_nms, descending=True)[:MAX_DETS_PER_IMAGE]
        final_scores = scores_nms[keep_topk]
        final_labels = labels_nms[keep_topk]
        final_boxes_xywh = box_xyxy_to_xywh(boxes_xyxy[keep_topk])

        for score, label, box in zip(final_scores, final_labels, final_boxes_xywh):
            predictions.append({
                "image_id": image_id,
                "bbox": [float(v) for v in box.cpu().tolist()],
                "score": float(score.item()),
                "category_id": int(label.item() + 1),
            })

    with open("pred.json", "w", encoding="utf-8") as file:
        json.dump(predictions, file)

    print("Inference complete. Saved to pred.json")


if __name__ == "__main__":
    run_inference()