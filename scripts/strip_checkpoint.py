"""Strip optimizer state from a training checkpoint -> inference-only weights (~halves the size)."""
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
torch.save({"model_state": ck["model_state"], "config": "A3_rgb_plus_all5_content_mask",
            "epoch": ck.get("epoch"), "val_iou": ck.get("val_iou")}, dst)
print("wrote", dst)
