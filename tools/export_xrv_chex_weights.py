from pathlib import Path
import torch
import torchxrayvision as xrv

# Load pretrained CheX weights (this already works for you)
model = xrv.models.DenseNet(weights="densenet121-res224-chex")

# Make sure models/ exists
out_dir = Path("models")
out_dir.mkdir(parents=True, exist_ok=True)

# Save state_dict locally (offline forever after this)
out_path = out_dir / "xrv-densenet121-res224-chex.pth"
torch.save(model.state_dict(), out_path)

# Also save labels so you can display them reliably
labels_path = out_dir / "xrv_pathologies.txt"
labels = [p for p in model.pathologies if p.strip() != ""]
labels_path.write_text("\n".join(labels), encoding="utf-8")

print("Saved weights to:", out_path.resolve())
print("Saved labels to :", labels_path.resolve())
print("Num labels:", len(labels))
