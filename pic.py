from PIL import Image
import torch
import os
from torchvision.models import resnet18, ResNet18_Weights

# Get image path from user
image_path = input("Enter image path: ").strip()

# Remove quotes
image_path = image_path.strip('"').strip("'")

# Convert Windows backslashes to forward slashes
image_path = image_path.replace("\\", "/")

# Convert Windows path to WSL path
if image_path.lower().startswith("c:/"):
    image_path = "/mnt/c/" + image_path[3:]

print("Using:", image_path)
print("Exists:", os.path.exists(image_path))

# Load image
img = Image.open(image_path).convert("RGB")

# Load model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()

# Preprocess image
preprocess = weights.transforms()
img_tensor = preprocess(img).unsqueeze(0)

# Run inference
with torch.no_grad():
    outputs = model(img_tensor)

# Show top prediction
prediction_idx = outputs.argmax(dim=1).item()
categories = weights.meta["categories"]

print("Prediction:", categories[prediction_idx])

# Show top 5 predictions
probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)

print("\nTop 5 predictions:")
for i in range(5):
    print(
        f"{categories[top5_catid[i]]}: "
        f"{top5_prob[i].item() * 100:.2f}%"
    )