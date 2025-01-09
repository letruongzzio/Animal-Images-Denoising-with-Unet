import torch
import os
import sys
PROJECT_ROOT = os.path.expanduser("~/Animal-Images-Denoising-with-Unet/model")
sys.path.append(PROJECT_ROOT)
import model.animal_dataset
from model.unet_model import UNet
from model.modelPipeline import evaluate, display_prediction
from constants import MODEL_DIRNAME


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = UNet(in_channels=3, out_channels=3).to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_DIRNAME, "storage/trained_unet.pth"), weights_only=True))
criterion = torch.nn.MSELoss()

# Load the test loader
test_loader = torch.load(os.path.join(MODEL_DIRNAME, "storage/test_loader.pth"))

# Evaluate the model
test_loss = evaluate(model, test_loader, criterion, device, val_display=False)
print(f"Test Loss: {test_loss:.4f}")

# Display test predictions
storage_images = set()
for i in range(3):
    input_img, target_img = test_loader.dataset[i]
    if input_img in storage_images:
        continue
    storage_images.add(input_img)
    display_prediction(
        model=model,
        image=input_img.unsqueeze(0).to(device),
        target=target_img.unsqueeze(0).to(device),
        device=device,
        title="test_prediction"
   )
