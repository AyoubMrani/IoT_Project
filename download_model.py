import torch
import torchvision.models as models
import os

print("📥 Downloading pre-trained MobileNetV2 model...")
# This single line downloads the pre-trained weights automatically
model = models.mobilenet_v2(weights='DEFAULT')
print("✅ Model downloaded successfully!")

print("🔧 Modifying classifier layer for 5 classes (LC25000 dataset)...")
# IMPORTANT: MobileNetV2 was trained for 1000 classes. 
# You must change the final layer to '5' for your LC25000 dataset:
model.classifier[1] = torch.nn.Linear(model.last_channel, 5)
print("✅ Classifier layer modified!")

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the model
model_path = "models/original_mobilenet_v2_5classes.pt"
torch.save(model.state_dict(), model_path)
print(f"💾 Model saved to: {model_path}")

print("\n📊 Model Summary:")
print(f"Architecture: MobileNetV2")
print(f"Output classes: 5")
print(f"Classifier layer: {model.classifier[1]}")
