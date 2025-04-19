import torch
from torchvision import models, transforms
import qai_hub
from typing import Tuple

# Custom wrapper class for preprocessing and MobileNetV2
class PreprocessedMobileNetV2(torch.nn.Module):
    def __init__(self, num_classes, pretrained_weights_path):
        super(PreprocessedMobileNetV2, self).__init__()
        # Load MobileNetV2 with the specified number of classes
        self.mobilenet_v2 = models.mobilenet_v2(num_classes=num_classes)

        # Load pretrained weights from .pth file
        state_dict = torch.load(pretrained_weights_path, weights_only=True)
        self.mobilenet_v2.load_state_dict(state_dict)

        # Define preprocessing steps
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, img):
        # Apply preprocessing
        if isinstance(img, torch.Tensor) and len(img.shape) == 4:
            # If already a batch tensor, skip preprocessing
            img_tensor = img
        else:
            # Preprocess a PIL image or similar input
            img_tensor = self.preprocess(img).unsqueeze(0)  # Add batch dimension

        # Pass the preprocessed image through the model
        return self.mobilenet_v2(img_tensor)

def run_inference(model, device, input_dataset):
    """Submit an inference job for the model."""
    inference_job = qai_hub.submit_inference_job(
        model=model,
        device=device,
        inputs=input_dataset,
        options="--max_profiler_iterations 1"
    )
    return inference_job.download_output_data()

# Parameters
num_classes = 64

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor()          # Convert to tensor
])

