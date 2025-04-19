import torch
import numpy
import requests

pretrained_path = "./model/mobilenet_v2_coco.pth"  # Replace with your .pth file path

# Create the model
model = PreprocessedMobileNetV2(num_classes=num_classes, pretrained_weights_path=pretrained_path)

# Inference
model.eval()

# Trace model
input_shape: Tuple[int, ...] = (1, 3, 224, 224)
example_input = torch.rand(input_shape)
pt_model = torch.jit.trace(model, example_input)

# Compile model on a specific device
compile_job = qai_hub.submit_compile_job(
    pt_model,
    name="coco_imagenet", # Replace with your model name
    device=qai_hub.Device("Samsung Galaxy S24 (Family)"),
    input_specs=dict(image=input_shape),
)

compiled_model = compile_job.get_target_model()

# inference_output = run_inference(compiled_model, device = qai_hub.Device("Samsung Galaxy S24 (Family)"), input_dataset = qai_hub.get_dataset(""))
# output_array = inference_output['output_0']