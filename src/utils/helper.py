import numpy as np
import requests
import qai_hub as hub
import torch

def get_imagenet_categories():
    """ get_imagenet_categories
    
    Downloads the list of imagenet categories from a given URL and returns it as a list.
    The URL is a public asset hosted on AWS S3.

    """
    sample_classes = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/imagenet_classes.txt"
    response = requests.get(sample_classes, stream=True)
    response.raw.decode_content = True
    return [str(s.strip()) for s in response.raw]

# Nisam siguran da li dobro radi kada se ne posalji ndarray
def print_probablities_from_output(output, categories = None, top = 5, modelname = "", filename = ""):
    """ print_probabilities_from_output

    Accepts as input the output from the model before softmax,
    and prints the top 5 most probable classes with their probabilities.

    Parameters
    ----------
    output : torch.tensor or numpy.ndarray
        Output from the model before softmax, can be either a torch tensor or numpy array.
        If it's a torch tensor, it will be converted to numpy array.

    categories : list
        List of class names. If None, it will download the list from the internet.
        Default is None.

    top : int  
        Number of top predictions to print. Default is 5.

    modelname : str
        Name of the model. Used for printing purposes.

    filename : str
        Name of the file. Used for printing purposes.

    Returns
    -------

    None

    """
    if(type(output) != np.ndarray):
        output = output.cpu().detach().numpy()
    if categories is None:
        categories = get_imagenet_categories()
    probabilities = np.exp(output) / np.sum(np.exp(output), axis=1)
    # Print top five predictions for the on-device model
    print(f"Top-{top} predictions for {modelname} on {filename}:".format(top))
    top_classes = np.argsort(probabilities[0], axis=0)[-top:]
    for clas in reversed(top_classes):
        print(f"{clas} {categories[clas]:20s} {probabilities[0][clas]:>6.1%}")

def inference_job_probabilities(inference_job_object: hub.client.InferenceJob):
    """ inference job probabilities

    Takes in the inference job object and downloads the output data.
    It then prints the top 5 most probable classes with their probabilities.

    Parameters
    ----------

    inference_job_object : qai_hub.client.InferenceJob
        Inference job object that contains the output data.
        This object is created by the QAI Hub client and contains the results of the inference job.

    Returns
    -------

    None
    
    """
    on_device_output = inference_job_object.download_output_data()
    output_name = list(on_device_output.keys())[0]
    out = on_device_output[output_name][0]
    print_probablities_from_output(out, top=5, modelname="Cloud model")

import torchvision.transforms as transforms
from dataset.utils import CLASSES_IMAGENET
from matplotlib import pyplot as plt

def check_accuracy(model_eval, dataloader, device):
    """ check_accuracy

    Checks the accuracy of the model on the given dataloader.

    Parameters
    ----------

    model_eval : torch.nn.Module
        Model koji se evaluira

    dataloader : torch.utils.data.DataLoader
        Dataloader na kojem se evaluira (dataset)

    device : torch.device
        Device koji se koristi

    Returns
    -------

    accuracy : float
        Tacnost modela na datasetu
    
    """

    totalPost = 0
    accuracyPost = 0
    model_eval.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            totalPost += labels.size(0)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model_eval(images)
            _, predicted = torch.max(outputs, 1)
            accuracyPost += (predicted == labels).sum().item()
    
    return accuracyPost / totalPost
