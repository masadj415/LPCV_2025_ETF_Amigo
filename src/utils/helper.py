import numpy as np
import requests
import qai_hub as hub
import torch

def get_imagenet_categories():
    """ get_imagenet_categories
    
    Skida sa neta sve kategorije za imagenet i vraca listu stringova.
    Sluzi za testiranje modela pre fine-tune-ovanja na 64 klase

    """
    sample_classes = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/imagenet_classes.txt"
    response = requests.get(sample_classes, stream=True)
    response.raw.decode_content = True
    return [str(s.strip()) for s in response.raw]

# Nisam siguran da li dobro radi kada se ne posalji ndarray
def print_probablities_from_output(output, categories = None, top = 5, modelname = "", filename = ""):
    """ print_probabilities_from_output

    Prima output neke mreze, radi softmax i prettyprintuje ga prakticno

    Parameters
    ----------
    output : torch.tensor or numpy.ndarray
        Izlaz iz mreze pre softmaxa

    categories : list
        lista stringova sa imenima kategorija

    top : int  
        Broj najverovatnijih klasa koje se ispisuju

    modelname : str
        Ime modela, sluzi samo za print

    filename : str
        Ime fajla gde je model, sluzi samo za print

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

    Prima inference job object, ono sto vrati qai hub kad se izvrsi inf job, 
    verovatnoce kad se izvrsi kod njih :)
    Stampa njih lepo 

    Parameters
    ----------

    inference_job_object : qai_hub.client.InferenceJob
        Objekat koji vrati qai hub inference job

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

    Proverava tacnost modela na datasetu koji je u dataloader-u, 
    racuna top 1 tacnost

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
    
    transform = transforms.ToPILImage()
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            totalPost += labels.size(0)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model_eval(images)
            _, predicted = torch.max(outputs, 1)
            accuracyPost += (predicted == labels).sum().item()
            class_names = CLASSES_IMAGENET
            
            # Check for incorrect predictions
            incorrect_indices = (predicted != labels).nonzero(as_tuple=True)[0]
            
            # for idx in incorrect_indices:
            #     img = transform(images[idx].cpu())  # Convert tensor to image
            #     true_label = class_names[labels[idx].item()]
            #     pred_label = class_names[predicted[idx].item()]
                
            #     plt.imshow(img)
            #     plt.title(f"True: {true_label}, Predicted: {pred_label}")
            #     plt.axis("off")
            #     plt.show()
    
    return accuracyPost / totalPost
