# Wrapper za pravljenje prostih jobova za AIHUB, trenutno samo kompajliranje, profiliranje i inferencija

# Probacu ovo da promenim da radi kao compile job

import qai_hub as hub
import torch
import utils.input_getter as input_getter
import numpy as np
import copy
import requests
import utils.helper as helper



def compile_job(model, name : str = None, input_shape = (1, 3, 224, 224)):
    """
    compile_job

    Creates a compile job on the QAI Hub.
    This function takes in a model and submits it to the QAI Hub for compilation.
    It can handle different types of models, including traced models and AIMET quantized models.
    It also allows for the specification of the input shape and the name of the model.

    Parameters
    ----------
    model : torch.jit.ScriptModule or torch.nn.Module or str
        1. A traced model (torch.jit.ScriptModule)
        2. A regular PyTorch model (torch.nn.Module), which will be traced inside this function
        3. A string ending in ".aimet"
        This is the path to a folder where an AIMET quantized model is saved
        (e.g., "model.aimet"). It should contain "model.onnx" and "model.encodings".

        - If it doesn't work, try passing an absolute path.

    name : str
        Name of the model, will be displayed on QAI Hub.
        If None, QAI Hub will generate a name (usually the class name).

    input_shape : tuple
        Shape of the input image, e.g., (1, 3, 224, 224)

    Returns
    -------
    compile_job : hub.client.CompileJob
        The CompileJob object returned from QAI Hub, which contains the target model
    """

    
    # aimet model
    if type(model) == str:
        name = name + " quantized"
        compile_job = hub.submit_compile_job(
            model=model,
            device=hub.Device("Snapdragon 8 Elite QRD"),
            # input_specs=dict(image=input_shape),
            name=name
        )

    elif type(model) == torch.jit.ScriptModule:
        compile_job = hub.submit_compile_job(
            model=model,
            device=hub.Device("Snapdragon 8 Elite QRD"),
            input_specs=dict(image=input_shape),
            name=name
        )

    elif type(model) == torch.nn.Module:
        traced_model = torch.jit.trace(model, torch.rand(input_shape))

        compile_job = hub.submit_compile_job(
            model=traced_model,
            device=hub.Device("Snapdragon 8 Elite QRD"),
            input_specs=dict(image=input_shape),
            name=name
        )
        
    return compile_job

def profile_job(compile_job: hub.CompileJob, name : str = None):
    """
    profile_job

    Wrapper that sends a profile job to QAI Hub.
    Takes a compile job object and returns a profile job object.

    Parameters
    ----------
    compile_job : hub.CompileJob
        The CompileJob object returned from QAI Hub, containing the target model

    name : str
        Name of the profile job, will be displayed on QAI Hub.
        If None, a name will be auto-generated.

    Returns
    -------
    profile_job : hub.ProfileJob
        The ProfileJob object returned from QAI Hub, which includes execution times
        and general model info
    """


    model = compile_job.get_target_model()

    profile_job = hub.submit_profile_job(
        model=model,
        device=hub.Device("Snapdragon 8 Elite QRD"),
        name=name
    )

    return profile_job
    
def inference_job(compile_job: hub.CompileJob, input_array : np.ndarray):
    """
    inference_job

    Wrapper that sends an inference job to QAI Hub.
    Takes a compile job object and an input image for inference, and returns the inference job object.

    Parameters
    ----------
    compile_job : hub.CompileJob
        The CompileJob object returned from QAI Hub, containing the target model

    input_array : np.ndarray
        The image to be used for inference, as a NumPy array

    Returns
    -------
    inference_job : hub.InferenceJob
        The InferenceJob object returned from QAI Hub, containing inference results
    """

    model = compile_job.get_target_model()

    inference_job = hub.submit_inference_job(
        model=model,
        device=hub.Device("Snapdragon 8 Elite QRD"),
        inputs=dict(image=[input_array])
    )
    return inference_job

def compile_profile_job(model, name = None, input_shape = (1, 3, 224, 224)):
    """
    compile_profile_job

    Performs both compile and profile jobs in one function.
    See the functions compile_job() and profile_job() for details.

    Parameters
    ----------
    model : torch.jit.ScriptModule or torch.nn.Module or str
        The model passed to compile_job()

        1. A traced model (torch.jit.ScriptModule)
        2. A regular PyTorch model (torch.nn.Module), which will be traced in this function
        3. A string ending in ".aimet"
        This is the path to a folder where an AIMET quantized model is saved
        (e.g., "model.aimet"). It should contain "model.onnx" and "model.encodings".

        - If it doesn't work, try passing an absolute path.

    name : str
        Name of the model, will be displayed on QAI Hub.
        If None, QAI Hub will generate a name (usually the class name)

    input_shape : tuple
        Shape of the input image, e.g., (1, 3, 224, 224)

    Returns
    -------
    tuple(hub.CompileJob, hub.ProfileJob)
        Returns the CompileJob and ProfileJob objects
    """

   
    compile_job_object = compile_job(model, name, input_shape)
    profile_job_object = profile_job(compile_job_object, name)


    return compile_job_object, profile_job_object

def compile_profile_inference(model: torch.nn.Module, input_getter: input_getter.input_getter):
    """
    compile_profile_inference

    Performs compile, profile, and inference in a single function.
    Uses an input_getter, which is now somewhat deprecated.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be compiled, profiled, and used for inference

    input_getter : input_getter.input_getter
        Object that provides input tensors in both NumPy and Torch formats

    Returns
    -------
    tuple(hub.CompileJob, hub.ProfileJob, hub.InferenceJob)
        Returns CompileJob, ProfileJob, and InferenceJob objects
    """
    input_shape = input_getter.get_input_numpy().shape
    traced_model = torch.jit.trace(model, input_getter.get_input_torch())
    compile_job_object = compile_job(traced_model, input_shape)
    qai_model = compile_job_object.get_target_model()
    profile_job_object = profile_job(qai_model)
    inference_job_object = inference_job(qai_model, input_getter.get_input_numpy())

    return compile_job_object, profile_job_object, inference_job_object

def compile_profile_inference_tensor(model: torch.nn.Module, input):
    """
    compile_profile_inference_tensor

    Performs compile, profile, and inference jobs using a provided torch tensor input
    (instead of input_getter).

    Parameters
    ----------
    model : torch.nn.Module
        The model to be used for all three jobs

    input : torch.Tensor 
        Tensor (image) used for inference

    Returns
    -------
    tuple(hub.CompileJob, hub.ProfileJob, hub.InferenceJob)
        Returns CompileJob, ProfileJob, and InferenceJob objects
    """

    input_shape = input.cpu().numpy().shape
    traced_model = torch.jit.trace(model, input)
    compile_job_object = compile_job(traced_model, input_shape)
    qai_model = compile_job_object.get_target_model()
    profile_job_object = profile_job(qai_model)
    inference_job_object = inference_job(qai_model, input.cpu().numpy())

    return compile_job_object, profile_job_object, inference_job_object
