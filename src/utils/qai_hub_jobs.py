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
    """ compile_job
    
    Pravi job za kompajliranje modela na AIHUBU
    prost wrapper

    NOTE: treba dodati mejlove koji dele podatke kad se pokrene
    
    Parameters
    ----------

    model : torch.jit.ScriptModule or torch.nn or str
        
        1. vec trace-ovan model (torch.jit.ScriptModule)
        2. obican torch model (torch.nn) koji se onda trace-uje u ovoj fn
        3. string koji se zavrsava sa ".aimet"
            to je putanja do foldera gde je sacuvan aimet kvantizovan model 
            (npr. "model.aimet"), u njemu treba da postoji npr. "model.onnx" i "model.encodings".

            - ako ne radi proslediti absolute path

    name : str
        Ime modela, ovo ce da pise na qai hubu
        Ako je None, qai hub generise neko ime (bude ime klase)

    input_shape : tuple
        oblik ulazne slike, npr. (1, 3, 224, 224)

    Returns
    -------

    compile_job : hub.client.CompileJob
        Objekat koji vraca compilejob na qaihubu, u njemu je target model
    
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
    """ profile_job

    Wrapper, salje profile job na qaihub
    prima compile job object vraca profile job object

    Parameters
    ----------
    
    compile_job : hub.CompileJob
        Objekat koji vraca compilejob na qaihubu, u njemu je target model

    name : str
        Ime profile joba, ovo ce da pise na qaihubu
        Ako je None, generise se samo

    Returns
    -------

    profile_job : hub.ProfileJob
        Objekat koji vraca profile job na qaihubu, u njemu su vreme izvrsavanja, 
        i podaci o modelu generalno
    
    """

    model = compile_job.get_target_model()

    profile_job = hub.submit_profile_job(
        model=model,
        device=hub.Device("Snapdragon 8 Elite QRD"),
        name=name
    )

    return profile_job
    
def inference_job(compile_job: hub.CompileJob, input_array : np.ndarray):
    """ inference_job

    Wrapper, salje inference job na qaihub
    Prima compilejob objekat i sliku na kojoj radi inference, vraca inf job obj.

    Parameters
    ----------

    compile_job : hub.CompileJob
        Objekat koji vraca compilejob na qaihubu, u njemu je target model
        
    input_array : np.ndarray
        Slika na kojoj se radi inference, numpy array

    Returns
    -------

    inference_job : hub.InferenceJob
        Objekat koji vraca inference job na qaihubu, u njemu su rezultati inference-a
    
    """
    model = compile_job.get_target_model()

    inference_job = hub.submit_inference_job(
        model=model,
        device=hub.Device("Snapdragon 8 Elite QRD"),
        inputs=dict(image=[input_array])
    )
    return inference_job

def compile_profile_job(model, name = None, input_shape = (1, 3, 224, 224)):
    """ compile_profile_job

    Radi compile job i profile job u jednoj funkciji,
    pogledati funkcije compile_job() i profile_job()

    Parameters
    ----------

    model : torch.jit.ScriptModule or torch.nn or str
        model parametar koji se prosledjuje compile_job() funkciji

        1. vec trace-ovan model (torch.jit.ScriptModule)
        2. obican torch model (torch.nn) koji se onda trace-uje u ovoj fn
        3. string koji se zavrsava sa ".aimet"
            to je putanja do foldera gde je sacuvan aimet kvantizovan model 
            (npr. "model.aimet"), u njemu treba da postoji npr. "model.onnx" i "model.encodings".

            - ako ne radi proslediti absolute path

    name : str
        Ime modela, ovo ce da pise na qai hubu
        Ako je None, qai hub generise neko ime (bude ime klase)

    input_shape : tuple
        oblik ulazne slike, npr. (1, 3, 224, 224)
    
    Returns
    -------

    tuple(hub.CompileJob, hub.ProfileJob)
        Vraca compile job i profile job objekte
        
    """
   
    compile_job_object = compile_job(model, name, input_shape)
    profile_job_object = profile_job(compile_job_object, name)


    return compile_job_object, profile_job_object

def compile_profile_inference(model: torch.nn.Module, input_getter: input_getter.input_getter):
    """ compile_profile_inference

    Treba da uradi sva 3 u jednoj funkciji, koristi input_getter koji je sad malo deprecated

    """
    input_shape = input_getter.get_input_numpy().shape
    traced_model = torch.jit.trace(model, input_getter.get_input_torch())
    compile_job_object = compile_job(traced_model, input_shape)
    qai_model = compile_job_object.get_target_model()
    profile_job_object = profile_job(qai_model)
    inference_job_object = inference_job(qai_model, input_getter.get_input_numpy())

    return compile_job_object, profile_job_object, inference_job_object

def compile_profile_inference_tensor(model: torch.nn.Module, input):
    """ compile_profile_inference_tensor

    Radi compile prile i inference job ali ne koristi input_getter nego prima 
    input koji je torch tensor

    Parameters
    ----------

    model : torch.nn.Module
        Model koji se koristi za sve ove jobove
    
    input : torch.Tensor 
        Tenzor (slika) koji se koristi za inference
    
    Returns
    -------

    tuple(hub.CompileJob, hub.ProfileJob, hub.InferenceJob)
        Vraca compile job, profile job i inference job objekte

    """

    input_shape = input.cpu().numpy().shape
    traced_model = torch.jit.trace(model, input)
    compile_job_object = compile_job(traced_model, input_shape)
    qai_model = compile_job_object.get_target_model()
    profile_job_object = profile_job(qai_model)
    inference_job_object = inference_job(qai_model, input.cpu().numpy())

    return compile_job_object, profile_job_object, inference_job_object
