import tensorflow.lite as tflite

def download_model_from_compile_job(compile_job, download_path):
    if(not download_path.endswith(".tflite")):
        download_path += ".tflite"
    target_model = compile_job.get_target_model()
    target_model.download(download_path)
    return download_path

class TFHelper:
    """ TFHelper
    
    Pomocna klasa za rad sa TFLite modelima.
    Posle compile job-a na qai hub-u model koji dobijemo preko .get_target_model() je 
    TFLiteModel objekat, koji se moze download-ovati kao .tflite fajl.
    Ova klasa sluzi za rad sa tim fajlovima, tj. za ucitavanje i pokretanje inference-a.

    Konstruktoru proslediti izlaz funkcije download_model_from_compile_job
    kojoj se daje compile job objekat i putanja na kojoj ce se sacuvati model.
    

    """

    def __init__(self, model_path):

        self.interpreter = tflite.Interpreter(model_path=model_path)
        
    def get_interpreter(self):
        return self.interpreter
    
    def run_inference(self, input_array):
        """ run_inference

        Pokrene inference na modelu skinutom sa qai hub-a, lokalno
        Prima numpy array(sliku), vraca numpy array(rezultat inference-a)
        Rezultati se verovatno nece poklapati sa onima na qaihub-u
        
        """
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'], input_array)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        return output_data
