import tensorflow.lite as tflite

def download_model_from_compile_job(compile_job, download_path):
    if(not download_path.endswith(".tflite")):
        download_path += ".tflite"
    target_model = compile_job.get_target_model()
    target_model.download(download_path)
    return download_path

class TFHelper:
    """ TFHelper

    Helper class for working with TFLite models.
    After a compile job on QAI Hub, the model returned from `.get_target_model()` 
    is a `TFLiteModel` object, which can be downloaded as a `.tflite` file.
    This class is used for working with those files — i.e., for loading and running inference.

    Pass the output of `download_model_from_compile_job` to the constructor,
    which takes a compile job object and a path where the model will be saved.
    """

    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        
    def get_interpreter(self):
        return self.interpreter
    
    def run_inference(self, input_array):
        """ run_inference

        Runs inference on the model downloaded from QAI Hub, locally.
        Takes a NumPy array (image) and returns a NumPy array (inference results).
        The results likely won’t match those from QAI Hub exactly.
        """
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'], input_array)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        return output_data
