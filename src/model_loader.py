# import tflite_runtime.interpreter as tf  # embedded runtime
import tensorflow as tf
import time



class Model:
    def __init__(self, model_path: str, logger=None):
        # self.interpreter = tf.Interpreter(model_path=model_path)
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.logger = logger
        self.time = time.time()
        print("TFLite model loaded.")


    def get_output_state(self, input_data):
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        if self.logger is not None:
            #calculate elapsed time in seconds
            elapsed_time = time.time() - self.time
            self.logger(output_data=output_data, time=elapsed_time)

        return output_data
