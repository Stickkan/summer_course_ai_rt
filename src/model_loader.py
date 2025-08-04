# import tflite_runtime.interpreter as tf  # embedded runtime
import tensorflow as tf


class Model:
    def __init__(self, model_path: str, logger=None):
        # self.interpreter = tf.interpreter(model_path=model_path)
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.logger = logger
        print("TFLite model loaded.")


    def get_output_state(self, input_data):
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        return self.interpreter.get_tensor(self.output_details[0]['index'])[0]
