import tensorflow as tf
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert a Keras model to TFLite format.")
    parser.add_argument("--model-name", type=str, required=True, help="Path to the Keras model file (e.g., 'model.h5').")
    parser.add_argument("--output", type=str, required=True, help="Name for the output TFLite model file (e.g., 'model.tflite').")
    return parser.parse_args()

def load_keras_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Successfully loaded Keras model.")
        print("Keras Model Output Shapes:")
        for output in model.outputs:
            print(output.shape)
        return model
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return None

def convert_to_tflite(model):
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        print("Successfully converted to TFLite model.")
        return tflite_model
    except Exception as e:
        print(f"Error during TFLite conversion: {e}")
        return None

def save_tflite_model(tflite_model, output_file):
    try:
        with open(output_file, "wb") as f:
            f.write(tflite_model)
        print(f"TFLite model saved to {output_file}")
    except Exception as e:
        print(f"Error saving TFLite model: {e}")

def verify_tflite_model(tflite_model):
    try:
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        output_details = interpreter.get_output_details()
        print("TFLite Model Output Details:")
        for output in output_details:
            print(output['shape'], output['dtype'])
    except Exception as e:
        print(f"Error verifying TFLite model: {e}")

def main():
    args = parse_arguments()
    
    model = load_keras_model(args.model_name)
    if model is None:
        return
    
    tflite_model = convert_to_tflite(model)
    if tflite_model is None:
        return
    
    save_tflite_model(tflite_model, args.output)
    verify_tflite_model(tflite_model)

if __name__ == "__main__":
    main()

