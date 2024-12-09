import warnings
import time
import os
import numpy as np
import argparse
from PIL import Image
import tflite_runtime.interpreter as tflite

# Ignore future warnings and deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-names', help='List of model names for classification', type=str, nargs='+', required=True)
    parser.add_argument('--captcha-dir', help='Directory containing CAPTCHA images', type=str, required=True)
    parser.add_argument('--output', help='File to save classification results', type=str, required=True)
    parser.add_argument('--symbols', help='File containing symbols used in CAPTCHAs', type=str, required=True)
    return parser.parse_args()

def decode(characters, predictions, length):
    # Define index orders based on CAPTCHA length
    index_orders = {
        1: [0],
        2: [1, 0],
        3: [1, 0, 2],
        4: [2, 0, 3, 1],
        5: [3, 1, 4, 2, 0],
        6: [4, 1, 5, 3, 0, 2]
    }
    # Get the index order for the specific length or use sequential order if not defined
    index_order = index_orders.get(length, range(length))
    
    # Reorder predictions based on index order for the specific length
    reordered_predictions = [predictions[i] for i in index_order]

    # Decode each character in the CAPTCHA
    y = np.argmax(np.array(reordered_predictions), axis=-1).flatten()
    decoded_text = ''.join([characters[x] for x in y])

    return decoded_text

def preprocess_image(image_path, input_shape):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((input_shape[2], input_shape[1]))
    image = np.array(image) / 255.0
    return image.astype(np.float32)

def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def get_prediction(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    return [interpreter.get_tensor(output_detail['index']) for output_detail in output_details]

def classify_captchas(args):
    start_time = time.time()
    with open(args.symbols, 'r') as symbols_file:
        captcha_symbols = symbols_file.readline().strip()

    models = [load_model(model_name) for model_name in args.model_names]

    with open(args.output, 'w') as output_file:
        for image_file in sorted(os.listdir(args.captcha_dir)):
            if not image_file.endswith('.png'):
                continue
            image_path = os.path.join(args.captcha_dir, image_file)
            try:
                # Preprocess the image
                image = preprocess_image(image_path, models[0].get_input_details()[0]['shape'])
                image = np.expand_dims(image, axis=0)

                # Initial classification with model 0 to determine CAPTCHA length
                prediction = get_prediction(models[0], image)
                initial_decoded = decode(captcha_symbols, prediction, len(prediction))  # infer length from model 0
                print("Initial result for {}: {}".format(image_file, initial_decoded))

                prediction = get_prediction(models[int(initial_decoded)], image)
                final_decoded = decode(captcha_symbols, prediction, int(initial_decoded))  # Use the inferred length
                print("Final result after model: {}".format(final_decoded))
    
                # Save result in the format imagename,Result
                output_file.write("{},{}\n".format(image_file, final_decoded))

            except Exception as e:
                print("Error processing {}: {}".format(image_file, e))

    total_time_taken = time.time() - start_time
    with open('classify_total_time.txt', 'w') as total_time_file:
        total_time_file.write("Total classification time: {:.2f} seconds\n".format(total_time_taken))
    print("Total classification time: {:.2f} seconds".format(total_time_taken))

if __name__ == '__main__':
    args = parse_arguments()
    classify_captchas(args)

