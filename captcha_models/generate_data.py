#!/usr/bin/env python3

import argparse
import os
import random
import time
import math
import numpy as np
import cv2
import captcha.image
from tqdm import tqdm

def get_symbols(args):
    with open(args.symbols, 'r') as symbols_file:
        captcha_symbols = symbols_file.readline().strip()
    return captcha_symbols

# Parse arguments for captcha generation
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate captchas with specific parameters.")
    
    parser.add_argument('--width', help='Width of the generated captcha images', type=int, required=True)
    parser.add_argument('--height', help='Height of the generated captcha images', type=int, required=True)
    parser.add_argument('--min-length', help='Minimum character length of the captchas', type=int, required=True)
    parser.add_argument('--max-length', help='Maximum character length of the captchas', type=int, required=True)
    parser.add_argument('--count', help='Number of captchas to generate', type=int, required=True)
    parser.add_argument('--output-dir', help='Directory to store the generated captchas', type=str, required=True)
    parser.add_argument('--symbols', help='File containing the symbol set for captchas', type=str, required=True)
    
    args = parser.parse_args()
    captcha_symbols = get_symbols(args)
    return args, captcha_symbols

def main():
    start_time = time.time()
    
    args, captcha_symbols = parse_arguments()
    custom_fonts = ['fonts/Ring_of_Kerry.otf', 'fonts/The_Jjester.otf']
    train_dir = os.path.join(args.output_dir, 'train')
    validate_dir = os.path.join(args.output_dir, 'validate')
    
    for path in [train_dir, validate_dir]:
        os.makedirs(path, exist_ok=True)
    
    captcha_name_count = {}
    train_count = math.floor(0.9 * args.count)

    for i in tqdm(range(args.count), desc="Generating Captchas"):
        captcha_length = 1 if args.min_length == 1 and i < args.count else random.randint(args.min_length, args.max_length)
        random_str = ''.join(random.choice(captcha_symbols) for _ in range(captcha_length))
        image_dir = train_dir if i < train_count else validate_dir
        
        if random_str in captcha_name_count:
            captcha_name_count[random_str] += 1
            unique_str = f"{random_str}_{captcha_name_count[random_str]}"
        else:
            captcha_name_count[random_str] = 1
            unique_str = random_str
        
        unique_str = unique_str.replace('|', ',').replace('\\', ';')
        image_path = os.path.join(image_dir, unique_str + '.png')
        captcha_gen = captcha.image.ImageCaptcha(width=args.width, height=args.height, fonts=[random.choice(custom_fonts)])
        
        image = np.array(captcha_gen.generate_image(random_str))
        cv2.imwrite(image_path, image)

    print(f"Execution time for generating {args.count} captchas: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()

