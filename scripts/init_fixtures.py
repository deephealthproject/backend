#!/bin/env python3

import os
import sys

sys.path.append(os.getcwd())

import gdown
from backend import settings
from os.path import join as opjoin

onnx_urls = [
    ('SegNet_Pneumothorax.onnx', 'https://drive.google.com/uc?id=132MiYbhqtLy_Q7_rInuFWhaukkINuabr'),
    ('UC12Segm_unet_224.onnx', 'https://drive.google.com/uc?id=1N0P6RRaC53ym5FT2YPz29SfRTSm3B-7_'),
    ('vgg16_in224_isic.onnx', 'https://drive.google.com/uc?id=1TRKffdOhsLdroGpDZSUx7QswnW1lBH_P'),
    ('lenet_mnist.onnx', 'https://drive.google.com/uc?id=1amyLCAtmVIw5wFpRX_JvwI3gi5B7B_--'),
    ('ResNet50-pytorch-imagenet.onnx', 'https://drive.google.com/uc?id=1dnn_4i7OYn4QhDVLhirQrchfNcksrkl8'),
]

if __name__ == "__main__":
    path = opjoin(settings.MODELS_DIR, 'onnx')
    os.makedirs(opjoin(path, 'pretrained'), exist_ok=True)
    # Download onnx from urls
    print('Downloading onnx...')
    for name, url in onnx_urls:
        gdown.download(url, opjoin(path, 'pretrained', name), quiet=True)

    # Change models location
    print('Setting up fixtures...')
    # Read in the file
    fixtures_path = opjoin(os.getcwd(), "backend_app", "fixtures")
    fnames = ["model.json", "modelweights.json"]
    for fname in fnames:
        with open(opjoin(fixtures_path, fname), 'r') as file:
            filedata = file.read()

        # Replace the target string
        filedata = filedata.replace('%MODELS_PATH%', str(path))

        # Write the file out again
        with open(opjoin(fixtures_path, fname), 'w') as file:
            file.write(filedata)
