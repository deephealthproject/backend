#!/bin/env python3

import os
import sys

sys.path.append(os.getcwd())

import requests
import zipfile
from backend import settings
from os.path import join as opjoin

if __name__ == "__main__":
    # Download the zip of weights
    url = "https://www.dropbox.com/sh/odt265yx3yrqpre/AAB_ap2xDt2G_LWv79TEFZw1a?dl=1"
    path = settings.MODELS_DIR
    headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
    r = requests.get(url, stream=True, headers=headers)
    print('Downloading...')
    with open(opjoin(path, "onnx.zip"), 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if not chunk:
                raise Exception("error in download")
            f.write(chunk)

    # Extract and delete it
    print('Extracting...')
    with zipfile.ZipFile(opjoin(path, "onnx.zip"), 'r') as zip:
        zip.extractall(opjoin(path, "onnx"))
    os.remove(opjoin(path, "onnx.zip"))

    # Change models location
    print('Setting up fixtures...')
    # Read in the file
    fixtures_path = opjoin(os.getcwd(), "backend_app", "fixtures")
    fnames = ["model.json", "modelweights.json"]
    for fname in fnames:
        with open(opjoin(fixtures_path, fname), 'r') as file:
            filedata = file.read()

        # Replace the target string
        filedata = filedata.replace('%MODELS_PATH%', str(opjoin(path, 'onnx')))

        # Write the file out again
        with open(opjoin(fixtures_path, fname), 'w') as file:
            file.write(filedata)
