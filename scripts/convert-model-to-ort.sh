#!/bin/bash

# copied from https://github.com/olilarkin/ort-builder

#python -m tf2onnx.convert --saved-model model --output model.onnx --opset 13
python -m onnxruntime.tools.convert_onnx_models_to_ort $1 --enable_type_reduction

# skip the part where we create h and c code, 1.2GB c file is too ridiculous

#mkdir -p ./model
#
## extract filename from $1 argument
#filename=$(basename -- "$1")
#
## remove extension from filename
#filename="${filename%.*}"
#
## python verify_model.py
##python -m bin2c -o ./model/${filename}.ort ${filename}.ort
