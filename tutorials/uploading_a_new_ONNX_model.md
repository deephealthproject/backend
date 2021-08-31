# Uploading a new ONNX model
This tutorial shows steps for adding a new ONNX model into the backend.
Therefore, we plan to add a new ONNX for ResNet-50 model pretrained on ImageNet dataset and exported with PyTorch.
**NB ONNX files contain both neural network graph model definition and weights-biases.**

## Summary
The following table summarizes the actions needed to run our training process. Every row of the table is also detailed below.

| Action | Backend URL |
| - | - |
| Log in - Retrieve access token | auth/token/ |
| Choose or create a project, setting a task (e.g., classification) | projects/ |
| Retrieve all models for a task, choose/create one | models?task_id=1 |
| Upload the new ONNX from disk or from URL | weights/ |
| Grab the new weight | weights/ |


## The first two steps are exactly the same of the [Launch a training process](#finetuning-a-model) tutorial.

## Retrieve all models for a task, then choose/create one
The model entity represents a sort of grouping or family of the same neural network architecture. 
The `/models` API lists the models available for a task (classification `task_id=1`):
```json
GET models?task_id=1
[
  {
    "id": 1,
    "name": "LeNet",
    "task_id": 1
  },
  {
    "id": 2,
    "name": "VGG16",
    "task_id": 1
  }
]
```
We then can create a new family models named `ResNet50` using the `models` API:
```json
POST models
{
  "name": "ResNet50",
  "task_id": 1
}
```
Which returns:
```json
[
  {
    "id": 1,
    "name": "LeNet",
    "task_id": 1
  },
  {
    "id": 2,
    "name": "VGG16",
    "task_id": 1
  },
  {
    "id": 4,
    "name": "ResNet50",
    "task_id": 1
  }
]
```

## Upload the new ONNX from disk or from URL
Now we have a model family (`model_id=4`), and we are ready to upload a new ONNX linked to ResNet50. The `weights` API is the entrypoint we are going to use for creating such a weight. 

1. Uploading from local disk.
   
   This uploading method requires `multipart/form-data` data type, which is often used in website for sending data or submitting forms. We will show the form data uploading example using cURL:
   ```bash
   curl --location --request POST 'backend/weights' \
      --header 'Authorization: Bearer aAOdqqacNVEWCxoGOI91xxwmWv6KFI' \
      --form 'name="ResNet50-pytorch-imagenet"' \
      --form 'onnx_data=@"//mnt/onnx/ResNet50-pytorch-imagenet.onnx"' \
      --form 'model_id="4"' \
      --form 'task_id="1"' \
      --form 'layer_to_remove="Gemm_174"'
   ```

1. Uploading from URL.
   
   Alternatively the user can upload a new model providing a URL which points to a ONNX file. An example POST request could be like:   
   ```json
   POST weights
   {
     "name": "ResNet50-pytorch-imagenet",
     "onnx_url": "https://drive.google.com/u/1/uc?id=1jVVVgJcImHit9xkzxpu4I9Rho4Yh2k2H&export=download",
     "model_id": 4,
     "task_id": 1,
     "layer_to_remove": "Gemm_174"
   }
   ```
   This procedure may take a few minutes, since the backend has to download a potentially large ONNX weight. 
   Therefore, an asynchronous task for downloading the ONNX is spawned and the user can check the status of the 
   operation using the `process_id` returned by the API:
   ```json
   {
     "result": "ok",
     "process_id": "62a10034-1f2f-44b9-9014-71a1992647be"
   }
   ```
   
   The user can then check the upload status polling the `weightsStatus` API:
   ```json
   GET weightsStatus?process_id=62a10034-1f2f-44b9-9014-71a1992647be
   
   {
     "process_type": "Model Weight downloading",
     "result": "SUCCESS"
   }
   ```



## Grab the new weight
Once we have uploaded our new weight we can check its existence using the `weights` API.
We know we have uploaded a new ResNet50 (`id = 5`) weight, then the list of pretrainings is retrievable with:
```json
GET weights?model_id=5
[
  {
    "id": 15,
    "name": "ResNet50_skinlesion",
    "model_id": 5,
    "dataset_id": 3,
    "pretrained_on": null,
    "public": false,
    "users": [
      {
        "username": "dhtest",
        "permission": "OWN"
      }
    ],
    "layer_to_remove": null,
  },
  {    
    "id": 23,
    "name": "ResNet50-pytorch-imagenet",
    "model_id": 5,
    "dataset_id": null,
    "pretrained_on": null,
    "public": false,
    "users": [
      {
        "username": "dhtest",
        "permission": "OWN"
      }
    ],
    "layer_to_remove": "Gemm_174",
  },
  ...
]
```
Our new weight is identified by `id = 23` and it is ready to be used for finetuning on different datasets (e.g. on skin lesion domain images).