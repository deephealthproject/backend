# Inference with our data
This tutorial is intended for who wants to use pretrained models and use them with his own data.
The following tutorial will exploits an already trained ResNet50 architecture testing it on a sample skin lesion image. The tutorial also shows the possibility to test the model on a different set of data, for example a private dataset of skin lesion.

## Summary
The following table summarizes the actions needed to run our training process. Every row of the table is also detailed below.

| Action | Backend URL |
| - | - |
| Log in - Retrieve access token | auth/token/ |
| Choose or create a project, setting a task (e.g., classification) | projects/ |
| Retrieve all models for a task and choose one | models?task_id=1 |
| Choose an existing pretraining of that model | weights?model_id=1 |
| Choose a dataset for the same task | datasets?task_id=1 |
| Launch the inference | inference |
| Get predictions of the model | outputs?process_id=624404a8-ba15-420c-8197-ead8f584b796 |


## The first three steps are exactly the same of the [Launch a training process](#finetuning-a-model) tutorial.

## Select the pretrained architecture
We must choose a model pretrained on the skin lesion classification task if we want to obtain useful results.
We decide to look for a weight of the model ResNet50 (`id = 5`). Then the list of pretrainings is retrievable with:
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
    ]
  },
  ...
]
```
The weight with `id = 15` is what we need for the inference task.


## Inferencing
1. _Inference on a sample image_
  
   We then decide to test the model with a sample image. In this example we use a PNG image, but the DeepHealth Toolkit supports a wide variety of image formats (e.g. JPG, TIFF, DICOM, NIfTI, etc.).

    ```json
    POST inferenceSingle
    {
      "modelweights_id": 15,
      "image_url": "https://url-to-skin-lesion.png",
      "project_id": 1
    }
    ```

2. _Inference on a different dataset_
  
   Another option could be testing our trained model on a dataset in the backend.
    ```json
    POST inference
    {
      "modelweights_id": 15,
      "dataset_id": 3,
      "project_id": 1
    }
    ```
    This requests automatically launch a inference process on the test set of a dataset, which in this case is the same we used for training the network (`dataset_id = 3`).

Both inference APIs reply with:
```json
{
  "result": "ok",
  "process_id": "338fdac5-89a1-4f89-8e74-64423a99a0d8"
}
```
The `process_id` field is what we need for querying the backend for task monitoring and output results.
So polling the `status` API we can get back the status of our inference:
```json
GET status?process_id=338fdac5-89a1-4f89-8e74-64423a99a0d8&full=false
[
  {
    "result": "ok",
    "status": {
      "process_type": "testing",
      "process_status": "running",
      "process_data": "Test - epoch [42/70] - batch [268/773] - metric=0.949"
    }
  }
]
```

## Visualization of results
Once our inference process has terminated we can proceed with outputs extraction. 

The `output` API can be invoked using:
```json
GET output?process_id=338fdac5-89a1-4f89-8e74-64423a99a0d8
{
  "outputs": [
    [
      "/mnt/data/DATA/isic_skin_lesion/images/ISIC_0000000.jpg",
      "[[1.16545313275712e-06, 5. 235364753687456e-08, 7.840961188776419e-06, 0.00021923755411989987, 1.5176419765339233e-07, 3.19655964631238e-06, 2.1568662411652895e-09, 0.9993834495544434]]"
    ]
  ]
}
```
The output API changes its behavior depending on the task, for our example (_classification_ task) it returns a list of tuples `(image_name, predictions_scores)`, where each position of `predictions_scores` encodes the score of a class.

In case of _segmentation_ task the outputs will return a list of URL tuples `(image_name, segmented_image)`.