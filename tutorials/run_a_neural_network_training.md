# Train a neural network
This tutorial shows the steps required to start a training process, loading an existing model already trained on a dataset (e.g., trained on ImageNet) and finetuning it on a new one.

For this specific tutorial we show the steps for training a classification network based on ResNet50 model fed with skin lesion images. At the end of the training process our neural networks will be able to classify the different skin lesion types of the dataset.

## Summary
The following table summarizes the actions needed to run our training process. Every row of the table is also detailed below.

| Action | Backend URL |
| - | - |
| Log in - Retrieve access token | auth/token/ |
| Choose or create a project, setting a task (e.g., classification) | projects/ |
| Retrieve all models for a task and choose one | models?task_id=1 |
| Choose a dataset of the same task | datasets?task_id=1 |
| Finetuning: choose an existing pretraining of that model | weights?model_id=1 |
| Retrieve the properties (hyperparameters) | properties/ |
| Check what are the default properties for a model-dataset | allowedProperties?property_id=1&model_id=1&dataset_id=1 |
| Launch training | train |
| Monitor training status | status?process_id=624404a8-ba15-420c-8197-ead8f584b796 |

## Authenticate
Authentication is mandatory for interacting within the backend, because it guarantees that the privacy of the uploaded is limited to a small number of user and not visible to the public.
We must call the `auth/token/` if we want authenticate using the built-in authorization server, passing the required fields:
```
grant_type=password&username=<user_name>&password=<password>&client_id=<client_id>
```
The user_name and password are the credential of our user. Response is something like:

```json
{
  "access_token": "<your_access_token>",
  "token_type": "Bearer",
  "expires_in": 36000,
  "refresh_token": "<your_refresh_token>",
  "scope": "read write"
}
```

**We must copy and use this token attaching it to each request to the backend.**
Example using curl: `curl -H "Authorization: Bearer <your_access_token>" http://mysite/backend/aMagicalAPI`


## Choose a project
Every process (a training or a inference) in the backend is bound to a project. So within a project we can start different trainings for instance enhancing, fixing, testing different combination of hyperparameter, until we reach our expected results.
Every project has a name and an associated task. If a project is associated to a specific task, for example classification, every request we makes within that project will serve information about classification task, like models which have been designed for classification but not for segmentation.

A project can also be shared and the `users` field lists the users whose have access too.

Calling `/projects` we access to all our created projects:
```json
GET projects
[{
  "id": 1,
  "name": "testProject",
  "task_id": 1,
  "users": [
    {
      "username": "dhtest",
      "permission": "OWN"
    }
  ]
}]
```
We proceed using the project with `id = 1`, which refers to classification task. We can verify that calling the `/tasks` API, that returns the tasks supported by the backend:
```json
GET tasks
[
  {
    "id": 1,
    "name": "Classification"
  },
  {
    "id": 2,
    "name": "Segmentation"
  }
]
```

## Retrieve the models
The model entity represents a sort of grouping or family of the same neural network architecture. 
The `/models` API lists the models available for a task (classification `task_id=1`):
```json
GET models?task_id=1
[
  {
    "id": 1,
    "name": "LeNet",
    "task_id": 1,
  },
  {
    "id": 2,
    "name": "VGG16",
    "task_id": 1,
  },
  {
    "id": 5,
    "name": "ResNet50",
    "task_id": 1,
  }
]
```
For this tutorial we decide we want classify using [ResNet50](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) architecture, therefore `id = 5`.  
## Which dataset do we want to use for training?
The backend stores different kind of dataset, which can  be uploaded by the users. We can fetch all available datasets for classification with:
```json
GET datasets?task_id=1
[
  {
    "id": 1,
    "name": "MNIST",
    "path": "/mnt/data/DATA/mnist/mnist.yml",
    "task_id": 1,
    "users": [],
    "public": true,
  },
  {
    "id": 3,
    "name": "ISIC classification",
    "path": "/mnt/data/DATA/isic_skin_lesion/isic.yml",
    "task_id": 1,
    "users": [],
    "public": true,
  },
  ...
]
```
The `id = 3` is a dataset for skin lesion classification and is the one want to use for training.

## Select the pretraining
Once we have defined which model to use, we can obtain a pretraining of that model. The `weights` API lists all the available pretrainings -- ONNX model with relative weights -- of a specific model.
```json
GET weights?model_id=5
[
  {
    "id": 12,
    "name": "ResNet50_Imagenet",
    "model_id": 5,
    "dataset_id": 7,
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
A pretraining consists of a ONNX file belonging to a specific user (_dhtest_ in this example), which can be shared among users or set public.
We decide to use the pretraining with `id=12` which has been pretrained on ImageNet and finetune it for our skin lesion classification task.


## Retrieve the properties (hyperparameters)
Every time a data scientist wants to launch a learning process he must decide also a reasonable value to assign to every hyperparameters.
Before launching a training process we must collect the properties (or hyperparameters) available on the backend. After knowing their possible values we can attach them to training request. 

The properties are retrieved through:
```json
GET properties
[
  {
    "id": 1,
    "name": "Learning rate",
    "type": "float",
    "default": "0.0001",
    "values": null
  },
  {
    "id": 2,
    "name": "Loss function",
    "type": "list",
    "default": "cross_entropy",
    "values": "cross_entropy,softmax_cross_entropy,mean_squared_error,binary_cross_entropy,dice"
  },
  {
    "id": 3,
    "name": "Epochs",
    "type": "integer",
    "default": "50",
    "values": null
  },
  ...
]
```
Every property has its own name, a type useful for parsing purposes, a global default value, and a global list of allowed values.

Moreover, the backend offers the `allowedProperties` API, which helps to get default and allowed values for a certain property-model[-dataset].
For example, if we want to discover the base image input size for skin lesion classification dataset and ResNet50 we can do:
```json
GET allowedProperties?property_id=7&model_id=5&dataset_id=3
[
  {
    "id": 13,
    "allowed_value": null,
    "default_value": "224",
    "property_id": 7,
    "model_id": 5,
    "dataset_id": 3
  }
]
```
Therefore, we first retrieve all the properties and their global values, and after that, we look for specific custom value of our chosen model-dataset.
Clearly, an expert user can discard the value proposed by the backend for `Learning rate` providing a different value that appears reasonable to him.

## Launch a training
We have collected all the information for launching a training process.
```json
POST train
{
  "project_id": 1,
  "dataset_id": 3,
  "weights_id": 12,
  "properties": [
    {
      "name": "Learning rate",
      "value": "1e-4"
    },
    {
      "name": "Loss function",
      "value": "softmax_cross_entropy"
    },
    {
      "name": "Metric",
      "value": "accuracy"
    },
    {
      "name": "Epochs",
      "value": "50"
    },
    {
      "name": "Input height",
      "value": "224"
    },
    {
      "name": "Input width",
      "value": "224"
    },
    {
      "name": "batch size",
      "value": "16"
    },
    {
      "name": "Training augmentations",
      "value": "SequentialAugmentationContainer\n    AugResizeDim dims=(224,224) interp=\"cubic\"\n    AugMirror p=0.5\n    AugFlip p=0.5\n    AugRotate angle=[-180,180]\n    AugAdditivePoissonNoise lambda=[0,10]\n    AugGammaContrast gamma=[0.5,1.5]\n    AugGaussianBlur sigma=[0.0,0.8]\n    AugCoarseDropout p=[0,0.03] drop_size=[0.0,0.05] per_channel=0.25\n    AugToFloat32 divisor=255\n    AugNormalize mean=(0.6681, 0.5301, 0.5247) std=(0.1337, 0.1480, 0.1595)\nend"
    },
    {
      "name": "Validation augmentations",
      "value": "SequentialAugmentationContainer\n    AugResizeDim dims=(224,224) interp=\"cubic\"\n    AugToFloat32 divisor=255\n    AugNormalize mean=(0.6681, 0.5301, 0.5247) std=(0.1337, 0.1480, 0.1595)\nend"
    },
    {
      "name": "Test augmentations",
      "value": "SequentialAugmentationContainer\n    AugResizeDim dims=(224,224) interp=\"cubic\"\n    AugToFloat32 divisor=255\n    AugNormalize mean=(0.6681, 0.5301, 0.5247) std=(0.1337, 0.1480, 0.1595)\nend"
    }
  ],
}
```
In short this _train_ request says: finetune the ResNet50 model (`model_id = 5`) pretrained on ImageNet (`weights_id = 12`) for skin lesion classification (`dataset_id = 3`). The hyperparameters, loss, and metric chosen are set using the `properties` list field.
Our learning process is launched an asynchronous process then the backend replies with:
```json
{
  "result": "ok",
  "process_id": "62a10034-1f2f-44b9-9014-71a1992647be"
}
```
**Our training process will stop after the execution of the specified number of epochs**


## Training status monitoring
The training can be monitored with the `status` API.
Using the `process_id` returned by `train` API we can retrieve the current status execution:
```json
GET status?process_id=62a10034-1f2f-44b9-9014-71a1992647be&full=false
[
  {
    "result": "ok",
    "status": {
      "process_type": "training",
      "process_status": "running",
      "process_data": "Train - epoch [42/70] - batch [268/773] - loss=0.139 - metric=0.949"
    }
  }
]
```
