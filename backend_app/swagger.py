from drf_yasg import openapi
from drf_yasg.views import get_schema_view
from rest_framework import permissions

from backend_app import serializers

desc = """
__This is a draft version of the DeepHealth Toolkit API.__

The main idea is to model the process that happens when training or predicting using existing NN models.

`/projects` let the user to retrieve all the projects or update/create a new one.

`/tasks` returns the tasks (e.g. classification or segmentation) this platform supports. Given a project, it sets the \
default value for the current project. 

`/properties`, `/models` and `/datasets` can be used to query the backend and obtain a list of the available options \
in the training process, a list of neural network models,and, a list of datasets for both training or finetuning.

`/weights` returns, specifying a neural network model, a list of pre-trained weights of that model.

`/train` and `/inference` allow to start a new training process, or a new inference process. 
The first one requires a project, a model and a dataset. When providing a weight, the training starts from the \
pre-trained model. It also accepts an optional dataset for finetuning the model.

`/inference` requires a weight and a dataset.

After starting a process, its status can be queried with `/status`, in order to obtain information on how the training \
is going, the loss, the accuracy, and so on. When the process is finished, the API allows to specify other information\
, such as a results file or what is implied by the process itself. 
"""
schema_view = get_schema_view(
    openapi.Info(
        title="DeepHealth Toolkit API",
        default_version='v1.0.2',
        description=desc,
        # terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="michele.cancilla@unimore.it"),
        license=openapi.License(name="Apache 2.0", url='http://www.apache.org/licenses/LICENSE-2.0.html'),
    ),
    url='https://jenkins-master-deephealth-unix01.ing.unimore.it/backend/',
    public=True,
    permission_classes=(permissions.AllowAny,),
)

# Responses

DatasetViewSet_create_response = {
    '200': openapi.Response('On a successful operation, it returns the new instance of dataset.',
                            serializers.DatasetSerializer,
                            examples={
                                "application/json": {
                                    "id": 7,
                                    "name": "dataset-test",
                                    "path": "/mnt/data/backend/data/datasets/dataset-test.yml",
                                    "task_id": 2,
                                    "owners": [
                                        {"username": "dhtest"}
                                    ]
                                }
                            }),
    '400': openapi.Response('Something is wrong in the request. Details in `error`.',
                            serializers.GeneralErrorResponse,
                            examples={
                                "application/json": {
                                    "result": "error",
                                    "error": "URL malformed"
                                }
                            })
}

inferences_post_responses = {
    '200': openapi.Response(
        'On a successful operation, it returns the `process_id`, used for polling the '
        'operation status.',
        serializers.InferenceResponseSerializer,
        examples={
            "application/json": {
                "result": "ok",
                "process_id": "62a10034-1f2f-44b9-9014-71a1992647be"
            }
        }),
    '400': openapi.Response('Something is wrong in the request. Details in `error`.',
                            serializers.GeneralErrorResponse,
                            examples={
                                "application/json": {
                                    "result": "error",
                                    "error": "Non existing weights_id"
                                }})
}
ModelStatusView_get_response = {
    '200': openapi.Response('Status of upload process', serializers.ModelStatusResponse, examples={
        "application/json": {
            "result": "SUCCESS"
        }
    })
}
OutputViewSet_get_responses = {
    '200': openapi.Response('response for classification', serializers.OutputsResponse, examples={
        "application/json": {
            "outputs": [[[
                "image.jpg",
                [5.2e-7, 3e-8, 0.00001397, 0.00003845, 9e-8, 7.2e-7, 0, 0.99987125, 1.4e-7, 0.0000748]
            ]]
            ]
        }}
                            )}

ProjectViewSet_create_response = {
    '200': openapi.Response('Successful operation',
                            serializers.ProjectSerializer(many=True),
                            examples={
                                "application/json":
                                    [
                                        {
                                            "id": 1,
                                            "name": "Classification mnist",
                                            "task_id": 1,
                                            "modelweights_id": None,
                                            "inference_id": 592,
                                            "users": [
                                                {"username": "dhtest"}
                                            ]
                                        },
                                        {
                                            "id": 2,
                                            "name": "Segmentation isic",
                                            "task_id": 2,
                                            "modelweights_id": None,
                                            "inference_id": 528,
                                            "users": [
                                                {"username": "dhtest"},
                                                {"username": "dhtest2"}
                                            ]
                                        },
                                        {
                                            "id": 3,
                                            "name": "Classification isic",
                                            "task_id": 1,
                                            "modelweights_id": None,
                                            "inference_id": None,
                                            "users": [
                                                {"username": "dhtest"}
                                            ]
                                        }
                                    ]
                            }
                            )
}

StatusView_get_response = {
    '200': openapi.Response('Status of process', serializers.StatusResponse())
}
StopProcessViewSet_post_response = {
    '200': openapi.Response('On a successful operation, it returns a message that confirmes the process interruption.',
                            serializers.GeneralResponse(),
                            examples={
                                "application/json": {
                                    "result": "Process stopped"
                                }
                            }
                            ),
    '400': openapi.Response('Something is wrong in the request. Details in `error`.',
                            serializers.GeneralErrorResponse(),
                            examples={
                                "application/json": {
                                    "result": "error",
                                    "error": "Process already stopped or non existing."
                                }
                            })
}

TrainViewSet_post_response = {
    '200': openapi.Response('On a successful operation, it returns the `process_id`, used for '
                            'polling the operation status.', serializers.TrainResponse(),
                            examples={
                                "application/json": {
                                    "result": "ok",
                                    "process_id": "62a10034-1f2f-44b9-9014-71a1992647be"
                                }
                            }),
    '400': openapi.Response('Something is wrong in the request. Details in `error`.',
                            serializers.GeneralErrorResponse(),
                            examples={
                                "application/json": {
                                    "result": "error",
                                    "error": "Non existing process_id"
                                }
                            })
}
