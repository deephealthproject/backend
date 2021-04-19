import json
import mimetypes
import uuid

import numpy as np
import yaml
from rest_framework import status
from rest_framework.response import Response

from backend import settings
from backend_app import models, views
from deeplearning.tasks import classification, segmentation
from deeplearning.utils import createConfig
from streamflow_app import models as sf_models


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class MyDumper(yaml.Dumper):
    # Increase the base indentation
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def guess_extension(ftype):
    if ftype == 'application/octet-stream':
        return '.bin'
    elif ftype == 'image/webp':
        return '.webp'
    ext = mimetypes.guess_extension(ftype)
    if ext in ['.jpe', '.jpeg']:
        ext = '.jpg'
    return ext


# class Custom64ImageField(Base64ImageField):
#     class Meta:
#         swagger_schema_fields = {
#             'type': 'string',
#             'title': 'File Content',
#             'description': 'Content of the file base64 encoded',
#             'read_only': False  # <-- FIX
#         }


def do_inference(request, serializer):
    uuid4 = uuid.uuid4().hex + '.log'
    # Dataset could be a normal dataset or a fake single image one
    dataset = serializer.validated_data['dataset_id']
    project = serializer.validated_data['project_id']
    weight = serializer.validated_data['modelweights_id']

    user = request.user

    # Check if dataset and weight is for same task
    if dataset.task_id != weight.dataset_id.task_id:
        error = {"Error": f"The dataset and weight chosen are not for the same task."}
        return Response(error, status=status.HTTP_400_BAD_REQUEST)

    # Check if current user can use an existing weight
    if not models.ModelWeightsPermission.objects.filter(modelweight=weight, user=user).exists() and \
            not weight.public:
        error = {"Error": f"The {user.username} user has no permission to access the chosen weight"}
        return Response(error, status=status.HTTP_401_UNAUTHORIZED)

    # Check if current user can use the dataset
    if not models.DatasetPermission.objects.filter(dataset=dataset, user=user).exists() and \
            not dataset.public:
        error = {"Error": f"The {user.username} user has no permission to access the chosen dataset"}
        return Response(error, status=status.HTTP_401_UNAUTHORIZED)

    i = models.Inference(
        modelweights_id=weight,
        dataset_id=dataset,
        project_id=project,
        stats='',  # todo change
        # logfile=models.default_logfile_path(settings.INFERENCE_DIR, 'logs'),
        # outputfile=models.default_logfile_path(settings.OUTPUTS_DIR),
        logfile=models.generate_file_path(uuid4, settings.INFERENCE_DIR, 'logs'),
        outputfile=models.generate_file_path(uuid4, settings.OUTPUTS_DIR),
    )
    i.save()
    task_name = project.task_id.name.lower()

    hyperparams = {}
    # Check if current model has some custom properties and load them
    props_allowed = models.AllowedProperty.objects.filter(model_id=i.modelweights_id.model_id_id).select_related(
        'property_id')
    if props_allowed:
        for p in props_allowed:
            hyperparams[p.property_id.name] = p.default_value

    # Load default values for those properties not in props_allowed
    props_general = models.Property.objects.all()
    for p in props_general:
        if hyperparams.get(p.name) is None:
            hyperparams[p.name] = p.default

    # Retrieve configuration of the specified modelweights
    qs = models.TrainingSetting.objects.filter(training_id__modelweights_id=i.modelweights_id).select_related(
        'property_id')

    # Create the dict of training settings
    for setting in qs:
        hyperparams[setting.property_id.name] = setting.value

    config = createConfig(i, hyperparams, 'inference')
    if not config:
        return Response({"Error": "Properties error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Launch the inference
    return launch_training_inference(
        serializer.validated_data['task_manager'],
        task_name,
        i,
        config,
        serializer.validated_data.get('env')
    )


def launch_training_inference(task_manager, task_name, task_instance, config, sf_env=None, new_training_weight=None):
    """
    @param task_manager: string Celery or StreamFlow
    @param task_name: String which indicates kind of task to execute (e.g. classification)
    @param task_instance: django model instance (Training or Inference)
    @param config: Dict with configuration parameters
    @param sf_env: Dict with the StreamFlow environment id and type fields
    @param new_training_weight: Integer with weight_id just created (only in training)
    @return:
    """
    # Launch the training or inference
    task_error = Response({'error': f'Task with name `{task_name} does not exist.`'},
                          status=status.HTTP_400_BAD_REQUEST)
    if task_manager == 'CELERY':
        # Differentiate the task and start training
        if task_name == 'classification':
            celery_id = classification.classificate.apply_async(args=[config],
                                                                link=views.enable_weight.s(new_training_weight))
            # celery_id = classification.classificate(config)
        elif task_name == 'segmentation':
            celery_id = segmentation.segment.apply_async(args=[config], link=views.enable_weight.s(new_training_weight))
            # celery_id = segmentation.segment(config)
        else:
            return task_error

        task_instance.celery_id = celery_id.id
        task_instance.save()
        response = {
            "result": "ok",
            "process_id": celery_id.id,
        }
    else:
        # sf_env['type'] could be SSH
        # sf_env['id'] could be 12

        # Get the specific environment chosen (SSH or Helm)
        sf_model = sf_models.choice_to_model(sf_env['type'])

        # Retrieve the environment by id (SSH env with id 12)
        environment = sf_model.objects.get(id=sf_env['id'])

        # TODO Pass environment information to StreamFlow

        # TODO Run task using StreamFlow
        if task_name == 'classification':
            pass
            # celery_id = classification.classificate(config)
        elif task_name == 'segmentation':
            pass
            # celery_id = segmentation.segment(config)
        else:
            return task_error
        response = {"result": "ok"}
    if new_training_weight:
        response.update({"weight_id": new_training_weight})
    return Response(response, status=status.HTTP_201_CREATED)
