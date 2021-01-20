import json
import mimetypes
import uuid

import numpy as np
import yaml
from rest_framework import status
from rest_framework.response import Response

from backend import settings
from backend_app import models, serializers
from deeplearning.tasks import classification
from deeplearning.tasks import segmentation
from deeplearning.utils import inference_settings


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
    dataset = serializer.validated_data['dataset_id']
    uuid4 = uuid.uuid4().hex + '.log'
    modelweights_id = serializer.validated_data['modelweights_id']
    dataset_id = serializer.validated_data['dataset_id']

    weight = modelweights_id
    user = request.user

    # Check if current user can use an existing weight
    if not models.ModelWeightsPermission.objects.filter(modelweight=weight, user=user).exists() and \
            not weight.public:
        error = {"Error": f"The {user.username} user has no permission to access the chosen weight"}
        return Response(error, status=status.HTTP_401_UNAUTHORIZED)

    # Check if current user can use the dataset
    if not models.DatasetPermission.objects.filter(dataset=weight.dataset_id, user=user).exists() and \
            not weight.dataset_id.public:
        error = {"Error": f"The {user.username} user has no permission to access the chosen dataset"}
        return Response(error, status=status.HTTP_401_UNAUTHORIZED)

    i = models.Inference(
        modelweights_id=modelweights_id,
        dataset_id=dataset_id,
        stats='',  # todo change
        # logfile=models.default_logfile_path(settings.INFERENCE_DIR, 'logs'),
        # outputfile=models.default_logfile_path(settings.OUTPUTS_DIR),
        logfile=models.generate_file_path(uuid4, settings.INFERENCE_DIR, 'logs'),
        outputfile=models.generate_file_path(uuid4, settings.OUTPUTS_DIR),
    )
    i.save()
    p_id = serializer.validated_data['project_id']
    project = models.Project.objects.get(id=p_id)
    project.inference_id = i
    project.save()
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

    config = inference_settings(inference_id=i.id, hyperparams=hyperparams, dataset_id=dataset.id)
    if not config:
        return Response({"Error": "Properties error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Launch the inference
    # Differentiate the task and start training
    if task_name == 'classification':
        celery_id = classification.classificate.delay(config)
        # celery_id = classification.classificate(config)
    elif task_name == 'segmentation':
        celery_id = segmentation.segment.delay(config)
        # celery_id = segmentation.segment(config)
    else:
        return Response({'error': 'error on task'}, status=status.HTTP_400_BAD_REQUEST)

    i.celery_id = celery_id.id
    i.save()
    response = serializers.InferenceResponseSerializer({
        "result": "ok",
        "process_id": celery_id.id,
    })
    return Response(response.data, status=status.HTTP_201_CREATED)
