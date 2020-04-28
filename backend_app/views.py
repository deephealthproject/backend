import datetime
from pathlib import Path

import numpy as np
import os
import requests
import yaml
import uuid
from django.db.models import Q
from os.path import join as opjoin
from rest_framework import mixins, status, views, viewsets, generics
from rest_framework.response import Response

from backend import celery_app, settings
from backend_app import mixins as BAMixins, models, serializers
from backend_app import utils
from deeplearning.tasks import classification, segmentation
from deeplearning.utils import nn_settings


class AllowedPropViewSet(BAMixins.ParamListModelMixin,
                         viewsets.GenericViewSet):
    """
    ## GET
    This method returns the values that a property can assume depending on the model employed.
    It provides a default value and a comma separated list of values to choose from.

    When this api returns an empty list, the property allowed values and default should be retrieved
    using the `/properties/{id}` API.

    ## Parameters
    `model_id`: integer
        Required integer representing the model.

    `property_id`: integer
        Required integer representing a property.
    """
    queryset = models.AllowedProperty.objects.all()
    serializer_class = serializers.AllowedPropSerializer
    params = ['model_id', 'property_id']

    def get_queryset(self):
        model_id = self.request.query_params.get('model_id')
        property_id = self.request.query_params.get('property_id')
        self.queryset = models.AllowedProperty.objects.filter(model_id=model_id, property_id=property_id)
        return self.queryset


class DatasetViewSet(mixins.ListModelMixin,
                     mixins.CreateModelMixin,
                     viewsets.GenericViewSet):
    """
    ## GET
    This method returns the datasets list that should be loaded in the power user component,
    on the left side of the page, in the same panel as the models list.

    ### Parameters
    `task_id` _(optional)_: integer
        Integer representing a task used to retrieve dataset of a specific task.

    ## POST
    This API uploads a dataset YAML file and stores it in the backend.
    The url must contain the url of a dataset, e.g.
    [`dropbox.com/s/ul1yc8owj0hxpu6/isic_segmentation.yml`](https://www.dropbox.com/s/ul1yc8owj0hxpu6/isic_segmentation.yml?dl=1).
    """
    queryset = models.Dataset.objects.filter(is_single_image=False)
    serializer_class = serializers.DatasetSerializer

    def get_queryset(self):
        task_id = self.request.query_params.get('task_id')
        if task_id:
            self.queryset = models.Dataset.objects.filter(task_id=task_id)
        return self.queryset

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Download the yml file in url
        url = serializer.validated_data['path']
        dataset_name = serializer.validated_data['name']
        dataset_out_path = f'{settings.DATASETS_DIR}/{dataset_name}.yml'
        if Path(f'{settings.DATASETS_DIR}/{dataset_name}.yml').exists():
            return Response({'error': f'The dataset `{dataset_name}` already exists'},
                            status=status.HTTP_400_BAD_REQUEST)
        try:
            r = requests.get(url, allow_redirects=True)
            if r.status_code == 200:
                yaml_content = yaml.load(r.content, Loader=yaml.FullLoader)
                with open(f'{settings.DATASETS_DIR}/{dataset_name}.yml', 'w') as f:
                    yaml.dump(yaml_content, f, Dumper=utils.MyDumper, sort_keys=False)

                # Update the path
                serializer.save(path=dataset_out_path)
                headers = self.get_success_headers(serializer.data)
                return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
        except requests.exceptions.RequestException:
            # URL malformed
            return Response({'error': 'URL malformed'}, status=status.HTTP_400_BAD_REQUEST)
        return Response({'error': 'URL malformed'}, status=status.HTTP_400_BAD_REQUEST)


class InferenceViewSet(views.APIView):
    """
    ## POST
    This is the main entry point to start the inference.
    It is mandatory to specify a pre-trained model and a dataset for finetuning.
    """

    def post(self, request):
        serializer = serializers.InferenceSerializer(data=request.data)

        if serializer.is_valid():
            return utils.do_inference(serializer)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class InferenceSingleViewSet(views.APIView):
    """
    ## POST
    This API allows the inference of a single image.
    It is mandatory to specify the same fields of `/inference` API, but for dataset_id which is replace by
    the url of the image to process.
    """

    def post(self, request):
        serializer = serializers.InferenceSingleSerializer(data=request.data)

        if serializer.is_valid():
            image_url = serializer.validated_data['image_url']
            project_id = serializer.validated_data['project_id']
            task_id = models.Project.objects.get(id=project_id).task_id

            # Create a dataset with the single image to process
            dummy_dataset = f'name: {image_url}\n' \
                            f'description: {image_url} auto-generated dataset\n' \
                            f'images: [{image_url}]\n' \
                            f'split:\n' \
                            f'  test: [0]'
            # Save dataset and get id
            d = models.Dataset(name=f'single-image-dataset', task_id=task_id, path='', is_single_image=True)
            d.save()
            yaml_content = yaml.load(dummy_dataset, Loader=yaml.FullLoader)
            with open(f'{settings.DATASETS_DIR}/single_image_dataset_{d.id}.yml', 'w') as f:
                yaml.dump(yaml_content, f, Dumper=utils.MyDumper, sort_keys=False)

            # Update the path
            d.path = f'{settings.DATASETS_DIR}/single_image_dataset_{d.id}.yml'
            d.save()

            serializer.validated_data['dataset_id'] = d
            return utils.do_inference(serializer)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ModelViewSet(mixins.ListModelMixin,
                   viewsets.GenericViewSet):
    """
    ## GET
    This API allows the client to know which Neural Network models are available in the system in order to allow
    their selection.

    The optional `task_id` parameter is used to filter them based on the task the models are used for.

    ## Parameters
    `task_id` _(optional)_: integer
        Optional integer for filtering the models based on task.
    """
    queryset = models.Model.objects.all()
    serializer_class = serializers.ModelSerializer

    def get_queryset(self):
        task_id = self.request.query_params.get('task_id')
        if task_id:
            self.queryset = models.Model.objects.filter(task_id=task_id)
        return self.queryset


class ModelWeightsViewSet(BAMixins.ParamListModelMixin,
                          mixins.RetrieveModelMixin,
                          viewsets.GenericViewSet):
    """
    ## GET
    When 'use pre-trained' is selected, it is possible to query the backend passing a `model_id` to obtain a list
    of dataset on which it was pretrained.

    ## Parameters
    `model_id`: integer
        Required integer representing the model.

    ## PUT
    This method updates an existing model weight (e.g. change the name).
    """
    queryset = models.ModelWeights.objects.all()
    serializer_class = serializers.ModelWeightsSerializer
    params = ['model_id']

    def get_queryset(self):
        model_id = self.request.query_params.get('model_id')
        self.queryset = models.ModelWeights.objects.filter(model_id=model_id)
        return self.queryset

    def get_obj(self, id):
        try:
            return models.ModelWeights.objects.get(id=id)
        except models.ModelWeights.DoesNotExist:
            return None

    def put(self, request, *args, **kwargs):
        weight = self.get_obj(request.data['id'])
        if not weight:
            error = {"Error": f"Weight {request.data['id']} does not exist"}
            return Response(data=error, status=status.HTTP_400_BAD_REQUEST)
        serializer = self.serializer_class(weight, data=request.data)
        if serializer.is_valid():
            serializer.save()
            # Returns all the elements with model_id in request
            queryset = models.ModelWeights.objects.filter(model_id=weight.model_id)
            serializer = self.get_serializer(queryset, many=True)
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class OutputViewSet(views.APIView):
    """
    ## GET
    This API provides information about an `inference` process.In classification task it returns the list
    of images and an array composed of the classes prediction scores.
    In segmentation task it returns the URLs of the segmented images.
    """

    @staticmethod
    def trunc(values, decs=0):
        return np.trunc(values * 10 ** decs) / (10 ** decs)

    def get(self, request):
        if not self.request.query_params.get('process_id'):
            error = {'Error': f'Missing required parameter `process_id`'}
            return Response(data=error, status=status.HTTP_400_BAD_REQUEST)
        process_id = self.request.query_params.get('process_id')
        infer = models.Inference.objects.filter(celery_id=process_id)
        if not infer:
            # already deleted weight/training or inference
            return Response({"result": "Process stopped before finishing or non existing"},
                            status=status.HTTP_404_NOT_FOUND)

        infer = infer.first()
        # Differentiate classification and segmentation
        # if infer.modelweights_id.task_id.name.lower() == 'classification':
        if not os.path.exists(opjoin(settings.OUTPUTS_DIR, infer.outputfile)):
            return Response({"result": "Output file not found"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        outputs = open(opjoin(settings.OUTPUTS_DIR, infer.outputfile), 'r')
        if infer.modelweights_id.model_id.task_id.name.lower() == 'classification':
            lines = outputs.read().splitlines()
            lines = [line.split(';') for line in lines]
            # preds = self.trunc(preds, decs=8)
        else:
            # Segmentation
            # output file contains path of files
            uri = request.build_absolute_uri(settings.MEDIA_URL)
            lines = outputs.read().splitlines()
            lines = [l.replace(settings.OUTPUTS_DIR, uri) for l in lines]
        response = {'outputs': lines}
        return Response(response, status=status.HTTP_200_OK)


class ProjectViewSet(mixins.ListModelMixin,
                     mixins.RetrieveModelMixin,
                     mixins.CreateModelMixin,
                     viewsets.GenericViewSet):
    """
    ## GET
    Lists all the available projects or a single one using `projects/{id}` link.

    ## POST
    Lets to create a new project.

    ## PUT
    Updates a project.
    """
    queryset = models.Project.objects.all()
    serializer_class = serializers.ProjectSerializer

    def get_obj(self, id):
        try:
            return models.Project.objects.get(id=id)
        except models.Project.DoesNotExist:
            return None

    def put(self, request, *args, **kwargs):
        project = self.get_obj(request.data['id'])
        if not project:
            error = {"Error": f"Project {request.data['id']} does not exist"}
            return Response(data=error, status=status.HTTP_400_BAD_REQUEST)
        serializer = serializers.ProjectSerializer(project, data=request.data)
        if serializer.is_valid():
            serializer.save()
            # Returns all the elements
            return self.list(request)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class PropertyViewSet(mixins.ListModelMixin,
                      mixins.RetrieveModelMixin,
                      viewsets.GenericViewSet):
    """
    ## GET
    This API allows the client to know which properties are "globally" supported by the backend.


    A model can have different default and allowed values if the `/allowedProperties` return an entry.
    """
    queryset = models.Property.objects.all()
    serializer_class = serializers.PropertyListSerializer

    def get_queryset(self):
        name = self.request.query_params.get('name')
        # Substitute underscore with space if present
        if name:
            name = [name, name.replace('_', ' ')]
            self.queryset = models.Property.objects.filter(Q(name__icontains=name[0]) | Q(name__icontains=name[1]))
        return self.queryset


class StatusView(views.APIView):
    """
    ## GET
    This  API allows the frontend to query the status of a training or inference, identified by a `process_id`
    (which is returned by `/train` or `/inference` APIs).
    """

    def get(self, request):
        if not self.request.query_params.get('process_id'):
            error = {'Error': f'Missing required parameter `process_id`'}
            return Response(data=error, status=status.HTTP_400_BAD_REQUEST)
        process_id = self.request.query_params.get('process_id')

        if models.ModelWeights.objects.filter(celery_id=process_id).exists():
            process = models.ModelWeights.objects.filter(celery_id=process_id).first()
        elif models.Inference.objects.filter(celery_id=process_id).exists():
            process = models.Inference.objects.filter(celery_id=process_id).first()
        else:
            res = {
                "result": "error",
                "error": "Process not found."
            }
            return Response(data=res, status=status.HTTP_404_NOT_FOUND)

        try:
            with open(process.logfile, 'r') as f:
                lines = f.read().splitlines()
                last_line = lines[-1]
        except:
            res = {
                "result": "error",
                "error": "Log file not found"
            }
            return Response(data=res, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        if last_line == '<done>':
            process_status = 'finished'
            last_line = lines[-2]
        else:
            process_status = 'running'

        res = {
            'result': 'ok',
            'status': {
                'process_type': 'training',
                'process_status': process_status,
                'process_data': last_line,
            }
        }
        return Response(data=res, status=status.HTTP_200_OK)


class StopProcessViewSet(views.APIView):
    """
    ## POST
    Stop a training process specifying a `process_id` (which is returned by `/train` or `/inference` APIs).
    """

    def post(self, request):
        serializer = serializers.StopProcessSerializer(data=request.data)
        if serializer.is_valid():
            process_id = serializer.data['process_id']
            weights = models.ModelWeights.objects.filter(celery_id=process_id)
            infer = models.Inference.objects.filter(celery_id=process_id)
            response = {"result": "Process stopped"}
            if not weights.exists() and not infer.exists():
                # already deleted weight/training or inference
                return Response({"result": "Process already stopped or non existing"}, status=status.HTTP_404_NOT_FOUND)
            elif weights:
                weights = weights.first()
                celery_id = weights.celery_id
                celery_app.control.revoke(celery_id, terminate=True, signal='SIGUSR1')
                response = {"result": "Training stopped"}
                # delete the ModelWeights entry from db
                # also delete ModelWeights fk in project
                weights.delete()
            elif infer:
                infer = infer.first()
                celery_id = infer.celery_id
                celery_app.control.revoke(celery_id, terminate=True, signal='SIGUSR1')
                response = {"result": "Inference stopped"}
                # delete the ModelWeights entry from db
                infer.delete()

            # todo delete log file? delete weight file?
            return Response(response, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class TaskViewSet(mixins.ListModelMixin,
                  viewsets.GenericViewSet):
    """
    ## GET
    This API allows the client to know which task this platform supports. e.g. classification or segmentation tasks.
    """
    queryset = models.Task.objects.all()
    serializer_class = serializers.TaskSerializer


class TrainViewSet(views.APIView):
    """
    ## POST
    This is the main entry point to start the training of a model on a dataset.
    It is mandatory to specify a model to be trained and a pretraining dataset.
    When providing a weights_id, the training starts from the pre-trained model.
    """

    def post(self, request):
        serializer = serializers.TrainSerializer(data=request.data)

        if serializer.is_valid():
            # Create a new modelweights and start training
            weight = models.ModelWeights()
            weight.dataset_id_id = serializer.data['dataset_id']
            weight.model_id_id = serializer.data['model_id']

            if not models.Dataset.objects.filter(id=weight.dataset_id_id).exists():
                error = {"Error": f"Dataset with id `{weight.dataset_id_id}` does not exist"}
                return Response(error, status=status.HTTP_400_BAD_REQUEST)

            if not models.Model.objects.filter(id=weight.model_id_id).exists():
                error = {"Error": f"Model with id `{weight.model_id_id}` does not exist"}
                return Response(error, status=status.HTTP_400_BAD_REQUEST)

            if not models.Project.objects.filter(id=serializer.data['project_id']).exists():
                error = {"Error": f"Project with id `{serializer.data['project_id']}` does not exist"}
                return Response(error, status=status.HTTP_400_BAD_REQUEST)

            # Check if dataset and model are both for same task
            if weight.model_id.task_id != weight.dataset_id.task_id:
                error = {"Error": f"Model and dataset must belong to the same task"}
                return Response(error, status=status.HTTP_400_BAD_REQUEST)

            project = models.Project.objects.get(id=serializer.data['project_id'])
            task_name = project.task_id.name.lower()
            weight.task_id = project.task_id
            weight.name = f'{weight.model_id.name}_{weight.dataset_id.name}_' \
                          f'{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
            if serializer.data['weights_id']:
                weight.pretrained_on_id = serializer.data['weights_id']

                if not models.ModelWeights.objects.filter(id=weight.pretrained_on_id).exists():
                    error = {"Error": f"Model weight with id `{weight.pretrained_on_id}` does not exist"}
                    return Response(error, status=status.HTTP_400_BAD_REQUEST)

            weight.save()  # Generate an id for the weight
            ckpts_dir = opjoin(settings.TRAINING_DIR, 'ckpts')
            weight.location = Path(opjoin(ckpts_dir, f'{weight.id}.bin')).absolute()
            # Create a logfile
            weight.logfile = models.generate_file_path(f'{uuid.uuid4().hex}.log', settings.TRAINING_DIR, 'logs')
            weight.save()

            hyperparams = {}

            # Check if current model has some custom properties and load them
            props_allowed = models.AllowedProperty.objects.filter(model_id=weight.model_id_id)
            if props_allowed:
                for p in props_allowed:
                    hyperparams[p.property_id.name] = p.default_value

            # Load default values for those properties not in props_allowed
            props_general = models.Property.objects.all()
            for p in props_general:
                if hyperparams.get(p.name) is None:
                    hyperparams[p.name] = p.default

            # Overwrite hyperparams with ones provided by the user
            props = serializer.data['properties']
            for p in props:
                ts = models.TrainingSetting()
                # Get the property by name
                name = p['name']
                name = [name, name.replace('_', ' ')]
                queryset = models.Property.objects.filter(Q(name__icontains=name[0]) | Q(name__icontains=name[1]))
                if len(queryset) == 0:
                    # Property does not exist, delete the weight and its associated properties (cascade)
                    weight.delete()
                    error = {"Error": f"Property `{p['name']}` does not exist"}
                    return Response(error, status=status.HTTP_400_BAD_REQUEST)
                property = queryset[0]
                ts.property_id = property
                ts.modelweights_id = weight
                ts.value = str(p['value'])
                ts.save()
                hyperparams[property.name] = ts.value

            config = nn_settings(modelweight=weight, hyperparams=hyperparams)
            if not config:
                return Response({"Error": "Properties error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Differentiate the task and start training
            if task_name == 'classification':
                celery_id = classification.classificate.delay(config)
                # celery_id = classification.training(config)
            elif task_name == 'segmentation':
                celery_id = segmentation.segment.delay(config)
                # celery_id = segmentation.training(config)
            else:
                return Response({'error': 'error on task'}, status=status.HTTP_400_BAD_REQUEST)

            weight = models.ModelWeights.objects.get(id=weight.id)
            weight.celery_id = celery_id.id
            weight.save()

            # todo what if project already has a modelweight?
            # Training started, store the training in project
            project.modelweights_id = weight
            project.save()

            response = {
                "result": "ok",
                "process_id": celery_id.id,
                "weight_id": weight.id
            }
            return Response(response, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class TrainingSettingViewSet(BAMixins.ParamListModelMixin,
                             viewsets.GenericViewSet):
    """
    ## GET
    This API returns the value used for a property in a specific training
    (a modelweights). It requires a modelweights_id, indicating a training process, and a property_id.

    ## Parameters
    `modelweights_id`: integer
        Required integer representing the ModelWeights.

    `property_id`: integer
        Required integer representing a property.
    """
    queryset = models.TrainingSetting.objects.all()
    serializer_class = serializers.TrainingSettingSerializer
    params = ['modelweights_id', 'property_id']

    def get_queryset(self):
        modelweights_id = self.request.query_params.get('modelweights_id')
        property_id = self.request.query_params.get('property_id')
        self.queryset = models.TrainingSetting.objects.filter(modelweights_id=modelweights_id, property_id=property_id)
        return self.queryset
