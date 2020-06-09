import datetime
import os
import uuid
from os.path import join as opjoin
from pathlib import Path

import numpy as np
import requests
import yaml
from celery.result import AsyncResult
from django.db.models import Q
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from rest_framework import mixins, status, views, viewsets
from rest_framework.response import Response

from backend import celery_app, settings
from backend_app import mixins as BAMixins, models, serializers, swagger
from backend_app import utils
from deeplearning.tasks import classification, segmentation
from deeplearning.utils import nn_settings


class AllowedPropViewSet(BAMixins.ParamListModelMixin,
                         mixins.CreateModelMixin,
                         viewsets.GenericViewSet):
    queryset = models.AllowedProperty.objects.all()
    serializer_class = serializers.AllowedPropertySerializer
    params = ['model_id', 'property_id']

    def get_queryset(self):
        model_id = self.request.query_params.get('model_id')
        property_id = self.request.query_params.get('property_id')
        self.queryset = models.AllowedProperty.objects.filter(model_id=model_id, property_id=property_id)
        return self.queryset

    @swagger_auto_schema(
        manual_parameters=[openapi.Parameter('model_id', openapi.IN_QUERY, "Integer representing a model",
                                             required=True, type=openapi.TYPE_INTEGER),
                           openapi.Parameter('property_id', openapi.IN_QUERY, "Integer representing a property",
                                             required=True, type=openapi.TYPE_INTEGER)]
    )
    def list(self, request, *args, **kwargs):
        """Return the allowed and default values of a property

        This method returns the values that a property can assume depending on the model employed. \
        It provides a default value and a comma separated list of values to choose from.
        When this api returns an empty list, the property allowed values and default should be retrieved \
        using the `/properties/{id}` API.
        """
        return super().list(request, *args, **kwargs)

    def create(self, request, *args, **kwargs):
        """Create a new AllowedProperty

         This method create a new AllowedProperty
        """
        return super().create(request, *args, **kwargs)


class DatasetViewSet(mixins.ListModelMixin,
                     mixins.RetrieveModelMixin,
                     mixins.CreateModelMixin,
                     viewsets.GenericViewSet):
    queryset = models.Dataset.objects.filter(is_single_image=False)
    serializer_class = serializers.DatasetSerializer

    def get_queryset(self):
        task_id = self.request.query_params.get('task_id')
        if task_id:
            self.queryset = models.Dataset.objects.filter(task_id=task_id, is_single_image=False)
            # self.queryset = models.Dataset.objects.filter(task_id=task_id)
        return self.queryset

    @swagger_auto_schema(
        manual_parameters=[openapi.Parameter('task_id', openapi.IN_QUERY, type=openapi.TYPE_INTEGER, required=False)]
    )
    def list(self, request, *args, **kwargs):
        """Get the list datasets to use for training or finetuning

        This method returns all the datasets in the backend.
        """
        return super().list(request, *args, **kwargs)

    def retrieve(self, request, *args, **kwargs):
        """Retrieve a single dataset

        This method returns the `{id}` dataset.
        """
        return super().retrieve(request, *args, **kwargs)

    @swagger_auto_schema(responses=swagger.DatasetViewSet_create_response)
    def create(self, request, *args, **kwargs):
        """Upload a new dataset downloading it from a URL

        This API uploads a dataset YAML file and stores it in the backend.
        The `path` field must contain the URL of a dataset, e.g. \
        [`dropbox.com/s/ul1yc8owj0hxpu6/isic_segmentation.yml`](https://www.dropbox.com/s/ul1yc8owj0hxpu6/isic_segmentation.yml?dl=1).
        """
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response({'error': 'Validation error. Request data is malformed.'},
                            status=status.HTTP_400_BAD_REQUEST)

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
    @swagger_auto_schema(request_body=serializers.InferenceSerializer,
                         responses=swagger.inferences_post_responses)
    def post(self, request):
        """Start an inference process using a pre-trained model on a dataset

        This is the main entry point to start the inference. \
        It is mandatory to specify a pre-trained model and a dataset.
        """
        serializer = serializers.InferenceSerializer(data=request.data)

        if serializer.is_valid():
            return utils.do_inference(serializer)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class InferenceSingleViewSet(views.APIView):
    @swagger_auto_schema(request_body=serializers.InferenceSingleSerializer,
                         responses=swagger.inferences_post_responses)
    def post(self, request):
        """Starts the inference providing an image URL

        This API allows the inference of a single image.
        It is mandatory to specify the same fields of `/inference` API, but for dataset_id which is replaced by \
        the url of the image to process.
        """
        serializer = serializers.InferenceSingleSerializer(data=request.data)

        if serializer.is_valid():
            image_url = serializer.validated_data['image_url']
            project_id = serializer.validated_data['project_id']
            task_id = models.Project.objects.get(id=project_id).task_id

            # Create a dataset with the single image to process
            dummy_dataset = f'name: "{image_url}"\n' \
                            f'description: "{image_url} auto-generated dataset"\n' \
                            f'images: ["{image_url}"]\n' \
                            f'split:\n' \
                            f'  test: [0]'
            # Save dataset and get id
            d = models.Dataset(name=f'single-image-dataset', task_id=task_id, path='', is_single_image=True)
            d.save()
            try:
                yaml_content = yaml.load(dummy_dataset, Loader=yaml.FullLoader)
            except yaml.YAMLError as e:
                d.delete()
                print(e)
                return Response({'error': 'Error in YAML parsing'}, status=status.HTTP_400_BAD_REQUEST)

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
    queryset = models.Model.objects.all()
    serializer_class = serializers.ModelSerializer

    def get_queryset(self):
        task_id = self.request.query_params.get('task_id')
        if task_id:
            self.queryset = models.Model.objects.filter(task_id=task_id)
        return self.queryset

    @swagger_auto_schema(
        manual_parameters=[openapi.Parameter('task_id', openapi.IN_QUERY,
                                             "Integer for filtering the models based on task.",
                                             type=openapi.TYPE_INTEGER, required=False)]
    )
    def list(self, request):
        """Returns the available Neural Network models

        This API allows the client to know which Neural Network models are available in the system in order to allow \
        their selection.

        The optional `task_id` parameter is used to filter them based on the task the models are used for.
        """
        return super().list(request)


class ModelWeightsViewSet(BAMixins.ParamListModelMixin,
                          mixins.RetrieveModelMixin,
                          mixins.UpdateModelMixin,
                          viewsets.GenericViewSet):
    queryset = models.ModelWeights.objects.all()
    serializer_class = serializers.ModelWeightsSerializer
    params = ['model_id']

    def get_queryset(self):
        if self.action == 'list':
            model_id = self.request.query_params.get('model_id')
            self.queryset = models.ModelWeights.objects.filter(model_id=model_id)
            return self.queryset
        else:
            return super(ModelWeightsViewSet, self).get_queryset()

    @swagger_auto_schema(
        manual_parameters=[openapi.Parameter('model_id', openapi.IN_QUERY,
                                             "Return the modelweights obtained on `model_id` model.",
                                             type=openapi.TYPE_INTEGER, required=False)]
    )
    def list(self, request):
        """Returns the available Neural Network models

        When 'use pre-trained' is selected, it is possible to query the backend passing a `model_id` to obtain a list
        of dataset on which it was pretrained.
        """
        return super().list(request)

    def retrieve(self, request, *args, **kwargs):
        """Retrieve a single modelweight

        This API returns the modelweight with the requested`{id}`.
        """
        return super().retrieve(request, *args, **kwargs)

    def get_obj(self, id):
        try:
            return models.ModelWeights.objects.get(id=id)
        except models.ModelWeights.DoesNotExist:
            return None

    def put(self, request, *args, **kwargs):
        """Update an existing weight

        This method updates an existing model weight (e.g. change the name).
        """
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
            # serializer = self.serializer_class(queryset, many=True)
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, *args, **kwargs):
        """Update an existing weight

        This method updates an existing model weight (e.g. change the name).
        """
        return super().update(request, *args, **kwargs)

    @swagger_auto_schema(auto_schema=None)
    def partial_update(self, request, *args, **kwargs):
        return super().partial_update(request, *args, **kwargs)


class OutputViewSet(views.APIView):
    @staticmethod
    def trunc(values, decs=0):
        return np.trunc(values * 10 ** decs) / (10 ** decs)

    @swagger_auto_schema(
        manual_parameters=[openapi.Parameter('process_id', openapi.IN_QUERY,
                                             "Pass a required UUID representing a finished process.",
                                             type=openapi.TYPE_STRING, format=openapi.FORMAT_UUID, required=False)],
        responses=swagger.OutputViewSet_get_responses
    )
    def get(self, request, *args, **kwargs):
        """Retrieve results about an inference process

        This API provides information about an `inference` process.In classification task it returns the list \
        of images and an array composed of the classes prediction scores.
        In segmentation task it returns the URLs of the segmented images.
        """
        if not self.request.query_params.get('process_id'):
            error = {'Error': f'Missing required parameter `process_id`'}
            return Response(data=error, status=status.HTTP_400_BAD_REQUEST)
        process_id = self.request.query_params.get('process_id')
        infer = models.Inference.objects.filter(celery_id=process_id)
        if not infer:
            # already deleted weight/training or inference
            return Response({"result": "Process stopped before finishing or non existing."},
                            status=status.HTTP_404_NOT_FOUND)

        if AsyncResult(process_id).status == 'PENDING':
            return Response({"result": "Process in execution. Try later for output results."},
                            status=status.HTTP_200_OK)

        infer = infer.first()
        if not os.path.exists(opjoin(settings.OUTPUTS_DIR, infer.outputfile)):
            return Response({"result": "Output file not found"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        outputs = open(opjoin(settings.OUTPUTS_DIR, infer.outputfile), 'r')
        # Differentiate classification and segmentation
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
                     mixins.UpdateModelMixin,
                     viewsets.GenericViewSet):
    queryset = models.Project.objects.all()
    serializer_class = serializers.ProjectSerializer

    def get_obj(self, id):
        try:
            return models.Project.objects.get(id=id)
        except models.Project.DoesNotExist:
            return None

    def list(self, request, *args, **kwargs):
        """Loads all the projects

        This method lists all the available projects.
        """
        return super().list(request, *args, **kwargs)

    def retrieve(self, request, *args, **kwargs):
        """Retrieve a single project

        Returns a project by `{id}`.
        """
        return super().retrieve(request, *args, **kwargs)

    @swagger_auto_schema(responses=swagger.ProjectViewSet_create_response)
    def create(self, request, *args, **kwargs):
        """Create a new project

        Create a new project.
        """
        return super().create(request, *args, **kwargs)

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

    def update(self, request, *args, **kwargs):
        """Update an existing project

        Update a project instance by providing its `{id}`.
        """
        return super().update(request, *args, **kwargs)

    @swagger_auto_schema(auto_schema=None)
    def partial_update(self, request, *args, **kwargs):
        return super().partial_update(request, *args, **kwargs)


class PropertyViewSet(mixins.ListModelMixin,
                      mixins.RetrieveModelMixin,
                      viewsets.GenericViewSet):
    queryset = models.Property.objects.all()
    serializer_class = serializers.PropertyListSerializer

    def get_queryset(self):
        name = self.request.query_params.get('name')
        # Substitute underscore with space if present
        if name:
            name = [name, name.replace('_', ' ')]
            self.queryset = models.Property.objects.filter(Q(name__icontains=name[0]) | Q(name__icontains=name[1]))
        return self.queryset

    def list(self, request, *args, **kwargs):
        """Return the Properties supported by backend

        This API allows the client to know which properties are "globally" supported by the backend.
        A model can have different default value and allowed values if the `/allowedProperties` return an entry.
        """
        return super().list(request, *args, **kwargs)

    def retrieve(self, request, *args, **kwargs):
        """Retrieve a single property

        Return a property by `{id}`.
        """
        return super().retrieve(request, *args, **kwargs)


class StatusView(views.APIView):
    @swagger_auto_schema(manual_parameters=[openapi.Parameter('process_id', openapi.IN_QUERY,
                                                              "UUID representing a process",
                                                              required=True, type=openapi.TYPE_STRING,
                                                              format=openapi.FORMAT_UUID)],
                         responses=swagger.StatusView_get_response
                         )
    def get(self, request):
        """Return the status of an training or inference process

        This  API allows the frontend to query the status of a training or inference, identified by a `process_id` \
        (which is returned by `/train` or `/inference` APIs).
        """
        if not self.request.query_params.get('process_id'):
            error = {'Error': f'Missing required parameter `process_id`'}
            return Response(data=error, status=status.HTTP_400_BAD_REQUEST)
        process_id = self.request.query_params.get('process_id')

        if models.ModelWeights.objects.filter(celery_id=process_id).exists():
            process_type = 'training'
            process = models.ModelWeights.objects.filter(celery_id=process_id).first()
        elif models.Inference.objects.filter(celery_id=process_id).exists():
            process_type = 'inference'
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
                'process_type': process_type,
                'process_status': process_status,
                'process_data': last_line,
            }
        }
        return Response(data=res, status=status.HTTP_200_OK)


class StopProcessViewSet(views.APIView):
    @swagger_auto_schema(request_body=serializers.StopProcessSerializer,
                         responses=swagger.StopProcessViewSet_post_response
                         )
    def post(self, request):
        """Kill a training or inference process

        Stop a training process specifying a `process_id` (which is returned by `/train` or `/inference` APIs).
        """
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
    queryset = models.Task.objects.all()
    serializer_class = serializers.TaskSerializer

    def list(self, request, *args, **kwargs):
        """Return the tasks supported by backend

        This API allows the client to know which task this platform supports. e.g. classification or segmentation tasks.
        """
        return super().list(request, *args, **kwargs)


class TrainViewSet(views.APIView):
    @swagger_auto_schema(request_body=serializers.TrainSerializer,
                         responses=swagger.TrainViewSet_post_response
                         )
    def post(self, request):
        """Starts the training of a (possibly pre-trained) model on a dataset

        This is the main entry point to start the training of a model on a dataset. \
        It is mandatory to specify a model to be trained and a dataset.
        When providing a `weights_id`, the training starts from the pre-trained model.
        """
        serializer = serializers.TrainSerializer(data=request.data)

        if serializer.is_valid():
            # Create a new modelweights and start training
            weight = models.ModelWeights()
            weight.dataset_id_id = serializer.data['dataset_id']
            weight.model_id_id = serializer.data['model_id']

            if not models.Dataset.objects.filter(id=weight.dataset_id_id, is_single_image=False).exists():
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
                # celery_id = classification.classificate(config)
            elif task_name == 'segmentation':
                celery_id = segmentation.segment.delay(config)
                # celery_id = segmentation.segment(config)
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
    queryset = models.TrainingSetting.objects.all()
    serializer_class = serializers.TrainingSettingSerializer
    params = ['modelweights_id', 'property_id']

    def get_queryset(self):
        modelweights_id = self.request.query_params.get('modelweights_id')
        property_id = self.request.query_params.get('property_id')
        self.queryset = models.TrainingSetting.objects.filter(modelweights_id=modelweights_id, property_id=property_id)
        return self.queryset

    @swagger_auto_schema(
        manual_parameters=[openapi.Parameter('modelweights_id', openapi.IN_QUERY, "Integer representing a ModelWeights",
                                             required=True, type=openapi.TYPE_INTEGER),
                           openapi.Parameter('property_id', openapi.IN_QUERY, "Integer representing a Property",
                                             required=True, type=openapi.TYPE_INTEGER)]
    )
    def list(self, request, *args, **kwargs):
        """Returns settings used for a training

        This API returns the value used for a property in a specific training (a modelweights).
        It requires a `modelweights_id`, indicating a training process, and a `property_id`.
        """
        return super().list(request, *args, **kwargs)
