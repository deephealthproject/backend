import base64
import datetime
import os
import uuid
from os.path import join as opjoin
from pathlib import Path

import numpy as np
import requests
import yaml
from celery import shared_task
from celery.result import AsyncResult
from django.contrib.auth.models import User
from django.core.exceptions import ObjectDoesNotExist
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
    queryset = models.Dataset.objects.filter(is_single_image=False, public=True)
    serializer_class = serializers.DatasetSerializer

    def get_queryset(self):
        user = self.request.user
        task_id = self.request.query_params.get('task_id')
        q_perm = models.Dataset.objects.filter(datasetpermission__user=user.id)  # Get datasets of current user
        if task_id:
            self.queryset = self.queryset.filter(task_id=task_id)
            q_perm = q_perm.filter(task_id=task_id)
        self.queryset = self.queryset.union(q_perm)  # Extend the public datasets with ones of the user
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
        """Create a new dataset downloading it from URL or path

        This API creates a dataset YAML file and stores it in the backend.
        The `path` field must contain the URL of a dataset, e.g. \
        [`dropbox.com/s/ul1yc8owj0hxpu6/isic_segmentation.yml`](https://www.dropbox.com/s/ul1yc8owj0hxpu6/isic_segmentation.yml?dl=1) \
        or a local path pointing to a YAML file.
        """
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response({'error': 'Validation error. Request data is malformed.'},
                            status=status.HTTP_400_BAD_REQUEST)

        # Download the yml file in url
        url = serializer.validated_data['path']
        dataset_name = serializer.validated_data['name']
        if Path(f'{settings.DATASETS_DIR}/{dataset_name}.yml').exists():
            return Response({'error': f'The dataset `{dataset_name}` already exists'},
                            status=status.HTTP_400_BAD_REQUEST)  # TODO delete the file when delete the object
        try:
            dataset_out_path = f'{settings.DATASETS_DIR}/{dataset_name}.yml'
            r = requests.get(url, allow_redirects=True)
            if r.status_code == 200:
                yaml_content = yaml.load(r.content, Loader=yaml.FullLoader)
                with open(f'{settings.DATASETS_DIR}/{dataset_name}.yml', 'w') as f:
                    yaml.dump(yaml_content, f, Dumper=utils.MyDumper, sort_keys=False)

                # Update the path
                serializer.save(path=dataset_out_path)

                headers = self.get_success_headers(serializer.data)
                return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
        except (requests.exceptions.MissingSchema, requests.exceptions.InvalidSchema):
            # Local YAML file
            if os.path.isfile(url):
                serializer.save()
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
            return utils.do_inference(request, serializer)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class InferenceSingleViewSet(views.APIView):
    @swagger_auto_schema(request_body=serializers.InferenceSingleSerializer,
                         responses=swagger.inferences_post_responses)
    def post(self, request):
        """Starts the inference providing an url or base64 content image

        This API allows the inference of a single image.
        It is mandatory to specify a `project_id` and a `modelweights_id` to use, then at least one of `image_url` and \
        `image_data` must be passed.

        The image_url data parameter represents an image url to download and test, while image_data represents the \
        base64 content of an image to process.
        """
        serializer = serializers.InferenceSingleSerializer(data=request.data)

        if serializer.is_valid():
            project_id = serializer.validated_data['project_id']
            task_id = models.Project.objects.get(id=project_id).task_id
            dummy_dataset = None
            if serializer.validated_data.get('image_data'):
                image_data = serializer.validated_data['image_data']
                header, data_encoded = image_data.split(',', 1)
                header = header.split(':')[1].split(';')[0]
                ext = utils.guess_extension(header)
                # Create image from base64
                image_name = "UserImage" + str(uuid.uuid4()) + ext
                with open(opjoin(settings.DATASETS_DIR, image_name), "wb") as f:
                    f.write(base64.b64decode(data_encoded))

                # Create a dataset with the single image to process
                dummy_dataset = f'name: "User single image"\n' \
                                f'description: "User single image auto-generated dataset"\n' \
                                f'images: ["{image_name}"]\n' \
                                f'split:\n' \
                                f'  test: [0]'
            elif serializer.validated_data.get('image_url'):
                image_url = serializer.validated_data.get('image_url')
                # Create a dataset with the single image to process
                dummy_dataset = f'name: "{image_url}"\n' \
                                f'description: "{image_url} auto-generated dataset"\n' \
                                f'images: ["{image_url}"]\n' \
                                f'split:\n' \
                                f'  test: [0]'
            else:
                error = {'error': 'Missing data. One of `image_url` and `image_data` must be included in body request.'}
                return Response(error, status=status.HTTP_400_BAD_REQUEST)

            # Save dataset and get id
            d = models.Dataset(name=f'single-image-dataset', task_id=task_id, path='', is_single_image=True)
            d.save()
            try:
                yaml_content = yaml.load(dummy_dataset, Loader=yaml.FullLoader)
            except yaml.YAMLError as e:
                d.delete()
                print(e)
                return Response({'error': 'Error in YAML parsing'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            with open(f'{settings.DATASETS_DIR}/single_image_dataset_{d.id}.yml', 'w') as f:
                yaml.dump(yaml_content, f, Dumper=utils.MyDumper, sort_keys=False)

            # Update the path
            d.path = f'{settings.DATASETS_DIR}/single_image_dataset_{d.id}.yml'
            d.save()

            serializer.validated_data['dataset_id'] = d
            return utils.do_inference(request, serializer)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@shared_task
def onnx_download(url, model_out_path):
    r = requests.get(url, allow_redirects=True)
    if r.status_code == 200:
        with open(model_out_path, 'wb') as f:
            f.write(r.content)


class ModelStatusViewSet(views.APIView):
    @swagger_auto_schema(
        manual_parameters=[openapi.Parameter('process_id', openapi.IN_QUERY,
                                             "Pass a required UUID representing a model upload process.",
                                             type=openapi.TYPE_STRING, format=openapi.FORMAT_UUID, required=False)],
        responses=swagger.ModelStatusView_get_response
    )
    def get(self, request, *args, **kwargs):
        """Retrieve results about a model upload process

        This API provides information about a model upload process. It returns the status of the operation:
        - PENDING: The task is waiting for execution.
        - STARTED: The task has been started.
        - RETRY: The task is to be retried, possibly because of failure.
        - FAILURE: The task raised an exception, or has exceeded the retry limit.
        - SUCCESS: The task executed successfully.
        """
        if not self.request.query_params.get('process_id'):
            error = {'Error': f'Missing required parameter `process_id`'}
            return Response(data=error, status=status.HTTP_400_BAD_REQUEST)
        process_id = self.request.query_params.get('process_id')
        model = models.Model.objects.filter(celery_id=process_id)
        if not model:
            # already deleted model
            return Response({"result": "Process stopped before finishing or non existing."},
                            status=status.HTTP_404_NOT_FOUND)

        return Response({"result": AsyncResult(process_id).status}, status=status.HTTP_200_OK)


class ModelViewSet(mixins.ListModelMixin,
                   mixins.CreateModelMixin,
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

    def create(self, request, *args, **kwargs):
        """TODO docs

        """
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            name = serializer.validated_data.get('name')
            # task = serializer.validated_data.get('task_id')
            model_out_path = f'{settings.MODELS_DIR}/{name}.onnx'  # TODO this overwrite the file if exists
            celery_id = None
            response = None
            if serializer.validated_data.get('onnx_url'):
                # download onnx file from url
                try:
                    url = serializer.validated_data.pop('onnx_url')
                    celery_id = onnx_download.delay(url, model_out_path)
                    celery_id = celery_id.id
                    response = {"result": "ok", "process_id": celery_id}
                except requests.exceptions.RequestException:
                    # URL malformed
                    return Response({'error': 'URL is malformed'}, status=status.HTTP_400_BAD_REQUEST)
            elif serializer.validated_data.get('onnx_data'):
                onnx_data = serializer.validated_data.pop('onnx_data')
                # onnx file was uploaded
                onnx_data = onnx_data.read()
                with open(model_out_path, 'wb') as f:
                    f.write(onnx_data)
                response = {"result": "ok"}
            else:
                return Response({'error': 'How did you get here?'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Check for dataset_id parameter
            # If given the current model has been already trained on that dataset
            if serializer.validated_data.get('dataset_id'):
                # Current onnx contains weight on the "dataset" dataset
                dataset = serializer.validated_data.pop('dataset_id')
                # Update the path and celery_id and save
                model = serializer.save(location=model_out_path, celery_id=celery_id)
                weight = models.ModelWeights.objects.create(
                    location=model.location,
                    name=model.name + '_ONNX',
                    model_id=model,
                    dataset_id=dataset
                )
                models.ModelWeightsPermission.objects.create(modelweight=weight, user=self.request.user)
            else:
                serializer.save(location=model_out_path, celery_id=celery_id)

            headers = self.get_success_headers(serializer.data)
            return Response(response, status=status.HTTP_201_CREATED, headers=headers)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ModelWeightsViewSet(BAMixins.ParamListModelMixin,
                          mixins.RetrieveModelMixin,
                          mixins.UpdateModelMixin,
                          viewsets.GenericViewSet):
    queryset = models.ModelWeights.objects.filter(public=True)
    serializer_class = serializers.ModelWeightsSerializer
    params = ['model_id']

    def get_queryset(self):
        if self.action == 'list':
            user = self.request.user

            model_id = self.request.query_params.get('model_id')
            self.queryset = self.queryset.filter(model_id=model_id)
            q_perm = models.ModelWeights.objects.filter(
                modelweightspermission__user=user, model_id=model_id, public=False)  # Get weights of current user
            self.queryset = self.queryset.union(q_perm)
            return self.queryset
        else:
            return super(ModelWeightsViewSet, self).get_queryset()

    @swagger_auto_schema(
        manual_parameters=[openapi.Parameter('model_id', openapi.IN_QUERY,
                                             "Return the modelweights obtained on `model_id` model.",
                                             type=openapi.TYPE_INTEGER, required=False)]
    )
    def list(self, request, *args, **kwargs):
        """Returns the available Neural Network models

        When 'use pre-trained' is selected, it is possible to query the backend passing a `model_id` to obtain a list
        of dataset on which it was pretrained.
        """
        return super().list(request, *args, **kwargs)

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

    # Always filter projects of the user
    def get_queryset(self):
        if self.request.user.is_anonymous:
            self.queryset = None
        else:
            self.queryset = models.Project.objects.filter(users=self.request.user)
        return self.queryset

    def get_obj(self, id):
        try:
            return models.Project.objects.get(id=id)
        except models.Project.DoesNotExist:
            return None

    def list(self, request, *args, **kwargs):
        """Loads all users projects

        This method lists all the available projects for the current user.
        """
        return super().list(request, *args, **kwargs)

    def retrieve(self, request, *args, **kwargs):
        """Retrieve a single project

        Returns a project by `{id}`.
        """
        return super().retrieve(request, *args, **kwargs)

    # Add users to a project
    def manage_users(self, project, users):
        for u in users:
            user = User.objects.get(username__exact=u.get('username'))
            project.users.add(user)
        return project

    # Check if users (list of user) exist
    def check_users(self, users):
        if not len(users):
            return Response({"Error": f"Users list cannot be empty"}, status=status.HTTP_400_BAD_REQUEST)
        for u in users:
            try:
                User.objects.get(username__exact=u.get('username'))
            except ObjectDoesNotExist:
                return Response({"Error": f"User `{u.get('username')}` does not exist"},
                                status=status.HTTP_400_BAD_REQUEST)
        return False

    @swagger_auto_schema(responses=swagger.ProjectViewSet_create_response)
    def create(self, request, *args, **kwargs):
        """Create a new project

        Create a new project given name and an associated task.
        """
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            p = serializer.save()
            headers = self.get_success_headers(serializer.data)
            users = request.data.get('users')
            response = self.check_users(users)
            if response:
                return response

            p = self.manage_users(p, users)  # Get users from initial_data, that's bad
            return Response(self.get_serializer(p).data, status=status.HTTP_201_CREATED, headers=headers)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def put(self, request, *args, **kwargs):
        """Update an existing project

        Update an existing project
        """
        project = self.get_obj(request.data['id'])
        project_id = project.id
        if not project:
            error = {"Error": f"Project {request.data['id']} does not exist"}
            return Response(data=error, status=status.HTTP_400_BAD_REQUEST)
        users = request.data.get('users')
        response = self.check_users(users)
        if response:
            return response

        serializer = serializers.ProjectSerializer(project, data=request.data)
        if serializer.is_valid():
            p = serializer.save()
            p.users.clear()
            p = self.manage_users(p, users)  # Get users from initial_data, it's bad
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

        if models.ModelWeights.objects.filter(training__celery_id=process_id).exists():
            process_type = 'training'
            process = models.ModelWeights.objects.filter(training__celery_id=process_id).first()
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
            training = models.Training.objects.filter(celery_id=process_id)
            infer = models.Inference.objects.filter(celery_id=process_id)
            response = {"result": "Process stopped"}
            if not training.exists() and not infer.exists():
                # already deleted weight/training or inference
                return Response({"result": "Process already stopped or non existing"}, status=status.HTTP_404_NOT_FOUND)
            elif training:
                training = training.first()
                celery_id = training.celery_id
                celery_app.control.revoke(celery_id, terminate=True, signal='SIGUSR1')
                response = {"result": "Training stopped"}
                # delete the ModelWeights entry from db
                # also delete ModelWeights fk in project
                training.delete()
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
                  mixins.RetrieveModelMixin,
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
        user = request.user
        if serializer.is_valid():
            # Create a new modelweights and start training
            weight = models.ModelWeights()
            weight.dataset_id_id = serializer.validated_data['dataset_id']
            weight.model_id_id = serializer.validated_data['model_id']

            if serializer.validated_data['weights_id']:
                weight.pretrained_on_id = serializer.validated_data['weights_id']

                # Does pretraining really exist?
                if not models.ModelWeights.objects.filter(id=weight.pretrained_on_id).exists() and not weight.public:
                    error = {"Error": f"Model weight with id `{weight.pretrained_on_id}` does not exist"}
                    return Response(error, status=status.HTTP_400_BAD_REQUEST)

                # Check if current user can use an existing weight as pretraining
                if not models.ModelWeightsPermission.objects.filter(modelweight_id=weight.pretrained_on_id,
                                                                    user=user).exists():
                    error = {
                        "Error": f"The {user.username} user has no permission to access the chosen pretraining weight"}
                    return Response(error, status=status.HTTP_401_UNAUTHORIZED)

            if not models.Dataset.objects.filter(id=weight.dataset_id_id, is_single_image=False).exists():
                error = {"Error": f"Dataset with id `{weight.dataset_id_id}` does not exist"}
                return Response(error, status=status.HTTP_400_BAD_REQUEST)

            if not models.Model.objects.filter(id=weight.model_id_id).exists():
                error = {"Error": f"Model with id `{weight.model_id_id}` does not exist"}
                return Response(error, status=status.HTTP_400_BAD_REQUEST)

            if not models.Project.objects.filter(id=serializer.validated_data['project_id']).exists():
                error = {"Error": f"Project with id `{serializer.validated_data['project_id']}` does not exist"}
                return Response(error, status=status.HTTP_400_BAD_REQUEST)

            # Check if dataset and model are both for same task
            if weight.model_id.task_id != weight.dataset_id.task_id:
                error = {"Error": f"Model and dataset must belong to the same task"}
                return Response(error, status=status.HTTP_400_BAD_REQUEST)

            # Check if current user can use the dataset
            if not models.DatasetPermission.objects.filter(dataset_id=weight.dataset_id_id, user=user).exists() and \
                    not weight.dataset_id.public:
                error = {"Error": f"The {user.username} user has no permission to access the chosen dataset"}
                return Response(error, status=status.HTTP_401_UNAUTHORIZED)

            project = models.Project.objects.get(id=serializer.validated_data['project_id'])
            task_name = project.task_id.name.lower()
            # weight.task_id = project.task_id

            weight.name = f'{weight.model_id.name}_{weight.dataset_id.name}_' \
                          f'{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'

            weight.save()  # Generate an id for the weight
            ckpts_dir = opjoin(settings.TRAINING_DIR, 'ckpts')
            weight.location = Path(opjoin(ckpts_dir, f'{weight.id}.onnx')).absolute()
            weight.save()

            # Must assign permissions to new weights
            # Grant permission to current user
            models.ModelWeightsPermission.objects.create(modelweight=weight, user=user,
                                                         permission=models.PERM[0][0])

            # Create a logfile
            training = models.Training(modelweights_id=weight, project_id=project)
            training.logfile = models.generate_file_path(f'{uuid.uuid4().hex}.log', settings.TRAINING_DIR, 'logs')
            training.save()

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
                ts.training_id = training
                ts.value = str(p['value'])
                ts.save()
                hyperparams[property.name] = ts.value

            config = nn_settings(training=training, hyperparams=hyperparams)
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

            training = models.Training.objects.get(id=training.id)
            training.celery_id = celery_id.id
            training.save()

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
    params = ['training_id', 'property_id']

    def get_queryset(self):
        training_id = self.request.query_params.get('training_id')
        property_id = self.request.query_params.get('property_id')
        self.queryset = models.TrainingSetting.objects.filter(training_id=training_id, property_id=property_id)
        return self.queryset

    @swagger_auto_schema(
        manual_parameters=[openapi.Parameter('training_id', openapi.IN_QUERY, "Integer representing a Training",
                                             required=True, type=openapi.TYPE_INTEGER),
                           openapi.Parameter('property_id', openapi.IN_QUERY, "Integer representing a Property",
                                             required=True, type=openapi.TYPE_INTEGER)]
    )
    def list(self, request, *args, **kwargs):
        """Returns settings used for a training process

        This API returns the value used for a property in a specific Training.
        It requires a `training_id`, indicating a training process, and a `property_id`.
        """
        return super().list(request, *args, **kwargs)
