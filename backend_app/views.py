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
from django.core.exceptions import ObjectDoesNotExist
from django.db.models import Q
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from rest_framework import exceptions, mixins, status, views, viewsets
from rest_framework.response import Response

from backend import celery_app, settings
from backend_app import mixins as BAMixins, models, serializers, swagger, utils
from deeplearning.utils import FINAL_LAYER, createConfig
from urllib.parse import urlparse

def check_permission(instance, user, operation):
    excp = exceptions.PermissionDenied({'Error': f"'{user}' has no permission to {operation} {str(instance)}"})
    try:
        instance_perm = instance.permission.get(user=user)
    except ObjectDoesNotExist:
        raise excp
    if instance_perm.permission != models.Perm.OWNER:
        raise excp


class AllowedPropViewSet(BAMixins.ParamListModelMixin,
                         mixins.CreateModelMixin,
                         viewsets.GenericViewSet):
    queryset = models.AllowedProperty.objects.all()
    serializer_class = serializers.AllowedPropertySerializer
    params = ['model_id', 'property_id']

    def get_queryset(self):
        model_id = self.request.query_params.get('model_id')
        property_id = self.request.query_params.get('property_id')
        dataset_id = self.request.query_params.get('dataset_id')
        self.queryset = models.AllowedProperty.objects.filter(model_id=model_id, property_id=property_id,
                                                              dataset_id=dataset_id)
        return self.queryset

    @swagger_auto_schema(
        manual_parameters=[openapi.Parameter('model_id', openapi.IN_QUERY, "Integer representing a model",
                                             required=True, type=openapi.TYPE_INTEGER),
                           openapi.Parameter('property_id', openapi.IN_QUERY, "Integer representing a property",
                                             required=True, type=openapi.TYPE_INTEGER),
                           openapi.Parameter('dataset_id', openapi.IN_QUERY, "Integer representing a dataset",
                                             required=False, type=openapi.TYPE_INTEGER)]
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
        """Create new allowed and default values of a property

        This method create a new AllowedProperty for a Property and Model and Dataset.
        """
        return super().create(request, *args, **kwargs)


class DatasetViewSet(mixins.ListModelMixin,
                     mixins.RetrieveModelMixin,
                     mixins.CreateModelMixin,
                     mixins.DestroyModelMixin,
                     viewsets.GenericViewSet):
    queryset = models.Dataset.objects.filter(is_single_image=False, public=True)
    serializer_class = serializers.DatasetSerializer

    def get_queryset(self):
        user = self.request.user
        task_id = self.request.query_params.get('task_id')
        # Get datasets of current user
        q_perm = models.Dataset.objects.filter(permission__user=user.id, public=False)
        if task_id:
            self.queryset = self.queryset.filter(task_id=task_id)
            q_perm = q_perm.filter(task_id=task_id)
        self.queryset |= q_perm  # Extend the public datasets with ones of the user
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
        """Create a new dataset downloading it from URL or local path

        This API creates a dataset YAML file and stores it in the backend.
        The `path` field must contain the URL of a dataset, e.g. \
        [`dropbox.com/s/ul1yc8owj0hxpu6/isic_segmentation.yml`](https://www.dropbox.com/s/ul1yc8owj0hxpu6/isic_segmentation.yml?dl=1) \
        or a local path pointing to a YAML file.
        The `ctype` and `ctype_gt` indicates the kind of color type of the image and ground truth (if exists).
        """
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response({**{'error': 'Validation error. Request data is malformed.'}, **serializer.errors},
                            status=status.HTTP_400_BAD_REQUEST)

        # Download the yml file in url
        url = serializer.validated_data['path']
        dataset_name = serializer.validated_data['name']
        if Path(f'{settings.DATASETS_DIR}/{dataset_name}.yml').exists():
            # If file already exists append a suffix
            dataset_name = f'{dataset_name}_{uuid.uuid4().hex}'
        d = None
        yaml_content = None
        try:
            dataset_out_path = f'{settings.DATASETS_DIR}/{dataset_name}.yml'
            r = requests.get(url, allow_redirects=True)
            if r.status_code == 200:
                yaml_content = yaml.load(r.content, Loader=yaml.FullLoader)
                with open(dataset_out_path, 'w') as f:
                    yaml.dump(yaml_content, f, Dumper=utils.MyDumper, sort_keys=False)

                # Update the path
                d = serializer.save(path=dataset_out_path)
        except (requests.exceptions.MissingSchema, requests.exceptions.InvalidSchema):
            # Local YAML file
            if os.path.isfile(url):
                d = serializer.save()
        except requests.exceptions.RequestException:
            # URL malformed
            return Response({'error': 'URL malformed'}, status=status.HTTP_400_BAD_REQUEST)

        if d is None:
            return Response({'error': 'URL malformed'}, status=status.HTTP_400_BAD_REQUEST)

        if not yaml_content:
            # Read yaml and look for classes
            yaml_content = yaml.load(d.path, Loader=yaml.FullLoader)

        # Save the classes if found
        if yaml_content.get('classes'):
            d.classes = ','.join(yaml_content.get('classes'))
            d.save()

        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def destroy(self, request, *args, **kwargs):
        """Delete a dataset

        Delete a dataest by providing its `{id}`.
        """
        check_permission(instance=self.get_object(), user=request.user, operation='delete')
        return super().destroy(request, *args, **kwargs)


def inference_get(request):
    if not request.query_params.get('project_id'):
        error = {'Error': f'Missing required parameter `project_id`'}
        return Response(data=error, status=status.HTTP_400_BAD_REQUEST)
    project_id = request.query_params.get('project_id')
    if not models.ProjectPermission.objects.filter(user=request.user, project_id=project_id).exists():
        raise exceptions.PermissionDenied(
            {'Error': f"'{request.user}' has no permission to view Project {project_id}"})
    else:
        queryset = models.Inference.objects.filter(project_id=project_id)
    serializer = serializers.InferenceSerializer(queryset, many=True)
    return Response(serializer.data, status=status.HTTP_200_OK)


class InferenceViewSet(views.APIView):
    @swagger_auto_schema(request_body=serializers.InferenceSerializer,
                         responses=swagger.inferences_post_responses)
    def post(self, request):
        """Start an inference process using a pre-trained model on a dataset

        This is the main entry point to start the inference. \
        It is mandatory to specify a pre-trained model and a dataset.

        The `task_manager` field (default: _CELERY_) chooses which "task manager" employ between Celery and \
        StreamFlow. The optional `env` parameter is mandatory when _STREAMFLOW_ is used as `task_manager`. \
        It contains `id` referring to an existing SF Environment (SSH or Helm for example) and a `type` choice \
        field which indicates the kind of environment to employ.
        """
        serializer = serializers.InferenceSerializer(data=request.data, context={'request': request})

        if serializer.is_valid():
            return utils.do_inference(request, serializer)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(manual_parameters=[
        openapi.Parameter('project_id', openapi.IN_QUERY, "Id representing a project", required=True,
                          type=openapi.TYPE_INTEGER)],
        responses=swagger.Inference_get_response)
    def get(self, request):
        """Return all the inferences made within a project

        This API returns all the inferences performed by users within a project.
        The `project_id` parameter is mandatory.
        """
        return inference_get(request)


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


        The `task_manager` field (default: _CELERY_) chooses which "task manager" employ between Celery and \
        StreamFlow. The optional `env` parameter is mandatory when _STREAMFLOW_ is used as `task_manager`. \
        It contains `id` referring to an existing SF Environment (SSH or Helm for example) and a `type` choice \
        field which indicates the kind of environment to employ.
        """
        serializer = serializers.InferenceSingleSerializer(data=request.data, context={'request': request})

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

    @swagger_auto_schema(manual_parameters=[openapi.Parameter('project_id', openapi.IN_QUERY,
                                                              "Id representing a project",
                                                              required=True, type=openapi.TYPE_INTEGER)],
                         responses=swagger.Inference_get_response
                         )
    def get(self, request):
        """Return all the inferenceSingle made within a project

        This API returns all the inferences performed by users within a project.
        The `project_id` parameter is mandatory.
        """
        return inference_get(request)


class ModelWeightsStatusViewSet(views.APIView):
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
        weight = models.ModelWeights.objects.filter(process_id=process_id)
        if not weight:
            # Already deleted Weight
            return Response({'result': 'Process stopped before finishing or non existing.'},
                            status=status.HTTP_404_NOT_FOUND)

        response = serializers.ModelStatusResponse({
            'process_type': 'Model Weight downloading',
            'result': AsyncResult(process_id).status
        })
        return Response(response.data, status=status.HTTP_200_OK)


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
        """Create a new model

        Create a new model providing its name and related task_id.
        """
        return super().create(request)


class ModelWeightsViewSet(BAMixins.ParamListModelMixin,
                          mixins.RetrieveModelMixin,
                          mixins.UpdateModelMixin,
                          mixins.CreateModelMixin,
                          mixins.DestroyModelMixin,
                          viewsets.GenericViewSet):
    queryset = models.ModelWeights.objects.all()  # filter(public=True)
    serializer_class = serializers.ModelWeightsSerializer
    params = ['model_id']

    def get_serializer_class(self):
        if self.request.method == 'POST':
            return serializers.ModelWeightsCreateSerializer
        return serializers.ModelWeightsSerializer

    def get_queryset(self):
        if self.action in ['list']:
            # Return public weights and the ones with own/view permission
            user = self.request.user
            model_id = self.request.query_params.get('model_id')
            self.queryset = self.queryset.filter(model_id=model_id, public=True, is_active=True)
            # Get weights of current user
            q_perm = models.ModelWeights.objects.filter(permission__user=user, model_id=model_id, public=False,
                                                        is_active=True)
            self.queryset = (self.queryset | q_perm).distinct()

            if self.request.query_params.get('dataset_id'):
                dataset_id = self.request.query_params.get('dataset_id')

                # Remove weights which have 'layer_to_remove' set to NULL and which have not been trained on
                # `dataset_id` because they can be finetuned only on the same dataset.
                self.queryset = self.queryset.filter(Q(layer_to_remove__isnull=False) | Q(dataset_id=dataset_id))
            return self.queryset
        else:
            return super().get_queryset()

    @swagger_auto_schema(
        manual_parameters=[
            openapi.Parameter('model_id', openapi.IN_QUERY, "Return the weights obtained on `model_id` model.",
                              type=openapi.TYPE_INTEGER,
                              required=not models.ModelWeights._meta.get_field('model_id').null),
            openapi.Parameter('dataset_id', openapi.IN_QUERY, "Return the weights trained on the specified `dataset_id`"
                                                              " or suitable for such a dataset.",
                              type=openapi.TYPE_INTEGER,
                              required=False)
        ], responses=swagger.ModelWeightsViewSet_list_request)
    def list(self, request, *args, **kwargs):
        """Returns the available Neural Network model weights

        The `model_id` parameter filters weights for such a model, while the optional `dataset_id` one \
        excludes datasets which are not suitable for training (e.g. dataset with layer_to_remove field empty)
        """
        return super().list(request, *args, **kwargs)

    @swagger_auto_schema(responses=swagger.ModelWeightsViewSet_list_request)
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

    @swagger_auto_schema(request_body=swagger.ModelWeightsViewSet_create_request,
                         responses=swagger.ModelWeightsViewSet_create_response)
    def create(self, request, *args, **kwargs):
        """Create a new model weight

        This API creates a new model weight which is defined through a ONNX file.
        The ONNX can be uploaded using the `onnx_data` body field or can be retrieved by the backend providing the \
        `onnx_url` field with values like `https://my_onnx_model.onnx` or `file:///home/my_onnx_model.onnx`.

        The `dataset_id` optional parameter indicates that the model has been already trained on a certain dataset. \
        The backend will then create a new ModelWeight instance for these Model and Dataset.

        If the weight to upload has been trained on a dataset that does not exist in the backend, then the `classes` \
        field can be provided too. This field works the same as Dataset `classes` field, and contains the list of \
        classes of the dataset employed for training the weight.
        """
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            name = serializer.validated_data.get('name')
            suffix = ''
            if Path(f'{settings.MODELS_DIR}/{name}.onnx').exists() or \
                    models.ModelWeights.objects.filter(name=name):
                suffix = '_' + uuid.uuid4().hex

            # If name or file path already exists append a suffix
            model_out_path = f'{settings.MODELS_DIR}/{name}{suffix}.onnx'
            name = name + suffix

            process_id = None
            response = {"result": "ok"}
            if serializer.validated_data.get('onnx_url'):
                # download onnx file from url
                url = serializer.validated_data.pop('onnx_url')
                if urlparse(url).scheme == 'file':
                    model_out_path = urlparse(url).netloc
                else:
                    try:
                        process_id = onnx_download.delay(url, model_out_path)
                        process_id = process_id.id
                        response["process_id"] = process_id
                    except requests.exceptions.RequestException:
                        # URL malformed
                        return Response({'error': 'URL is malformed'}, status=status.HTTP_400_BAD_REQUEST)
            elif serializer.validated_data.get('onnx_data'):
                onnx_data = serializer.validated_data.pop('onnx_data')
                # onnx file was uploaded
                onnx_data = onnx_data.read()
                with open(model_out_path, 'wb') as f:
                    f.write(onnx_data)
            else:
                return Response({'error': 'How did you get here?'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            weight = serializer.save(name=name, location=model_out_path, process_id=process_id, is_active=True)
            models.ModelWeightsPermission.objects.create(modelweight=weight, user=self.request.user)

            response['weight_id'] = weight.id
            headers = self.get_success_headers(serializer.data)
            return Response(response, status=status.HTTP_201_CREATED, headers=headers)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(request_body=swagger.ModelWeightsViewSet_update_request)
    def update(self, request, *args, **kwargs):
        """Update an existing weight

        This method updates an existing model weight (e.g. change the name and permissions).
        """
        return super().update(request, *args, **kwargs)

    @swagger_auto_schema(auto_schema=None)  # Remove swagger docs
    def partial_update(self, request, *args, **kwargs):
        return super().partial_update(request, *args, **kwargs)

    def destroy(self, request, *args, **kwargs):
        """Delete a weight

        Delete a weight and its ONNX file.
        """
        check_permission(instance=self.get_object(), user=request.user, operation='delete')
        return super().destroy(request, *args, **kwargs)


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
                     mixins.DestroyModelMixin,
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

    @swagger_auto_schema(responses=swagger.ProjectViewSet_get_response)
    def list(self, request, *args, **kwargs):
        """Loads all users projects

        This method lists all the available projects for the current user.
        """
        ret = super().list(request, *args, **kwargs)
        return ret

    @swagger_auto_schema(responses=swagger.ProjectViewSet_create_retrieve_update_response)
    def retrieve(self, request, *args, **kwargs):
        """Retrieve a single project

        Returns a project by `{id}`.
        """
        return super().retrieve(request, *args, **kwargs)

    @swagger_auto_schema(responses=swagger.ProjectViewSet_create_retrieve_update_response,
                         request_body=swagger.ProjectViewSet_create_update_request)
    def create(self, request, *args, **kwargs):
        """Create a new project

        Create a new project given name, an associated task and users who will own it.
        users list must contain the username of a user and his permission.
        """
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            headers = self.get_success_headers(serializer.data)
            return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(responses=swagger.ProjectViewSet_create_retrieve_update_response,
                         request_body=swagger.ProjectViewSet_create_update_request)
    def update(self, request, *args, **kwargs):
        """Update an existing project

        Update a project instance by providing its `{id}`.
        The new list of users replace the old one, so removing a user from the user list will remove that user's
        permissions.
        """
        return super().update(request, *args, **kwargs)

    @swagger_auto_schema(auto_schema=None)
    def partial_update(self, request, *args, **kwargs):
        return super().partial_update(request, *args, **kwargs)

    def destroy(self, request, *args, **kwargs):
        """Delete a project

        Delete a project by providing its `{id}`.
        """
        check_permission(instance=self.get_object(), user=request.user, operation='delete')
        return super().destroy(request, *args, **kwargs)


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
                                                              format=openapi.FORMAT_UUID),
                                            openapi.Parameter('full', openapi.IN_QUERY,
                                                              "If true return the full history of a training/inference",
                                                              required=False, default=False, type=openapi.TYPE_BOOLEAN)
                                            ],
                         responses=swagger.StatusView_get_response
                         )
    def get(self, request):
        """Return the status of an training or inference process.

        This  API allows the frontend to query the status of a training or inference, identified by a `process_id` \
        (which is returned by `/train` or `/inference` APIs).

        When the optional parameter `full=true` is provided, the status api returns the full log of the process \
        execution.

        The _process_status_ field can provide different codes:
         - PENDING: The task is waiting for execution.
         - STARTED: The task has been started.
         - RETRY: The task is to be retried, possibly because of failure.
         - FAILURE: The task raised an exception, or has exceeded the retry limit.
         - SUCCESS: The task executed successfully.
        """
        full_return_string = False

        serializer = serializers.StatusSerializer(data=request.query_params)

        serializer.is_valid(raise_exception=True)
        process_id = serializer.validated_data['process_id']
        full = serializer.validated_data['full']

        if models.Training.objects.filter(celery_id=process_id).exists():
            process_type = 'training'
            process = models.Training.objects.filter(celery_id=process_id).first()
        elif models.Inference.objects.filter(celery_id=process_id).exists():
            process_type = 'inference'
            process = models.Inference.objects.filter(celery_id=process_id).first()
        else:
            res = {
                "result": "error",
                "error": "Process not found"
            }
            return Response(data=res, status=status.HTTP_404_NOT_FOUND)
        try:
            with open(process.logfile, 'r') as f:
                lines = f.read()
        except:
            res = {
                "result": "error",
                "error": "Log file not found"
            }
            return Response(data=res, status=status.HTTP_404_NOT_FOUND)

        lines_split = lines.splitlines()
        last_line = -1
        if lines_split[last_line] == '<done>':
            last_line = -2
        process = AsyncResult(str(process_id))
        if process.status == 'FAILURE':
            res = {
                'result': 'error',
                'status': {
                    'process_type': process_type,
                    'process_status': process.status,
                    'process_data': str(process.result),
                }
            }
            return Response(data=res, status=status.HTTP_200_OK)

        if full:
            try:
                index = lines_split.index("Reading dataset")
                process_data = ','.join(lines_split[index + 1:last_line]) if full_return_string else lines_split[
                                                                                                     index + 1:last_line]
            except ValueError:
                # index does not find the string
                process_data = ','.join(lines_split[:last_line]) if full_return_string else lines_split[:last_line]
        else:
            process_data = lines_split[last_line]

        res = {
            'result': 'ok',
            'status': {
                'process_type': process_type,
                'process_status': process.status,
                'process_data': process_data,
            }
        }
        return Response(data=res, status=status.HTTP_200_OK)


class StopProcessViewSet(views.APIView):
    @swagger_auto_schema(request_body=serializers.StopProcessSerializer,
                         responses=swagger.StopProcessViewSet_post_response
                         )
    def post(self, request):
        """Kill a training or inference process

        Stop a training process specifying a `process_id` (which is returned by `/train` or `/inference` APIs) and
        delete all its data.
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
                # delete the Training entry from db
                # also delete Training fk in project
                training.delete()
            elif infer:
                infer = infer.first()
                celery_id = infer.celery_id
                celery_app.control.revoke(celery_id, terminate=True, signal='SIGUSR1')
                response = {"result": "Inference stopped"}
                # delete the Inference entry from db
                infer.delete()

            # todo delete log file? delete weight file?
            return Response(response, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class TaskViewSet(mixins.ListModelMixin,
                  mixins.RetrieveModelMixin,
                  viewsets.GenericViewSet):
    queryset = models.Task.objects.all()
    serializer_class = serializers.TaskSerializer

    def retrieve(self, request, *args, **kwargs):
        """Provide information about a task

        Provide information about a task.
        """
        return super().retrieve(request, *args, **kwargs)

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
        It is mandatory to specify a weight and a dataset.
        The `weights_id` field specifies which weight should be used for the training/finetuning.

        The `task_manager` field (default: _CELERY_) chooses which "task manager" employ between Celery and \
        StreamFlow. The optional `env` parameter is mandatory when _STREAMFLOW_ is used as `task_manager`. \
        It contains `id` referring to an existing SF Environment (SSH or Helm for example) and a `type` choice \
        field which indicates the kind of environment to employ.
        """
        serializer = serializers.TrainSerializer(data=request.data, context={'request': request})
        user = request.user
        if serializer.is_valid():
            # Create a new modelweights and start training
            weight = models.ModelWeights()
            weight.dataset_id = serializer.validated_data['dataset_id']
            weight.classes = weight.dataset_id.classes  # Inherit from dataset
            weight.pretrained_on = serializer.validated_data['weights_id']
            # Inherit model and layer_to_remove from weight used as pretraining
            weight.model_id = weight.pretrained_on.model_id
            if weight.pretrained_on.dataset_id == weight.dataset_id:
                # Same datasets between pretraining and current means that
                # last layer is already trained for such a dataset
                # weight.layer_to_remove inherited from parent and not changed in onnx
                # NB layer_to_remove could also be null
                weight.layer_to_remove = weight.pretrained_on.layer_to_remove
            elif weight.pretrained_on.dataset_id is not None and \
                    weight.pretrained_on.dataset_id != weight.dataset_id and \
                    weight.pretrained_on.dataset_id.classes == weight.dataset_id.classes:
                # Different dataset object but same classes
                # Maybe a dataset replica?
                # --> Same layer_to_remove of the parent
                weight.layer_to_remove = weight.pretrained_on.layer_to_remove
            elif weight.pretrained_on.dataset_id is None and \
                    weight.classes == weight.dataset_id.classes \
                    and weight.classes is not None and weight.dataset_id.classes is not None:
                # Current weight has classes which are the same of the current dataset
                # it means that you should not remove last layer (maybe)
                # --> Same layer_to_remove of the parent and thus last layer will not be removed
                weight.layer_to_remove = weight.pretrained_on.layer_to_remove
            elif weight.pretrained_on.layer_to_remove is not None:
                # Different dataset -> last layer will be replaced
                # weight.layer_to_remove replaced with a new one named as FINAL_LAYER variable
                weight.layer_to_remove = FINAL_LAYER
            else:
                error = {"Error": f"The Model weight with id `{weight.pretrained_on_id}` cannot be used with dataset"
                                  f" {weight.dataset_id}."}
                return Response(error, status=status.HTTP_400_BAD_REQUEST)

            # Does the pretraining really exist?
            if not models.ModelWeights.objects.filter(id=weight.pretrained_on_id).exists() and not weight.public:
                error = {"Error": f"Model weight with id `{weight.pretrained_on_id}` does not exist."}
                return Response(error, status=status.HTTP_400_BAD_REQUEST)

            # Check if current user can use an existing weight as pretraining
            if not models.ModelWeightsPermission.objects.filter(modelweight_id=weight.pretrained_on_id,
                                                                user=user).exists() \
                    and not weight.pretrained_on.public:
                error = {
                    "Error": f"The {user.username} user has no permission to access the chosen pretraining weight"}
                return Response(error, status=status.HTTP_401_UNAUTHORIZED)

            if not models.Dataset.objects.filter(id=weight.dataset_id_id, is_single_image=False).exists():
                error = {"Error": f"Dataset with id `{weight.dataset_id_id}` does not exist"}
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

            project = serializer.validated_data['project_id']
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
                                                         permission=models.Perm.OWNER)

            # Create a logfile
            training = models.Training(modelweights_id=weight, project_id=project)
            training.logfile = models.generate_file_path(f'{uuid.uuid4().hex}.log', settings.TRAINING_DIR, 'logs')
            training.save()

            hyperparams = {}

            # Check if current model has some custom properties and load them
            props_allowed = models.AllowedProperty.objects.filter(model_id=weight.model_id_id,
                                                                  dataset_id=None).select_related(
                'property_id')
            if props_allowed:
                for p in props_allowed:
                    hyperparams[p.property_id.name] = p.default_value
            props_allowed = models.AllowedProperty.objects.filter(model_id=weight.model_id_id,
                                                                  dataset_id=weight.dataset_id).select_related(
                'property_id')
            # Override with allowedproperties specific for the dataset
            if props_allowed:
                for p in props_allowed:
                    hyperparams[p.property_id.name] = p.default_value

            # Load default values for those properties not in props_allowed
            props_general = models.Property.objects.all()
            for p in props_general:
                if hyperparams.get(p.name) is None:
                    hyperparams[p.name] = p.default or ''

            # Overwrite hyperparams with ones provided by the user
            props = serializer.data['properties']
            for p in props:
                name = p['name']
                name = [name, name.replace('_', ' ')]
                prop_tmp = models.Property.objects.filter(
                    Q(name__icontains=name[0]) | Q(name__icontains=name[1])).first()
                hyperparams[prop_tmp.name] = str(p['value'])

            # Create a TrainingSetting for each hyperparameter
            for k, v in hyperparams.items():
                if not models.Property.objects.filter(name=k).exists():
                    raise exceptions.ParseError(f'Property with name `{k}` does not exist')
                    # error = {"Error": f"Property `{k}` does not exist"}
                    # return Response(error, status=status.HTTP_400_BAD_REQUEST)
                field = models.Property.objects.filter(name=k).first()
                models.TrainingSetting.objects.create(property_id=field, training_id=training, value=v)

            config = createConfig(training, hyperparams, 'training')
            if not config:
                return Response({"Error": "Properties error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            return utils.launch_training_inference(
                serializer.validated_data['task_manager'],
                task_name,
                training,
                config,
                serializer.validated_data.get('env'),
                training.modelweights_id_id
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class TrainingsViewSet(BAMixins.ParamListModelMixin,
                       viewsets.GenericViewSet):
    queryset = models.Training.objects.all()
    serializer_class = serializers.TrainingSerializer
    params = ['project_id']

    def get_queryset(self):
        user = self.request.user
        project_id = self.request.query_params.get('project_id')
        if not models.ProjectPermission.objects.filter(user=user, project=project_id).exists():
            raise exceptions.PermissionDenied({'Error': f"'{user}' has no permission to view Project {project_id}"})
        self.queryset = self.queryset.filter(project_id=project_id)
        # Retrieve optional modelweight_id
        modelweights_id = self.request.query_params.get('modelweights_id')
        if modelweights_id:
            if not models.ModelWeightsPermission.objects.filter(user=user, modelweight=modelweights_id).exists():
                raise exceptions.PermissionDenied(
                    {'Error': f"'{user}' has no permission to view Weight {modelweights_id}"})
            else:
                self.queryset = self.queryset.filter(modelweights_id=modelweights_id)
        return self.queryset

    @swagger_auto_schema(manual_parameters=[
        openapi.Parameter('project_id', openapi.IN_QUERY, "Integer representing a Project",
                          required=True, type=openapi.TYPE_INTEGER),
        openapi.Parameter('modelweights_id', openapi.IN_QUERY, "Integer representing a Weight",
                          required=False, type=openapi.TYPE_INTEGER),
    ])
    def list(self, request, *args, **kwargs):
        """Returns past training processes

        This API returns past trainings performed within the `project_id` project.
        The optional parameter `modelweights_id` filters trainings of for a fixed Project and ModelWeight.
        """
        return super().list(request, *args, **kwargs)


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


@shared_task
def onnx_download(url, model_out_path):
    r = requests.get(url, allow_redirects=True)
    if r.status_code == 200:
        with open(model_out_path, 'wb') as f:
            f.write(r.content)


@shared_task
def enable_weight(task_return_value: bool, weight_id: int) -> None:
    """
    Enable the weight already trained
    @param weight_id: id of the weight to enable
    @return: None
    """
    if weight_id:
        w = models.ModelWeights.objects.get(id=weight_id)
        w.is_active = True
        w.save()


@shared_task
def error_handler(request, exc, traceback):
    print('Task {0} raised exception: {1!r}\n{2!r}'.format(request.id, exc, traceback))
