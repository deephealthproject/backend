from django.db.models import Q
from rest_framework import mixins, status, views, viewsets
from rest_framework.response import Response

from backend_app import mixins as BAMixins, models, serializers
# from deeplearning.tasks.classification import classification_training, classification_inference
from deeplearning.tasks import classification
from deeplearning.tasks import segmentation
from deeplearning.utils import nn_settings
import datetime
import os
import json
import numpy as np
from pathlib import Path
from backend import celery_app, settings
from backend_app import utils


class AllowedPropViewSet(BAMixins.ParamListModelMixin,
                         viewsets.GenericViewSet):
    """
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
                     viewsets.GenericViewSet):
    """
    This method returns the datasets list that should be loaded in the power user component,
    on the left side of the page, in the same panel as the models list.

    ## Parameters
    `task_id` _(optional)_: integer
        Integer representing a task used to retrieve dataset of a specific task.
    """
    queryset = models.Dataset.objects.all()
    serializer_class = serializers.DatasetSerializer

    def get_queryset(self):
        a = datetime.datetime.now()
        from django.utils import timezone

        now = timezone.now()
        task_id = self.request.query_params.get('task_id')
        if task_id:
            self.queryset = models.Dataset.objects.filter(task_id=task_id)
        return self.queryset


class InferenceViewSet(views.APIView):
    """
    This is the main entry point to start the inference.
    It is mandatory to specify a pre-trained model and a dataset for finetuning.
    """

    def post(self, request):
        serializer = serializers.InferenceSerializer(data=request.data)

        if serializer.is_valid():
            i = models.Inference(
                modelweights_id=serializer.validated_data['modelweights_id'],
                dataset_id=serializer.validated_data['dataset_id'],
                stats=''  # todo change
            )
            i.save()
            p_id = serializer.data['project_id']
            project = models.Project.objects.get(id=p_id)
            project.inference_id = i
            project.save()
            task_name = project.task_id.name.lower()

            hyperparams = {}
            # Check if current model has some custom properties and load them
            props_allowed = models.AllowedProperty.objects.filter(model_id=i.modelweights_id.model_id_id)
            if props_allowed:
                for p in props_allowed:
                    hyperparams[p.property_id.name] = p.default_value

            # Load default values for those properties not in props_allowed
            props_general = models.Property.objects.all()
            for p in props_general:
                if hyperparams.get(p.name) is None:
                    hyperparams[p.name] = p.default

            # Retrieve configuration of the specified modelweights
            qs = models.TrainingSetting.objects.filter(modelweights_id=i.modelweights_id)

            # Create the dict of training settings
            for setting in qs:
                hyperparams[setting.property_id.name] = setting.value

            config = nn_settings(modelweight=i.modelweights_id, hyperparams=hyperparams, mode='inference')
            if not config:
                return Response({"Error": "Properties error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Launch the inference
            # Differentiate the task and start training
            if task_name == 'classification':
                # celery_id = classification.inference.delay(config)
                celery_id = classification.inference(config)
            elif task_name == 'segmentation':
                # celery_id = segmentation.inference.delay(config)
                celery_id = segmentation.inference(config)
            else:
                return Response({'error': 'error on task'}, status=status.HTTP_400_BAD_REQUEST)

            i.celery_id = celery_id.id
            i.save()
            response = {
                "result": "ok",
                "process_id": celery_id.id
            }
            return Response(response, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ModelViewSet(mixins.ListModelMixin,
                   viewsets.GenericViewSet):
    """
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
                          viewsets.GenericViewSet):
    """
    When 'use pre-trained' is selected, it is possible to query the backend passing a `model_id` to obtain a list
    of dataset on which it was pretrained.

    ## Parameters
    `model_id`: integer
        Required integer representing the model.
    """
    queryset = models.ModelWeights.objects.all()
    serializer_class = serializers.ModelWeightsSerializer
    params = ['model_id']

    def get_queryset(self):
        model_id = self.request.query_params.get('model_id')
        self.queryset = models.ModelWeights.objects.filter(model_id=model_id)
        return self.queryset


class ProjectViewSet(mixins.ListModelMixin,
                     mixins.RetrieveModelMixin,
                     mixins.CreateModelMixin,
                     viewsets.GenericViewSet):
    """
    View which provides _get_, _put_, and _post_ methods:

    - _get_ retrieves all the projects or a single one using `projects/{id}` link.
    - _post_ lets to create a new project.
    - _put_ updates a project.
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
    This API allows the client to know which properties are "globally" supported by the backend.


    A model can have different default and allowed values if the `/allowdProperties` return an entry.
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
    This  API allows the frontend to query the status of a training or inference, identified by a `process_id` (which is returned by `/train` or `/inference` APIs).
    """

    def get(self, request):
        if not self.request.query_params.get('process_id'):
            error = {'Error': f'Missing required parameter `process_id`'}
            return Response(data=error, status=status.HTTP_400_BAD_REQUEST)
        process_id = self.request.query_params.get('process_id')
        try:
            # _ = models.ModelWeights.objects.get(id=process_id)
            weight = models.ModelWeights.objects.get(celery_id=process_id)
        except models.ModelWeights.DoesNotExist:
            res = {
                "result": "error",
                "error": "This model weights has been deleted due a training stop."
            }
            return Response(data=res, status=status.HTTP_404_NOT_FOUND)
        try:
            with open(f'data/logs/{weight.id}.log', 'r') as f:
                lines = f.read().splitlines()
                last_line = lines[-1]
        except:
            res = {
                "result": "error",
                "error": "Server error"
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


class TaskViewSet(mixins.ListModelMixin,
                  viewsets.GenericViewSet):
    """
    This API allows the client to know which task this platform supports. e.g. classification or segmentation tasks.
    """
    queryset = models.Task.objects.all()
    serializer_class = serializers.TaskSerializer


class TrainViewSet(views.APIView):
    """
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

            weight.save()
            ckpts_dir = os.path.join(settings.TRAINING_DIR, 'ckpts/')
            weight.location = Path(f'{ckpts_dir}/{weight.id}.bin').absolute()
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
                celery_id = classification.training.delay(config)
                # celery_id = classification(args)
            elif task_name == 'segmentation':
                celery_id = segmentation.training.delay(config)
                # celery_id = segmentation(args)
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
                "process_id": celery_id.id
            }
            return Response(response, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class TrainingSettingViewSet(BAMixins.ParamListModelMixin,
                             viewsets.GenericViewSet):
    """
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


class StopProcessViewSet(views.APIView):
    """
    Stop a training process specifying a `process_id` (which is returned by `/train` or `/inference` APIs).
    """

    def post(self, request):
        serializer = serializers.StopProcessSerializer(data=request.data)
        if serializer.is_valid():
            process_id = serializer.data['process_id']
            weights = models.ModelWeights.objects.filter(celery_id=process_id)
            infer = models.Inference.objects.filter(celery_id=process_id)
            response = {"result": "Process stopped"}
            if not weights and not infer:
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


class OutputViewSet(views.APIView):
    """
    This API provides information about an `inference` process.In classification task it returns the list
    of images and an array composed of the classes predictionsand the ground truth as last element.
    In segmentation task it returns the URLs of the segmented images.
    """

    def trunc(self, values, decs=0):
        return np.trunc(values * 10 ** decs) / (10 ** decs)

    def get(self, request):
        preds_dir = os.path.join(settings.INFERENCE_DIR, 'predictions')
        process_id = request.GET['process_id']
        infer = models.Inference.objects.filter(celery_id=process_id)
        if not infer:
            # already deleted weight/training or inference
            return Response({"result": "Process stopped before finishing or non existing"},
                            status=status.HTTP_404_NOT_FOUND)

        infer = infer.first()
        # Differentiate classification and segmentation
        if infer.modelweights_id.task_id.name.lower() == 'classification':
            preds = np.load(f'{preds_dir}/{infer.modelweights_id.id}.npy').astype(np.float64)
            # response = json.dumps({'outputs': preds}, cls=utils.NumpyEncoder)
            # preds[:, :-1] = np.around(preds[:, :-1], 4)
            preds = self.trunc(preds, decs=8)
            response = {'outputs': preds.tolist()}
            return Response(response, status=status.HTTP_200_OK)
        else:
            # Segmentation
            pass
        # return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
