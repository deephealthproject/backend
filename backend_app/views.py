from celery import shared_task
from django.db.models import Q
from rest_framework import mixins, status, views, viewsets
from rest_framework.response import Response

from backend_app import mixins as BAMixins, models, serializers
from deeplearning.utils import dotdict
from deeplearning.tasks.classification import classification


class AllowedPropViewSet(BAMixins.ParamListModelMixin,
                         viewsets.GenericViewSet):
    """
    todo description

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
        task_id = self.request.query_params.get('task_id')
        if task_id:
            self.queryset = models.Dataset.objects.filter(task_id=task_id)
        return self.queryset


class InferenceViewSet(views.APIView):
    """
    This is the main entry point to start the inference.
    It is mandatory to specify a pre-trained model and a dataset for finetuning.
    """

    def post(self, request, format=None):
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
            response = {
                "result": "ok",
                "process_id": i.id
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
    - _put_ updates a project.
    - _post_ lets to create a new project.
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
    todo description
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
    This is the API which allows the frontend to query the status of an
    operation, identified by its `process_id`. The result depends on the
    kind of operation identified by the `process_id`.
    """

    def get(self, request):
        if not self.request.query_params.get('process_id'):
            error = {'Error': f'Missing required parameter `process_id`'}
            return Response(data=error, status=status.HTTP_400_BAD_REQUEST)
        process_id = self.request.query_params.get('process_id')
        try:
            with open(f'log_{process_id}.log', 'r') as f:
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

    def post(self, request, format=None):
        serializer = serializers.TrainSerializer(data=request.data)

        if serializer.is_valid():
            # Create a new modelweights and start training
            weight = models.ModelWeights()
            weight.dataset_id_id = serializer.data['dataset_id']
            weight.model_id_id = serializer.data['model_id']
            project = models.Project.objects.get(id=serializer.data['project_id'])
            weight.task_id = project.task_id
            weight.location = 'location'
            weight.name = 'name'
            if serializer.data['weights_id']:
                weight.pretrained_on_id = serializer.data['weights_id']
            weight.save()

            props = serializer.data['properties']
            hyperparams = {}
            for p in props:
                ts = models.TrainingSetting()
                # Get the property by name
                name = p['name']
                name = [name, name.replace('_', ' ')]
                queryset = models.Property.objects.filter(Q(name__icontains=name[0]) | Q(name__icontains=name[1]))
                if len(queryset) == 0:
                    # Property does not exist
                    weight.delete()
                    error = {"Error": f"Property `{p['name']}` does not exist"}
                    return Response(error, status=status.HTTP_400_BAD_REQUEST)
                property = queryset[0]
                ts.property_id = property
                ts.modelweights_id = weight
                ts.value = str(p['value'])
                ts.save()
                hyperparams[property.name] = ts.value

            args = {
                'weight_id': weight.id,
                'model': 'LeNet',
                'num_classes': 10,
                'pretrained': False,

                'batch_size': 64,
                'test_batch_size': 8,
                'lr': 1e-4,
                'epochs': 1,

                'log_interval': 50,
                'save_model': True,
                'gpu': True,
                'in_ds': '/mnt/data/DATA/mnist/mnist.yml',
            }

            # Start training todo
            classification.delay(args)
            # classification(args)

            response = {
                "result": "ok",
                "process_id": weight.id
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


class Dummy(views.APIView):

    def get(self, request):
        args = {
            'weight_id': 1,
            'model': 'LeNet',
            'num_classes': 10,
            'pretrained': False,

            'batch_size': 64,
            'test_batch_size': 8,
            'lr': 1e-4,
            'epochs': 1,

            'log_interval': 50,
            'save_model': True,
            'gpu': True,
            'in_ds': '/mnt/data/DATA/mnist/mnist.yml',
        }

        classification.delay(args)
        # classification(args)

        # print(res.get())
        return Response({'ok': 'ok'}, status=status.HTTP_200_OK)


class DummyStatus(views.APIView):

    def get(self, request):
        if not self.request.query_params.get('process_id'):
            error = {'Error': f'Missing required parameter `process_id`'}
            return Response(data=error, status=status.HTTP_400_BAD_REQUEST)
        process_id = self.request.query_params.get('process_id')
        try:
            with open(f'log_{process_id}.log', 'r') as f:
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
