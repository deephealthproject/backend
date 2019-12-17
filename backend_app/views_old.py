from django.http import Http404
from rest_framework import mixins, status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from backend_app import models, serializers

models_str = {
    'model': [models.Model, None],
    'pretraining': [models.Dataset, True],
    'finetuning': [models.Dataset, False],
    'loss': [models.PropertyInstance, models.Property],
    'input_size': [models.PropertyInstance, models.Property],
    'optimizer': [models.PropertyInstance, models.Property],
    'learning_rate': [models.PropertyInstance, models.Property],
    'epochs': [models.PropertyInstance, models.Property],
}

property_str = {
    'input_size': '',
    'loss': 'Loss function',
    'optimizer': '',
    'learning_rate': 'Learning rate',
    'epochs': 'epochs',
}


# serializers_str = {
#     'model': ModelSchema,
#     'pretraining': DatasetSchema,
#     'finetuning': DatasetSchema,
#     'input_size': PropertyInstanceSchema,
#     'loss': PropertyInstanceSchema,
#     'optimizer': PropertyInstanceSchema,
#     'learning_rate': PropertyInstanceSchema,
#     'epochs': PropertyInstanceSchema,
# }

class TaskViewSet(mixins.ListModelMixin,
                  viewsets.GenericViewSet):
    """View which retrieves all the tasks"""
    queryset = models.Task.objects.all()
    serializer_class = serializers.TaskSerializer

    def get_serializer_context(self):
        return {'project_id': self.request.GET.get('project_id')}


class ProjectViewSet(mixins.ListModelMixin,
                     mixins.RetrieveModelMixin,
                     # mixins.CreateModelMixin,
                     # mixins.UpdateModelMixin,
                     viewsets.GenericViewSet):
    """
    View which provides `/get`, `/put`, and `/post` methods

    - `/get` retrieves all the projects.
    - `/put` updates a project.
    - `/post` lets to create a new project.
    """
    queryset = models.Project.objects.all()
    serializer_class = serializers.ProjectSerializer

    def get_object(self, pk):
        try:
            return models.Project.objects.get(pk=pk)
        except models.Project.DoesNotExist:
            raise Http404

    def put(self, request, *args, **kwargs):
        project = self.get_object(request.data['id'])
        serializer = serializers.ProjectSerializer(project, data=request.data)
        if serializer.is_valid():
            serializer.save()
            # Returns all the elements
            return self.list(request)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        # Returns all the elements
        return self.list(request, status=status.HTTP_201_CREATED, headers=headers)


class ModelWeightsList(APIView):
    """
      List all modelweights of a certain model
      """

    def get(self, request):
        model_id = request.GET.get('model_id')
        if model_id is None:
            return Response(status=status.HTTP_400_BAD_REQUEST)
        model_id = int(model_id)
        try:
            model = models.Model.objects.get(id=model_id)
            # Retrieve all the modelweights related to model
            weights = model.modelweights_set.all()
            serializer = serializers.WeightsSerializer(weights, many=True)
            return Response(serializer.data)
        except models.Model.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)


class DropDownList(APIView):
    """
      TODO
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.task_dependent_entities = ['model', 'pretraining', 'finetuning']
        self.task_dependent_entities = [models.Model, models.Dataset]

    def get(self, request):
        project_id = request.GET.get('project_id')
        dropdown_name = request.GET.get('dropdown_name')
        if project_id is None or dropdown_name is None:
            return Response(status=status.HTTP_400_BAD_REQUEST)
        project_id = int(project_id)
        try:
            project = models.Project.objects.get(id=project_id)
        except models.Project.DoesNotExist:
            return Response(data={'error': 'project_id not found'}, status=status.HTTP_404_NOT_FOUND)

        try:
            entity = models_str[dropdown_name][0]
            entity_meta = models_str[dropdown_name][1]
        except KeyError:
            return Response(data={'error': 'dropdown_name not present'}, status=status.HTTP_400_BAD_REQUEST)

        if project.modelweights_id is None:
            # New project

            if entity in self.task_dependent_entities:
                # Entity that depends on task

                queryset = entity.objects.filter(task_id=project.task_id)
                if entity_meta is not None:
                    queryset = queryset.filter(ispretraining=entity_meta)
                serializer = serializers.DropDownSerializer(queryset, many=True)
                # Set the first of the query as True
                if serializer.data:
                    serializer.data[0]['default'] = True
                return Response(serializer.data)
            else:
                # Entity that does not depend on task --> Property

                name = property_str[dropdown_name]

                queryset = entity_meta.objects.filter(name__iexact=name)
                serializer = serializers.DropDownSerializer(queryset, many=True)
                # Set the first of the query as True
                if serializer.data:
                    serializer.data[0]['default'] = True
                    for s, q in zip(serializer.data, queryset):
                        s['name'] = q.default
                        # s['id'] = None
                return Response(serializer.data)
        else:
            pass
        queryset = entity.objects.filter()
        # project.

        # entity.

        # try:
        #     model = models.Model.objects.get(id=model_id)
        #     # Retrieve all the modelweights related to model
        #     weights = model.modelweights_set.all()
        #     serializer = serializers.WeightsSerializer(weights, many=True)
        #     return Response(serializer.data)
        # except models.Model.DoesNotExist:
        #     return Response(status=status.HTTP_404_NOT_FOUND)
