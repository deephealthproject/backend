from rest_framework import serializers

from backend_app import models


class TaskSerializer(serializers.ModelSerializer):
    default = serializers.SerializerMethodField(default=False)

    class Meta:
        model = models.Task
        fields = '__all__'

    def get_default(self, obj):
        project_id = self.context.get('project_id')
        if project_id is None:
            raise Exception('project_id parameter is required')
        if int(project_id) == obj.id:
            return True
        return False


class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Project
        fields = '__all__'
        # exclude = ['task', 'modelweights']


class PretrainingSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Dataset
        fields = ['id', 'name']


class WeightsSerializer(serializers.ModelSerializer):
    pretraining = PretrainingSerializer(source='pretraining_id')

    class Meta:
        model = models.ModelWeights
        fields = ['id', 'pretraining']


class DropDownSerializer(serializers.Serializer):
    default = serializers.BooleanField(read_only=True, default=False)
    id = serializers.IntegerField()
    name = serializers.CharField()
