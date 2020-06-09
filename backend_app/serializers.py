from rest_framework import serializers

from backend_app import models


class AllowedPropertySerializer(serializers.ModelSerializer):
    class Meta:
        model = models.AllowedProperty
        fields = '__all__'
        # exclude = ['id']


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Dataset
        fields = ['id', 'name', 'path', 'task_id']
        write_only_fields = ['name', 'path', 'task_id']  # Only for post


class InferenceSerializer(serializers.ModelSerializer):
    project_id = serializers.IntegerField()

    class Meta:
        model = models.Inference
        fields = ['project_id', 'modelweights_id', 'dataset_id']
        # exclude = ['stats']


class InferenceSingleSerializer(serializers.ModelSerializer):
    project_id = serializers.IntegerField()
    image_url = serializers.URLField()

    class Meta:
        model = models.Inference
        exclude = ['stats', 'dataset_id', 'logfile']
        # write_only_fields = ['modelweights_id', 'image_url', 'project_id']


class ModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Model
        fields = ['id', 'name', 'location', 'task_id']


class ModelWeightsSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ModelWeights
        fields = ['id', 'name', 'celery_id', "model_id", "dataset_id", "pretrained_on"]
        read_only_fields = ['location', 'celery_id', 'logfile']
        write_only_fields = ['id']


class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Project
        fields = '__all__'
        # fields = ['id', 'name', 'task_id', 'modelweights_id', 'inference_id']
        # exclude = ['task', 'modelweights']


class PropertyListSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Property
        # fields = ['id', 'name']
        fields = '__all__'


class PropertyTrainSerializer(serializers.ModelSerializer):
    value = serializers.CharField()

    class Meta:
        model = models.Property
        fields = ['id', 'name', 'value']


class TaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Task
        fields = '__all__'


class TrainSerializer(serializers.Serializer):
    dataset_id = serializers.IntegerField()
    model_id = serializers.IntegerField()
    project_id = serializers.IntegerField()
    properties = PropertyTrainSerializer(many=True)
    weights_id = serializers.IntegerField(allow_null=True)


class TrainingSettingSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.TrainingSetting
        fields = '__all__'
        # exclude = ['id']


class StopProcessSerializer(serializers.Serializer):
    process_id = serializers.UUIDField()


# RESPONSES SERIALIZERS

class GeneralResponse(serializers.Serializer):
    result = serializers.CharField()


class GeneralErrorResponse(serializers.Serializer):
    result = serializers.CharField()
    error = serializers.CharField()


class InferenceResponseSerializer(serializers.Serializer):
    result = serializers.CharField()
    process_id = serializers.UUIDField()


class OutputsResponse(serializers.Serializer):
    outputs = serializers.ListField(
        child=serializers.ListField(
            child=serializers.ListField(child=serializers.Field(), min_length=2, max_length=2)))


class TrainResponse(serializers.Serializer):
    result = serializers.CharField()
    process_id = serializers.UUIDField()


class StatusStatusResponse(serializers.Serializer):
    process_type = serializers.CharField()
    process_status = serializers.CharField()
    process_data = serializers.CharField()


class StatusResponse(serializers.Serializer):
    result = serializers.CharField()
    status = StatusStatusResponse()
