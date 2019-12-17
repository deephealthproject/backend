from rest_framework import serializers

from backend_app import models


class AllowedPropSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.AllowedProperty
        # fields = '__all__'
        exclude = ['id']


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Dataset
        fields = ['id', 'name']


class DropDownSerializer(serializers.Serializer):
    default = serializers.BooleanField(read_only=True, default=False)
    id = serializers.IntegerField()
    name = serializers.CharField()


class InferenceSerializer(serializers.ModelSerializer):
    project_id = serializers.IntegerField()

    class Meta:
        model = models.Inference
        exclude = ['stats']


class ModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Model
        fields = ['id', 'name']


class ModelWeightsSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ModelWeights
        # fields = ['id', 'pretrained_on']
        # fields = '__all__'
        exclude = ['location']


class PretrainingSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Dataset
        fields = ['id', 'name']


class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Project
        fields = '__all__'
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
        # fields = '__all__'
        exclude = ['id']


class WeightsSerializer(serializers.ModelSerializer):
    pretraining = PretrainingSerializer(source='pretraining_id')

    class Meta:
        model = models.ModelWeights
        fields = ['id', 'pretraining']
