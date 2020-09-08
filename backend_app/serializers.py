from django.contrib.auth.models import User
from rest_framework import serializers
from rest_framework.utils import model_meta

from auth import serializers as auth_serializers
from backend_app import models


class AllowedPropertySerializer(serializers.ModelSerializer):
    class Meta:
        model = models.AllowedProperty
        fields = '__all__'
        # exclude = ['id']


class DatasetSerializer(serializers.ModelSerializer):
    owners = auth_serializers.UserSerializerNotUnique(many=True)

    class Meta:
        model = models.Dataset
        fields = ['id', 'name', 'path', 'task_id', 'owners', 'public']
        write_only_fields = ['name', 'path', 'task_id']  # Only for post

    def create(self, validated_data):
        owners_data = validated_data.pop('owners')
        dataset = models.Dataset.objects.create(**validated_data)
        perm = models.PERM[0][0]
        for owner_data in owners_data:
            user = User.objects.get(username__exact=owner_data.get('username'))
            _ = models.DatasetPermission.objects.get_or_create(user=user, dataset=dataset, permission=perm)
        return dataset

    # def update(self, instance, validated_data):
    #     pass


class InferenceSerializer(serializers.ModelSerializer):
    project_id = serializers.IntegerField()

    class Meta:
        model = models.Inference
        fields = ['project_id', 'modelweights_id', 'dataset_id']
        # exclude = ['stats']


class InferenceSingleSerializer(serializers.ModelSerializer):
    project_id = serializers.IntegerField()
    image_url = serializers.URLField(required=False)
    image_data = serializers.CharField(required=False)

    class Meta:
        model = models.Inference
        fields = ['project_id', 'modelweights_id', 'image_url', 'image_data']

    def validate(self, data):
        if not data.get('image_url') and not data.get('image_data'):
            raise serializers.ValidationError("At least one of `image_url` and `image_data` is needed.")
        return data


class ModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Model
        fields = ['id', 'name', 'location', 'task_id']


class ModelWeightsSerializer(serializers.ModelSerializer):
    owners = auth_serializers.UserSerializerNotUnique(many=True)

    class Meta:
        model = models.ModelWeights
        fields = ['id', 'name', 'model_id', 'dataset_id', 'pretrained_on', 'public', 'owners']
        read_only_fields = ['location']
        write_only_fields = ['id']

    def update(self, weight, validated_data):
        info = model_meta.get_field_info(weight)
        for attr, value in validated_data.items():
            if attr not in info.relations or not info.relations[attr].to_many:
                setattr(weight, attr, value)
        weight.save()

        owners_data = validated_data.pop('owners')

        # Weights must belong to at least 1 user
        if not len(owners_data):
            raise serializers.ValidationError({"Error": f"Owners list cannot be empty"})

        perm = models.PERM[0][0]
        users = []
        # Give permissions to new users
        for owner_data in owners_data:
            user = User.objects.get(username__exact=owner_data.get('username'))
            users.append(user)
            _ = models.ModelWeightsPermission.objects.get_or_create(user=user, modelweight=weight, permission=perm)
        # Remove permissions to excluded users
        for owner in weight.owners.all():
            if owner not in users:
                weight.owners.remove(owner)
        return weight


class ProjectSerializer(serializers.ModelSerializer):
    users = auth_serializers.UserSerializer(many=True, read_only=True)

    class Meta:
        model = models.Project
        fields = ['id', 'name', 'task_id', 'users']


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


class StopProcessSerializer(serializers.Serializer):
    process_id = serializers.UUIDField()


class TaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Task
        fields = '__all__'


class TrainingSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Training
        fields = ['id', 'celery_id', 'logfile', 'project_id', 'modelweights_id']


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


##########################
#### RESPONSES SERIALIZERS
##########################


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
