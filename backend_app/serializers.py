from django.contrib.auth import get_user_model
from django.core.exceptions import ObjectDoesNotExist
from rest_framework import serializers
from rest_framework.utils import model_meta

from auth import serializers as auth_serializers
from backend_app import models
from rest_framework import exceptions


class ReadWriteSerializerMethodField(serializers.SerializerMethodField):
    def __init__(self, method_name=None, **kwargs):
        super().__init__(**kwargs)
        kwargs['source'] = '*'
        self.read_only = False

    def to_internal_value(self, data):
        return {self.field_name: data}


def check_users(users):
    if not len(users):
        raise exceptions.ParseError({"Error": f"'users' list cannot be empty"})
    if not isinstance(users, list):
        raise exceptions.ParseError({"Error": f"'users' must be a list of dictionaries"})
    for u in users:
        try:
            get_user_model().objects.get(username__exact=u.get('username'))
        except ObjectDoesNotExist:
            raise exceptions.ParseError({"Error": f"User `{u.get('username')}` does not exist"})


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
        perm = models.Perm.OWNER
        for owner_data in owners_data:
            user = get_user_model().objects.get(username__exact=owner_data.get('username'))
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
            raise serializers.ValidationError("At least one between `image_url` and `image_data` is needed.")
        return data


class ModelSerializer(serializers.ModelSerializer):
    onnx_url = serializers.URLField(required=False)
    onnx_data = serializers.FileField(required=False)
    dataset_id = serializers.PrimaryKeyRelatedField(required=False, queryset=models.Dataset.objects.all())

    class Meta:
        model = models.Model
        fields = ['id', 'name', 'task_id', 'onnx_url', 'onnx_data', 'dataset_id']
        write_only_fields = ['onnx_url', 'onnx_data', 'dataset_id']

    def validate(self, data):
        if not data.get('onnx_url') and not data.get('onnx_data'):
            raise serializers.ValidationError("At least one between `onnx_url` and `onnx_data` is needed.")
        return data


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

        perm = models.Perm.OWNER
        users = []
        # Give permissions to new users
        for owner_data in owners_data:
            user = get_user_model().objects.get(username__exact=owner_data.get('username'))
            users.append(user)
            _ = models.ModelWeightsPermission.objects.get_or_create(user=user, modelweight=weight, permission=perm)
        # Remove permissions to excluded users
        for owner in weight.owners.all():
            if owner not in users:
                weight.owners.remove(owner)
        return weight


class ProjectPermissionSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username')

    class Meta:
        model = models.ProjectPermission
        fields = ('username', 'permission')


class ProjectSerializer(serializers.ModelSerializer):
    users = ReadWriteSerializerMethodField()

    class Meta:
        model = models.Project
        fields = ['id', 'name', 'task_id', 'users']

    def get_users(self, obj):
        qset = models.ProjectPermission.objects.filter(project=obj)
        return [ProjectPermissionSerializer(pp).data for pp in qset]

    def create(self, validated_data):
        users = validated_data.pop('users')
        check_users(users)
        p = models.Project.objects.create(**validated_data)
        for u in users:
            user = get_user_model().objects.get(username__exact=u.get('username'))
            perm = u.get('permission')
            _ = models.ProjectPermission.objects.get_or_create(user=user, project=p, permission=perm)
        return p

    def update(self, instance, validated_data):
        users = validated_data.pop('users')
        check_users(users)
        current_user = self.context['request'].user
        current_user_perm = instance.projectpermission_set.get(user=current_user).permission
        if current_user_perm == models.Perm.VIEWER:
            # User has no rights. Abort
            raise exceptions.PermissionDenied(
                {"Error": f'User `{current_user}` does not have owner permission on this Project'})

        # Revoke permissions for every users
        instance.users.clear()

        # Add new permissions
        for u in users:
            user = get_user_model().objects.get(username__exact=u.get('username'))
            perm = u.get('permission')
            pp = models.ProjectPermission.objects.get_or_create(user=user, project=instance)[0]
            pp.permission = perm
            pp.save()
        return instance


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


class ModelStatusResponse(serializers.Serializer):
    result = serializers.ChoiceField(["PENDING", "STARTED", "RETRY", "FAILURE", "SUCCESS"])
