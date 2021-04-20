from typing import Dict, List

from django.contrib.auth import get_user_model
from django.core.exceptions import ObjectDoesNotExist
from rest_framework import exceptions
from rest_framework import serializers
from rest_framework.utils import model_meta

from backend_app import models
from streamflow_app import serializers as sf_serializers


class ReadWriteSerializerMethodField(serializers.SerializerMethodField):
    """
    Let to serialize M2M through models
    """

    def __init__(self, method_name=None, **kwargs):
        super().__init__(**kwargs)
        kwargs['source'] = '*'
        self.read_only = False
        self.required = True

    def to_internal_value(self, data):
        return {self.field_name: data}


class PermissionSerializer(serializers.ModelSerializer):
    """
    Base class which models a Permission class
    """
    username = serializers.CharField(source='user.username')

    class Meta:
        fields = ('username', 'permission')


def check_users(users: List[Dict], field_name: str = 'users') -> None:
    """Check if user and permission have been sent correctly

    :param users: List of users and their permissions
    :param field_name: Field name to return in errors
    """
    owners = 0
    if not len(users):
        raise exceptions.ParseError({"Error": f"'{field_name}' list cannot be empty"})
    if not isinstance(users, list):
        raise exceptions.ParseError({"Error": f"'{field_name}' must be a list of dictionaries"})
    for u in users:
        try:
            if u.get('permission') is None:
                raise exceptions.ParseError({"Error": f"'{field_name}' field must be always associated with its "
                                                      f"permission level"})
            if u.get('permission') == models.Perm.OWNER: owners += 1
            get_user_model().objects.get(username__exact=u.get('username'))
        except ObjectDoesNotExist:
            raise exceptions.ParseError({"Error": f"User `{u.get('username')}` does not exist"})
    if not owners:
        raise exceptions.ParseError({"Error": f"At least one user in '{field_name}' must be Owner"})


class AllowedPropertySerializer(serializers.ModelSerializer):
    class Meta:
        model = models.AllowedProperty
        fields = '__all__'


class DatasetPermissionSerializer(PermissionSerializer):
    class Meta(PermissionSerializer.Meta):
        model = models.DatasetPermission


class M2MSerializer(serializers.ModelSerializer):
    users = ReadWriteSerializerMethodField()

    def __init__(self, name, *args, **kwargs):
        self.name_ = name
        super(M2MSerializer, self).__init__(*args, **kwargs)

    def get_users(self, obj):
        m = eval('models.' + self.name_)
        s = eval(self.name_ + 'Serializer')
        field = [f.name for f in m._meta.fields if f.name not in ['user', 'id', 'permission']]
        assert len(field) == 1
        qset = m.objects.filter(**{field[0]: obj})
        return [s(i).data for i in qset]

    def validate(self, attrs):
        check_users(attrs.get('users'))
        return super().validate(attrs)


class DatasetSerializer(M2MSerializer):
    def __init__(self, *args, **kwargs):
        super().__init__('DatasetPermission', *args, **kwargs)

    class Meta:
        model = models.Dataset
        fields = ['id', 'name', 'path', 'task_id', 'users', 'public', 'ctype', 'ctype_gt', 'classes']
        extra_kwargs = {
            'ctype': {'required': False},
            'ctype_gt': {'required': False},
            'classes': {'read_only': True},
        }

    def create(self, validated_data):
        users = validated_data.pop('users')
        instance = models.Dataset.objects.create(**validated_data)
        for u in users:
            user = get_user_model().objects.get(username__exact=u.get('username'))
            perm = u.get('permission')
            _ = models.DatasetPermission.objects.get_or_create(user=user, dataset=instance, permission=perm)
        return instance


class InferenceSerializer(serializers.ModelSerializer):
    task_manager = serializers.ChoiceField(choices=['CELERY', 'STREAMFLOW'], write_only=True, default='CELERY')
    env = sf_serializers.SFEnvSerializer(required=False)

    class Meta:
        model = models.Inference
        fields = ['project_id', 'modelweights_id', 'dataset_id', 'celery_id', 'task_manager', 'env']
        read_only_fields = ['celery_id', 'logfile', 'outputfile', 'stats']
        extra_kwargs = {
            'task_manager': {'write_only': True},
            'env': {'write_only': True, 'required': False},
        }

    def validate(self, data):
        if data.get('task_manager') == 'STREAMFLOW' and data.get('env') is None:
            # there must be env
            raise serializers.ValidationError(
                {'task_manager': f"`STREAMFLOW` task_manager also requires the `env` parameters."})
        return data


class InferenceSingleSerializer(serializers.ModelSerializer):
    image_url = serializers.URLField(required=False)
    image_data = serializers.CharField(required=False)
    task_manager = serializers.ChoiceField(choices=['CELERY', 'STREAMFLOW'], write_only=True, default='CELERY')
    env = sf_serializers.SFEnvSerializer(required=False)

    class Meta:
        model = models.Inference
        fields = ['project_id', 'modelweights_id', 'image_url', 'image_data', 'task_manager', 'env']
        extra_kwargs = {
            'task_manager': {'write_only': True},
            'env': {'write_only': True, 'required': False},
        }

    def validate(self, data):
        if not data.get('image_url') and not data.get('image_data'):
            raise serializers.ValidationError("At least one between `image_url` and `image_data` is needed.")
        if data.get('task_manager') == 'STREAMFLOW' and data.get('env') is None:
            # there must be env
            raise serializers.ValidationError(
                {'task_manager': f"`STREAMFLOW` task_manager also requires the `env` parameters."})
        return data


class ModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Model
        fields = ['id', 'name', 'task_id']


class ModelWeightsPermissionSerializer(PermissionSerializer):
    class Meta(PermissionSerializer.Meta):
        model = models.ModelWeightsPermission


class ModelWeightsCreateSerializer(serializers.ModelSerializer):
    onnx_url = serializers.URLField(required=False)
    onnx_data = serializers.FileField(required=False)

    class Meta:
        model = models.ModelWeights
        fields = ['id', 'name', 'model_id', 'dataset_id', 'layer_to_remove', 'onnx_url', 'onnx_data']
        extra_kwargs = {
            'onnx_url': {'write_only': True},
            'onnx_data': {'write_only': True},
        }

    def validate(self, data):
        if not data.get('onnx_url') and not data.get('onnx_data'):
            raise serializers.ValidationError("At least one between `onnx_url` and `onnx_data` fields is needed.")
        return data


class ModelWeightsSerializer(M2MSerializer):

    def __init__(self, *args, **kwargs):
        super().__init__('ModelWeightsPermission', *args, **kwargs)

    class Meta:
        model = models.ModelWeights
        fields = ['id', 'name', 'model_id', 'dataset_id', 'pretrained_on', 'public', 'users', 'process_id',
                  'layer_to_remove', 'is_active']
        extra_kwargs = {
            'process_id': {'read_only': True},
            'is_active': {'read_only': True},
        }

    def update(self, instance, validated_data):
        users = validated_data.pop('users')

        # Update existing attributes
        info = model_meta.get_field_info(instance)
        for attr, value in validated_data.items():
            if attr not in info.relations or not info.relations[attr].to_many:
                setattr(instance, attr, value)
        instance.save()

        current_user = self.context['request'].user

        try:
            current_user_perm = instance.permission.get(user=current_user).permission
        except instance.permission.DoesNotExist:
            # User has no permission on this object but can see it because it's public
            current_user_perm = None

        if current_user_perm == models.Perm.VIEWER or current_user_perm is None:
            # User has no rights. Abort
            raise exceptions.PermissionDenied(
                {"Error": f'User `{current_user}` does not have owner permission on this Weight'})

        # Revoke permissions for every users
        instance.users.clear()

        # Add new permissions
        for u in users:
            user = get_user_model().objects.get(username__exact=u.get('username'))
            perm = u.get('permission')
            pp = models.ModelWeightsPermission.objects.get_or_create(user=user, modelweight=instance)[0]
            pp.permission = perm
            pp.save()
        return instance


class ProjectPermissionSerializer(PermissionSerializer):
    class Meta(PermissionSerializer.Meta):
        model = models.ProjectPermission


class ProjectSerializer(M2MSerializer):
    def __init__(self, *args, **kwargs):
        super().__init__('ProjectPermission', *args, **kwargs)

    class Meta:
        model = models.Project
        fields = ['id', 'name', 'task_id', 'users']

    def create(self, validated_data):
        users = validated_data.pop('users')
        p = models.Project.objects.create(**validated_data)
        for u in users:
            user = get_user_model().objects.get(username__exact=u.get('username'))
            perm = u.get('permission')
            _ = models.ProjectPermission.objects.get_or_create(user=user, project=p, permission=perm)
        return p

    def update(self, instance, validated_data):
        users = validated_data.pop('users')
        # Update existing attributes
        info = model_meta.get_field_info(instance)
        for attr, value in validated_data.items():
            if attr not in info.relations or not info.relations[attr].to_many:
                setattr(instance, attr, value)
        instance.save()

        current_user = self.context['request'].user
        current_user_perm = instance.permission.get(user=current_user).permission
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


class StatusSerializer(serializers.Serializer):
    process_id = serializers.UUIDField()
    full = serializers.BooleanField(required=False)


class StopProcessSerializer(serializers.Serializer):
    process_id = serializers.UUIDField()


class TaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Task
        fields = '__all__'


class TrainingSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Training
        fields = ['id', 'celery_id', 'project_id', 'modelweights_id']


class TrainSerializer(serializers.Serializer):
    dataset_id = serializers.IntegerField()
    weights_id = serializers.IntegerField()
    project_id = serializers.IntegerField()
    properties = PropertyTrainSerializer(many=True)

    task_manager = serializers.ChoiceField(choices=['CELERY', 'STREAMFLOW'], write_only=True, default='CELERY')
    env = sf_serializers.SFEnvSerializer(required=False)

    def validate(self, data):
        if data.get('task_manager') == 'STREAMFLOW' and data.get('env') is None:
            # there must be env
            resp = {'task_manager': f"`STREAMFLOW` task_manager also requires the `env` parameters."}
            raise serializers.ValidationError(resp)

        return data


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
    process_type = serializers.CharField()
