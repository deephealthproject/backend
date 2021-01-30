from rest_framework import serializers

from streamflow_app import models


class SFEnvSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    type = serializers.ChoiceField(choices=models.SFEnvironment.SFConfigType.choices)

    def validate(self, data):
        user = self.context['request'].user
        id = data['id']
        type = data['type']
        model = models.choice_to_model(type)
        if not model.objects.filter(id=id).exists() or not model.objects.filter(id=id, user=user).exists():
            raise serializers.ValidationError({'id': f"There is no {type} environment with id `{id}`."})
        return data


class SFSSHSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.SFSSH
        fields = ['id', 'name', 'username', 'hostname', 'ssh_key', 'file']
        extra_kwargs = {
            'ssh_key_passphrase': {'write_only': True},
        }

    def create(self, validated_data):
        validated_data['user'] = self.context['request'].user
        o = self.Meta.model.objects.create(**validated_data)
        o.save()
        return o


class SFHelmSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.SFHelm
        fields = '__all__'
        extra_kwargs = {
            'password': {'write_only': True},
        }

    def create(self, validated_data):
        validated_data['user'] = self.context['request'].user
        o = self.Meta.model.objects.create(**validated_data)
        o.save()
        return o
