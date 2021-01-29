from rest_framework import serializers

from streamflow_app import models


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
