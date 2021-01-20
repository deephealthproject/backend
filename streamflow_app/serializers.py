from rest_framework import serializers

from streamflow_app import models


class SFSSHSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.SFSSH
        fields = ['id', 'name', 'username', 'hostname', 'ssh_key', 'file']
        write_only_fields = ['ssh_key_passphrase']

    def create(self, validated_data):
        o = models.SFSSH.objects.create(**validated_data)
        o.type = models.SFEnvironment.SFConfigType.SSH
        o.user = self.context['request'].user
        o.save()
        return o
