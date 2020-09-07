from django.contrib.auth.models import Group, User
from rest_framework import serializers
from django.contrib.auth.validators import UnicodeUsernameValidator


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['username']
        read_only = ['id', 'email', 'first_name', 'last_name']


class UserSerializerNotUnique(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['username']
        read_only = ['id', 'email', 'first_name', 'last_name']
        extra_kwargs = {
            'username': {'validators': [UnicodeUsernameValidator()], }
        }


# class GroupSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Group
#         fields = ("name",)


class SocialSerializer(serializers.Serializer):
    """
    Serializer which accepts an OAuth2 access token and provider.
    """
    provider = serializers.CharField(max_length=255, required=True)
    access_token = serializers.CharField(max_length=4096, required=True, trim_whitespace=True)
