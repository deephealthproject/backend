from django.contrib.auth.models import User
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from rest_framework import permissions
from rest_framework import views
from rest_framework.response import Response

from auth import serializers


# Test API
class TestAuthAPI(views.APIView):
    permission_classes = [permissions.IsAuthenticated]

    @swagger_auto_schema(responses={
        '200': openapi.Response('On a successful operation, it returns instance of dhtest user.',
                                serializers.UserSerializer,
                                examples={
                                    "application/json": {
                                        "username": "dhtest",
                                        "email": "",
                                        "first_name": "",
                                        "last_name": ""
                                    }
                                })
    })
    def get(self, request, *args, **kwargs):
        """Test the authentication returning information about dhtest user

        This method returns some information about dhtest user if the request include an authorization token.
        """
        u = User.objects.get(username='dhtest')
        return Response(serializers.UserSerializer(u).data)
