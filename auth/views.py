from rest_framework import mixins, status, views, viewsets, generics
from django.contrib.auth.models import User
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from rest_framework import permissions, views
from rest_framework.response import Response

from django.contrib.auth import get_user_model
from auth import serializers, swagger


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


class GetUserView(views.APIView):
    permission_classes = [permissions.IsAuthenticated]

    @swagger_auto_schema(responses={
        '200': openapi.Response('On a successful operation, it returns information about current user.',
                                serializers.UserSerializer,
                                examples={
                                    "application/json": {
                                        "username": "myuser",
                                        "email": "myemail",
                                        "first_name": "first_name",
                                        "last_name": "last_name"
                                    }
                                })
    })
    def get(self, request, *args, **kwargs):
        """Test the authentication returning information about the user

        This method returns some information about current user if the request include an authorization token.
        """
        return Response(serializers.UserSerializer(request.user).data)


class CreateUserView(generics.CreateAPIView):
    """Create a new user

    This method creates a new user with username e password provided.
    """
    model = User
    permission_classes = [
        permissions.AllowAny  # Otherwise anonymous are unable to register
    ]
    serializer_class = serializers.UserSerializer


class UsersViewSet(mixins.ListModelMixin,
                   mixins.RetrieveModelMixin,
                   viewsets.GenericViewSet):
    queryset = User.objects.all()
    serializer_class = serializers.UserSerializer

    @swagger_auto_schema(responses=swagger.users_list_response)
    def list(self, request, *args, **kwargs):
        """Get the list users

        This method returns all the registered users.
        """
        return super().list(request, *args, **kwargs)

    @swagger_auto_schema(responses=swagger.users_retrieve_response)
    def retrieve(self, request, *args, **kwargs):
        """Retrieve a single user

        This method returns the `{id}` user.
        """
        return super().retrieve(request, *args, **kwargs)


class ChangePasswordView(generics.UpdateAPIView):
    serializer_class = serializers.ChangePasswordSerializer
    model = get_user_model()
    permission_classes = (permissions.IsAuthenticated,)

    def get_object(self, queryset=None):
        return self.request.user

    @swagger_auto_schema(responses=swagger.change_password_response)
    def put(self, request, *args, **kwargs):
        """Change a user password

        This method change the password of the logged user.
        """
        return super().update(request, *args, **kwargs)