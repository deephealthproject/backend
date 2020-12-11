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
        '200': openapi.Response('On a successful operation, it returns 200 OK response.',
                                examples={"application/json": {'result': 'ok'}})
    })
    def get(self, request, *args, **kwargs):
        """Test the authentication returning HTTP_200_OK

        This method returns HTTP_200_OK  if the request include an authorization token.
        """
        return Response({'result': 'ok'}, status=status.HTTP_200_OK)


# class GetUserView(views.APIView):
#     permission_classes = [permissions.IsAuthenticated]
#
#     @swagger_auto_schema(responses={
#         '200': openapi.Response('On a successful operation, it returns information about current user.',
#                                 serializers.UserSerializer,
#                                 examples={
#                                     "application/json": {
#                                         "username": "myuser",
#                                         "email": "myemail",
#                                         "first_name": "first_name",
#                                         "last_name": "last_name"
#                                     }
#                                 })
#     })
#     def get(self, request, *args, **kwargs):
#         """Test the authentication returning information about the user
#
#         This method returns some information about current user if the request include an authorization token.
#         """
#         return Response(serializers.UserSerializer(request.user).data)


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


class ChangePasswordView(mixins.UpdateModelMixin,
                         generics.GenericAPIView):
    model = get_user_model()
    permission_classes = (permissions.IsAuthenticated,)
    serializer_class = serializers.ChangePasswordSerializer

    def get_object(self, queryset=None):
        return self.request.user

    @swagger_auto_schema(responses=swagger.change_password_response)
    def put(self, request, *args, **kwargs):
        """Change a user password

        This method change the password of the logged user.
        """
        return super().update(request, *args, **kwargs)


class UserView(generics.CreateAPIView,
               generics.RetrieveUpdateAPIView,
               generics.DestroyAPIView
               ):
    model = get_user_model()
    permission_classes = (permissions.IsAuthenticated,)
    serializer_class = serializers.UserSerializer

    def get_permissions(self):
        if self.request.method == 'POST':
            return (permissions.AllowAny(),)  # Allow non registered users
        return (permissions.IsAuthenticated(),)

    def get_serializer_class(self):
        if self.request.method == 'PUT':
            return serializers.UserUpdateSerializer
        return serializers.UserSerializer

    def get_object(self):
        return self.request.user

    def post(self, request, *args, **kwargs):
        """Create a new user

        This method creates a new user with username e password provided.
        """
        return self.create(request, *args, **kwargs)

    def get(self, request, *args, **kwargs):
        """Retrieve the current user information

        This method retrieves all user information
        """
        return super().get(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        """Delete a user

        This method deletes a user and all of his data.
        Projects, Trainings, and Inferences will be deleted too.
        """
        return super().destroy(request, *args, **kwargs)

    def put(self, request, *args, **kwargs):
        """Update user data

        This method updates the user information.
        """
        return super().put(request, *args, **kwargs)
