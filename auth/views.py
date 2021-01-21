from django.contrib.auth import get_user_model
from django.contrib.auth.models import User
from django.core.mail import EmailMultiAlternatives
from django.dispatch import receiver
from django.template.loader import render_to_string
from django.urls import reverse
from django_rest_passwordreset.signals import reset_password_token_created
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from rest_framework import generics, mixins, permissions, status, views, viewsets
from rest_framework.response import Response

from auth import serializers, swagger
from backend import settings


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


@receiver(reset_password_token_created)
def password_reset_token_created(sender, instance, reset_password_token, *args, **kwargs):
    """
    Handles password reset tokens
    When a token is created, an e-mail needs to be sent to the user
    :param sender: View Class that sent the signal
    :param instance: View Instance that sent the signal
    :param reset_password_token: Token Model Object
    :param args:
    :param kwargs:
    :return:
    """
    # send an e-mail to the user
    context = {
        'current_user': reset_password_token.user,
        'username': reset_password_token.user.username,
        'email': reset_password_token.user.email,
        'token': reset_password_token.key,
        'reset_password_url': "{}?token={}".format(
            instance.request.build_absolute_uri(reverse('password_reset:reset-password-confirm')),
            reset_password_token.key),
        'reset_password_confirm_url': instance.request.build_absolute_uri('confirm'),
    }

    # render email text
    email_html_message = render_to_string('email/user_reset_password.html', context)
    email_plaintext_message = render_to_string('email/user_reset_password.txt', context)

    msg = EmailMultiAlternatives(
        # title:
        "Password Reset for {title}".format(title="DeepHealth Backend"),
        # message:
        email_plaintext_message,
        # from:
        settings.EMAIL_HOST,
        # to:
        [reset_password_token.user.email]
    )
    msg.attach_alternative(email_html_message, "text/html")
    msg.send()
