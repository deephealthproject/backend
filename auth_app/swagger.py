from drf_yasg import openapi
from drf_yasg.views import get_schema_view
from rest_framework import permissions

from auth_app import serializers

users_list_response = {
    '200': openapi.Response('On a successful operation, it returns the list of users',
                            serializers.UserSerializer(),
                            examples={
                                "application/json":
                                    [
                                        {
                                            "id": 1,
                                            "username": "admin",
                                            "first_name": "",
                                            "last_name": ""
                                        },
                                        {
                                            "id": 2,
                                            "username": "user",
                                            "first_name": "",
                                            "last_name": ""
                                        }
                                    ]
                            }),
}

users_retrieve_response = {
    '200': openapi.Response('On a successful operation, it returns the information about the `id` user',
                            serializers.UserSerializer(),
                            examples={
                                "application/json":
                                    {
                                        "id": 3,
                                        "username": "user",
                                        "first_name": "",
                                        "last_name": ""
                                    }
                            }),
}

change_password_response = {
    '200': openapi.Response('On a successful operation, it returns an empty response.',
                            examples={"application/json": {}}),
}
