from rest_framework import mixins, status, viewsets
from rest_framework.response import Response

from streamflow_app import models, serializers


class SFSSHViewSet(mixins.ListModelMixin,
                   mixins.RetrieveModelMixin,
                   mixins.CreateModelMixin,
                   # mixins.DestroyModelMixin,
                   viewsets.GenericViewSet):
    queryset = models.SFSSH.objects.all()
    serializer_class = serializers.SFSSHSerializer

    def get_queryset(self):
        self.queryset = self.queryset.filter(user=self.request.user)
        return self.queryset

    # @swagger_auto_schema(
    #     manual_parameters=[openapi.Parameter('task_id', openapi.IN_QUERY, type=openapi.TYPE_INTEGER, required=False)]
    # )
    def list(self, request, *args, **kwargs):
        """
        """
        return super().list(request, *args, **kwargs)

    def retrieve(self, request, *args, **kwargs):
        """
        """
        return super().retrieve(request, *args, **kwargs)

    # @swagger_auto_schema(responses=swagger.DatasetViewSet_create_response)
    def create(self, request, *args, **kwargs):
        """
        """
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response({**{'error': 'Validation error. Request data is malformed.'}, **serializer.errors},
                            status=status.HTTP_400_BAD_REQUEST)

        serializer.save(user=request.user)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
