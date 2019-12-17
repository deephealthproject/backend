from rest_framework import mixins, status
from rest_framework.response import Response


class ParamListModelMixin(mixins.ListModelMixin):
    def check_params(self):
        for p in self.params:
            if not self.request.query_params.get(p):
                error = {"Error": f"Missing required parameter `{p}`"}
                return Response(data=error, status=status.HTTP_400_BAD_REQUEST)
        return None

    def list(self, request, *args, **kwargs):
        r = self.check_params()
        if r is not None:
            return r
        return super().list(request, *args, **kwargs)
