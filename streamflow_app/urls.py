from django.urls import include, path, re_path

from streamflow_app import swagger, views
from backend_app.routers import HybridRouter

router = HybridRouter()

router.register(r'ssh-configs', views.SFSSHViewSet)


urlpatterns = [
    path(f'', include(router.urls)),
]
