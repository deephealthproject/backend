from django.urls import include, path

from backend_app.routers import HybridRouter
from streamflow_app import views

router = HybridRouter()

router.register(r'ssh-configs', views.SFSSHViewSet)
router.register(r'helm-configs', views.SFHelmViewSet)

urlpatterns = [
    path(f'', include(router.urls)),
]
