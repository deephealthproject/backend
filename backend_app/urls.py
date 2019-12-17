from django.urls import path, include
from rest_framework import routers
from backend_app import views
from backend_app.routers import HybridRouter

router = HybridRouter(trailing_slash=False)
# router = routers.DefaultRouter(trailing_slash=False)

router.get_api_root_view().cls.__name__ = "DeepHealth Backend"
router.get_api_root_view().cls.__doc__ = """
The structure of the backend can be viewed [here](https://drawsql.app/aimagelab/diagrams/api).
"""

router.register(r'allowedProperties', views.AllowedPropViewSet)
router.register(r'datasets', views.DatasetViewSet)
router.register(r'models', views.ModelViewSet)
router.register(r'projects', views.ProjectViewSet)
router.register(r'properties', views.PropertyViewSet)
router.register(r'tasks', views.TaskViewSet)
router.register(r'trainingSettings', views.TrainingSettingViewSet)
router.register(r'weights', views.ModelWeightsViewSet)

router.add_api_view('inference', path('inference', views.InferenceViewSet.as_view(), name='inference'))
router.add_api_view('train', path('train', views.TrainViewSet.as_view(), name='train'))

urlpatterns = [
    path('backend/', include(router.urls)),
]
