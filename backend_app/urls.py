from django.urls import path, include
from rest_framework import routers
from backend_app import views
from backend_app.routers import HybridRouter

router = HybridRouter()
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
router.add_api_view('inferenceSingle',
                    path('inferenceSingle', views.InferenceSingleViewSet.as_view(), name='inferenceSingle'))
router.add_api_view('status', path('status', views.StatusView.as_view(), name='status'))
router.add_api_view('stopProcess', path('stopProcess', views.StopProcessViewSet.as_view(), name='stopProcess'))
router.add_api_view('train', path('train', views.TrainViewSet.as_view(), name='train'))
router.add_api_view('output', path('output', views.OutputViewSet.as_view(), name='output'))

urlpatterns = [
    path('backend/', include(router.urls)),
]
