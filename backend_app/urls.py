from django.urls import include, path, re_path

# from rest_framework import routers
from backend_app import swagger, views
from backend_app.routers import HybridRouter

router = HybridRouter()

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
    path(f'', include(router.urls)),
    re_path(fr'^swagger(?P<format>\.json|\.yaml)$', swagger.schema_view.without_ui(cache_timeout=0),
            name='schema-json'),
    re_path(fr'^swagger$', swagger.schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    re_path(fr'^redoc$', swagger.schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]
