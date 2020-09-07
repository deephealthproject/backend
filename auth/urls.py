from django.contrib.auth import views as auth_views
from django.urls import include, path
from oauth2_provider.urls import management_urlpatterns

from auth import views

urlpatterns = [
    path('', include('rest_framework_social_oauth2.urls')),  # Social providers (GitHub)
    # path('', include('oauth2_provider.urls', namespace='oauth2_provider')),
    path('', include((management_urlpatterns, 'oauth2_provider'))),  # Internal Auth server
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('create/', views.CreateUserView.as_view(), name='create'),
    # Test views
    path('testUser/', views.TestAuthAPI.as_view(), name='testUser'),
]
