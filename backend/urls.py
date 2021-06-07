from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

from backend import settings

urlpatterns = [
    path(f'{settings.BASE_URL}/', include('backend_app.urls')),
    path('admin/', admin.site.urls),
    # For authentication
    path(f'{settings.BASE_URL}/auth/', include('auth_app.urls')),
    path(f'{settings.BASE_URL}/streamflow/', include('streamflow_app.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# urlpatterns = format_suffix_patterns(urlpatterns)
