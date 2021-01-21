"""
Django settings for backend project.
"""

import json
import os

import environ
from django.core.exceptions import ImproperlyConfigured

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default config values
env = environ.Env(
    DEBUG=(bool, False),
    ALLOWED_HOSTS=(list, ['localhost', '127.0.0.1']),
    CORS_ORIGIN_WHITELIST=(list, []),
    DATA_DIR=(environ.Path, os.path.join(BASE_DIR, 'data')),
    CELERY_ACCEPT_CONTENT=(list, ['json']),
    CELERY_RESULT_BACKEND=(str, 'db+sqlite:///results.sqlite'),
    CELERY_TASK_SERIALIZER=(str, 'json'),
    EDDL_WITH_CUDA=(bool, False),
)

env_path = os.environ.get("ENV_PATH", os.path.join(BASE_DIR, "config"))
env.read_env(env_path)  # reading .env file

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = env('SECRET_KEY')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = env('DEBUG')

# List of allowed hosts
ALLOWED_HOSTS = env.list("ALLOWED_HOSTS")

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'backend_app.apps.BackendAppConfig',
    'rest_framework',
    'corsheaders',
    'drf_yasg',  # Swagger auto-docs
    # OAuth2
    'oauth2_provider',
    'social_django',
    'rest_framework_social_oauth2',
    'django_rest_passwordreset',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'social_django.middleware.SocialAuthExceptionMiddleware',
]

ROOT_URLCONF = 'backend.urls'

SWAGGER_SETTINGS = {
    'DEEP_LINKING': True,  # Add links to paragraphs
    'DEFAULT_MODEL_RENDERING': 'example',
}

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'oauth2_provider.contrib.rest_framework.OAuth2Authentication',
        'rest_framework_social_oauth2.authentication.SocialAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    )
}

OAUTH2_PROVIDER = {
    # this is the list of available scopes
    'SCOPES': {'read': 'Read scope', 'write': 'Write scope'}
}

AUTHENTICATION_BACKENDS = (
    # GitHub OAuth2
    'social_core.backends.github.GithubOAuth2',
    'rest_framework_social_oauth2.backends.DjangoOAuth2',
    'django.contrib.auth.backends.ModelBackend',
)

SOCIAL_AUTH_PIPELINE = (
    'social_core.pipeline.social_auth.social_details',
    'social_core.pipeline.social_auth.social_uid',
    'social_core.pipeline.social_auth.auth_allowed',
    'social_core.pipeline.social_auth.social_user',

    # Make up a username for this person, appends a random string at the end if
    # there's any collision.
    'social_core.pipeline.user.get_username',

    # CUSTOM: this gets email address as the username and validates it matches
    # the logged in user's email address.
    # 'repairs_accounts.pipeline.get_username',

    'social_core.pipeline.social_auth.associate_by_email',
    'social_core.pipeline.user.create_user',
    'social_core.pipeline.social_auth.associate_user',
    'social_core.pipeline.social_auth.load_extra_data',
    'social_core.pipeline.user.user_details'
)

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                # OAuth2
                'social_django.context_processors.backends',
                'social_django.context_processors.login_redirect',
            ],
        },
    },
]

WSGI_APPLICATION = 'backend.wsgi.application'

# Database
DATABASES = {
    # read os.environ['DATABASE_URL'] and raises ImproperlyConfigured exception if not found
    'default': env.db()
}

# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.mysql',
#         'NAME': 'backend',
#         'USER': secrets.get_secret('DB_USER'),
#         'PASSWORD': secrets.get_secret('DB_PASSWORD'),
#         'HOST': '127.0.0.1',
#         'PORT': '3306',
#     }
# }

# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
     'OPTIONS': {
         'min_length': 8,
     }},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True
APPEND_SLASH = False

# Set data paths
DATA_DIR = env('DATA_DIR')
TRAINING_DIR = os.path.join(DATA_DIR, 'training')
INFERENCE_DIR = os.path.join(DATA_DIR, 'inference')
DATASETS_DIR = os.path.join(DATA_DIR, 'datasets')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
OUTPUTS_DIR = os.path.join(INFERENCE_DIR, 'outputs')

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.2/howto/static-files/
STATIC_ROOT = os.path.join(BASE_DIR, 'static')
STATIC_URL = env('STATIC_URL')
# Base url to serve media files
MEDIA_URL = env('MEDIA_URL')
# Path where media is stored
MEDIA_ROOT = OUTPUTS_DIR

BASE_URL = 'backend'
LOGIN_REDIRECT_URL = f'/{BASE_URL}/'
LOGIN_URL = f'/{BASE_URL}/auth/login/'

# Cross-Origin Resource Sharing (CORS) whitelist
CORS_ORIGIN_WHITELIST = env.list('CORS_ORIGIN_WHITELIST')

# CELERY_BROKER_URL = 'amqp://guest:guest@localhost'
CELERY_BROKER_URL = env('RABBITMQ_BROKER_URL')

# Only add pickle to this list if your broker is secured
# from unwanted access (see userguide/security.html)
accept_content = env.list('CELERY_ACCEPT_CONTENT')
result_backend = env('CELERY_RESULT_BACKEND')
task_serializer = env('CELERY_TASK_SERIALIZER')

SOCIAL_AUTH_USER_FIELDS = ['email', 'username']
SOCIAL_AUTH_GITHUB_KEY = env('SOCIAL_AUTH_GITHUB_KEY')
SOCIAL_AUTH_GITHUB_SECRET = env('SOCIAL_AUTH_GITHUB_SECRET')
SOCIAL_AUTH_GITHUB_SCOPE = ['user:email']

EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_USE_TLS = True
EMAIL_PORT = 587
EMAIL_HOST_USER = env('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = env('EMAIL_HOST_PASSWORD')
