import os
from os.path import join as opjoin
from django.apps import AppConfig
from backend import settings


class BackendAppConfig(AppConfig):
    name = 'backend_app'

    def ready(self):
        # Create directories
        os.makedirs(opjoin(settings.TRAINING_DIR, 'ckpts'), exist_ok=True)
        os.makedirs(opjoin(settings.TRAINING_DIR, 'logs'), exist_ok=True)
        os.makedirs(opjoin(settings.INFERENCE_DIR, 'ckpts'), exist_ok=True)
        os.makedirs(opjoin(settings.INFERENCE_DIR, 'logs'), exist_ok=True)

        os.makedirs(settings.DATASETS_DIR, exist_ok=True)
        os.makedirs(settings.OUTPUTS_DIR, exist_ok=True)
