import os

import celery

# set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

app = celery.Celery('backend')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print('Request: {0!r}'.format(self.request))


class CeleryTask(celery.Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        pass

    # def on_success(self, retval, task_id, args, kwargs):
    #     pass
    # def run(self, *args, **kwargs):
    #     pass
