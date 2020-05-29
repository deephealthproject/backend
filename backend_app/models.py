import uuid
from django.db import models
from backend import settings
from os.path import join as opjoin


# def default_logfile_path(basedir, *args):
#     return opjoin(basedir, *args, f'{uuid.uuid4().hex}.log')


def generate_file_path(filename, *args):
    return opjoin(*args, f'{filename}')


class AllowedProperty(models.Model):
    """
    This model keeps track of default and allowed values of the models.

    It is useful to provide default and admissible values for empty projects.
    """
    property_id = models.ForeignKey('Property', on_delete=models.PROTECT)
    model_id = models.ForeignKey('Model', on_delete=models.CASCADE)

    allowed_value = models.CharField(max_length=200, null=True, blank=True)
    default_value = models.CharField(max_length=200, null=True, blank=True)

    class Meta:
        ordering = ['model_id']
        unique_together = ["model_id", "property_id"]
        verbose_name_plural = "Allowed properties"

    def __str__(self):
        return f'{self.model_id.name} - {self.property_id.name}'


class Dataset(models.Model):
    name = models.CharField(max_length=32)
    path = models.CharField(max_length=2048)

    is_single_image = models.BooleanField(default=False)
    task_id = models.ForeignKey('Task', on_delete=models.PROTECT)

    class Meta:
        indexes = [models.Index(fields=['is_single_image'])]
        ordering = ['id']

    def __str__(self):
        return self.name


class Inference(models.Model):
    modelweights_id = models.ForeignKey('ModelWeights', on_delete=models.CASCADE)
    dataset_id = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    stats = models.CharField(max_length=2048)

    # Celery generates a random uuid4
    celery_id = models.CharField(max_length=50, null=True, blank=True)
    # logfile = models.FilePathField(path=opjoin(settings.INFERENCE_DIR, 'logs'), null=True, blank=True)
    logfile = models.CharField(max_length=2048, null=True, blank=True)
    outputfile = models.CharField(max_length=2048, null=True, blank=True)

    class Meta:
        indexes = [models.Index(fields=['celery_id'])]
        # ordering = ['id']

    def __str__(self):
        return f'{self.modelweights_id.name} - {self.dataset_id.name}'


class Model(models.Model):
    name = models.CharField(max_length=32)
    location = models.CharField(max_length=2048)

    task_id = models.ForeignKey('Task', on_delete=models.PROTECT)

    class Meta:
        indexes = [models.Index(fields=['name'])]
        ordering = ['id']

    def __str__(self):
        return self.name


class ModelWeights(models.Model):
    location = models.CharField(max_length=2048)
    name = models.CharField(max_length=200)

    model_id = models.ForeignKey(Model, on_delete=models.PROTECT)
    # task_id = models.ForeignKey('Task', on_delete=models.PROTECT)
    dataset_id = models.ForeignKey(Dataset, on_delete=models.PROTECT)
    pretrained_on = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True)

    # Celery generates a random uuid4
    celery_id = models.CharField(max_length=50, null=True, blank=True)
    logfile = models.CharField(max_length=2048, null=True, blank=True)

    class Meta:
        indexes = [models.Index(fields=['celery_id'])]
        # ordering = ['id']
        verbose_name_plural = "Model weights"

    def __str__(self):
        return self.name


class Project(models.Model):
    name = models.CharField(max_length=32)

    task_id = models.ForeignKey('Task', on_delete=models.PROTECT)
    modelweights_id = models.ForeignKey(ModelWeights, on_delete=models.SET_NULL, null=True, blank=True)
    inference_id = models.ForeignKey(Inference, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        indexes = [models.Index(fields=['name'])]
        ordering = ['id']

    def __str__(self):
        return self.name


class Property(models.Model):
    name = models.CharField(max_length=50)
    type = models.CharField(max_length=50)
    default = models.CharField(max_length=50, null=True, blank=True)
    values = models.CharField(max_length=1000, null=True, blank=True)

    class Meta:
        indexes = [models.Index(fields=['name'])]
        ordering = ['id']
        verbose_name_plural = "Properties"

    def __str__(self):
        return self.name


class Task(models.Model):
    name = models.CharField(max_length=32)

    class Meta:
        indexes = [models.Index(fields=['name'])]
        ordering = ['id']

    def __str__(self):
        return self.name


class TrainingSetting(models.Model):
    modelweights_id = models.ForeignKey(ModelWeights, on_delete=models.CASCADE)
    property_id = models.ForeignKey(Property, on_delete=models.CASCADE)

    value = models.CharField(max_length=50)

    class Meta:
        ordering = ['id']
        unique_together = ["modelweights_id", "property_id"]

    def __str__(self):
        return self.value
