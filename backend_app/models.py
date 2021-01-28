import os
from os.path import join as opjoin

from django.conf import settings
from django.db import models
from django.db.models import Q
from django.db.models.constraints import UniqueConstraint
from django.db.models.signals import pre_delete
from django.dispatch import receiver


class Perm(models.TextChoices):
    OWNER = 'OWN', 'Can change and delete'  # Can change and delete. Full control
    VIEWER = 'VIEW', 'Can view'  # Can only view


def generate_file_path(filename, *args):
    return opjoin(*args, f'{filename}')


class ColorTypes(models.TextChoices):
    none = 'NONE', 'ecvl.ColorType.none'
    GRAY = 'GRAY', 'ecvl.ColorType.GRAY'
    RGB = 'RGB', 'ecvl.ColorType.RGB'
    RGBA = 'RGBA', 'ecvl.ColorType.RGBA'
    BGR = 'BGR', 'ecvl.ColorType.BGR'
    HSV = 'HSV', 'ecvl.ColorType.HSV'
    YCbCr = 'YCBCR', 'ecvl.ColorType.YCbCr'


class AllowedProperty(models.Model):
    """
    This model keeps track of default and allowed values of the models.

    It is useful to provide default and admissible values for empty projects.
    """
    property_id = models.ForeignKey('Property', on_delete=models.PROTECT)
    model_id = models.ForeignKey('Model', on_delete=models.CASCADE)
    dataset_id = models.ForeignKey('Dataset', on_delete=models.CASCADE, null=True, blank=True)

    allowed_value = models.CharField(max_length=200, null=True, blank=True)
    default_value = models.CharField(max_length=200, null=True, blank=True)

    class Meta:
        constraints = [
            UniqueConstraint(fields=['property_id', 'model_id', 'dataset_id'],
                             name='unique_with_optional'),
            UniqueConstraint(fields=['property_id', 'model_id'], condition=Q(dataset_id=None),
                             name='unique_without_optional'),
        ]
        ordering = ['model_id']
        verbose_name_plural = "Allowed properties"

    def __str__(self):
        return f'{self.model_id.name} - {self.property_id.name}'


class Dataset(models.Model):
    name = models.CharField(max_length=32)
    path = models.CharField(max_length=2048)

    is_single_image = models.BooleanField(default=False)
    task_id = models.ForeignKey('Task', on_delete=models.PROTECT)
    public = models.BooleanField(default=False)

    users = models.ManyToManyField(settings.AUTH_USER_MODEL, through='DatasetPermission')

    ctype = models.CharField(choices=ColorTypes.choices, max_length=5, default=ColorTypes.RGB)
    ctype_gt = models.CharField(choices=ColorTypes.choices, max_length=5, null=True, blank=True)

    class Meta:
        indexes = [models.Index(fields=['is_single_image'])]
        ordering = ['id']

    def __str__(self):
        return self.name


class DatasetPermission(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='permission')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    permission = models.CharField(choices=Perm.choices, max_length=4, default=Perm.OWNER)


class Inference(models.Model):
    modelweights_id = models.ForeignKey('ModelWeights', on_delete=models.CASCADE)
    dataset_id = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    stats = models.CharField(max_length=2048)

    # Celery generates a random uuid4
    celery_id = models.CharField(max_length=50, null=True, blank=True)
    # logfile = models.FilePathField(path=opjoin(settings.INFERENCE_DIR, 'logs'), null=True, blank=True)
    logfile = models.CharField(max_length=2048, null=True, blank=True)
    outputfile = models.CharField(max_length=2048, null=True, blank=True)

    project_id = models.ForeignKey('Project', on_delete=models.CASCADE)

    class Meta:
        indexes = [models.Index(fields=['celery_id'])]
        ordering = ['id']

    def __str__(self):
        return f'{self.modelweights_id.name} - {self.dataset_id.name}'


class Model(models.Model):
    name = models.CharField(max_length=32)
    location = models.CharField(max_length=2048)

    # A new model onnx is uploaded to backend async with celery.
    celery_id = models.CharField(max_length=50, null=True, blank=True)  # Used for downloading ONNX from url
    task_id = models.ForeignKey('Task', on_delete=models.PROTECT)

    class Meta:
        indexes = [models.Index(fields=['name'])]
        ordering = ['id']

    def __str__(self):
        return self.name


class ModelWeights(models.Model):
    location = models.CharField(max_length=2048)
    name = models.CharField(max_length=200)

    model_id = models.ForeignKey(Model, on_delete=models.CASCADE)
    dataset_id = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    pretrained_on = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True)
    public = models.BooleanField(default=False)

    users = models.ManyToManyField(settings.AUTH_USER_MODEL, through='ModelWeightsPermission')

    class Meta:
        ordering = ['id']
        verbose_name_plural = "Model Weights"

    def __str__(self):
        return self.name


class ModelWeightsPermission(models.Model):
    modelweight = models.ForeignKey(ModelWeights, on_delete=models.CASCADE, related_name='permission')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    permission = models.CharField(choices=Perm.choices, max_length=4, default=Perm.OWNER)


class Project(models.Model):
    name = models.CharField(max_length=32)
    task_id = models.ForeignKey('backend_app.Task', on_delete=models.PROTECT)

    users = models.ManyToManyField(settings.AUTH_USER_MODEL, through='ProjectPermission', related_name='projects')

    class Meta:
        indexes = [models.Index(fields=['name'])]
        ordering = ['id']

    def __str__(self):
        return self.name


class ProjectPermission(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='permission')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    permission = models.CharField(choices=Perm.choices, max_length=4, default=Perm.OWNER)


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


class Training(models.Model):
    # Celery generates a random uuid4
    celery_id = models.CharField(max_length=50, null=True, blank=True)
    logfile = models.CharField(max_length=2048, null=True, blank=True)

    modelweights_id = models.ForeignKey('ModelWeights', on_delete=models.CASCADE)
    project_id = models.ForeignKey('Project', on_delete=models.CASCADE)

    class Meta:
        indexes = [models.Index(fields=['celery_id'])]
        # ordering = ['id']
        verbose_name_plural = "Trainings"

    # def __str__(self):
    #     return self.modelweights_id.name


class TrainingSetting(models.Model):
    training_id = models.ForeignKey(Training, on_delete=models.CASCADE)
    property_id = models.ForeignKey(Property, on_delete=models.CASCADE)

    value = models.CharField(max_length=50)

    class Meta:
        ordering = ['id']
        unique_together = ["training_id", "property_id"]

    def __str__(self):
        return self.value


@receiver(pre_delete, sender=Model, dispatch_uid='model_delete_signal')
@receiver(pre_delete, sender=ModelWeights, dispatch_uid='modelweight_delete_signal')
@receiver(pre_delete, sender=Training, dispatch_uid='training_delete_signal')
@receiver(pre_delete, sender=Inference, dispatch_uid='inference_delete_signal')
def delete_files(sender, instance, **kwargs):
    fields_to_delete = ['logfile', 'outputfile', 'location']
    for f in fields_to_delete:
        if hasattr(instance, f):
            f = eval(f'instance.{f}')
            if os.path.isfile(f):
                os.remove(f)
