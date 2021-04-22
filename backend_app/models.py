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
    property_id = models.ForeignKey('Property', on_delete=models.PROTECT,
                                    help_text='The property for which specify a custom values')
    model_id = models.ForeignKey('Model', on_delete=models.CASCADE, help_text='Model linked to this custom values')
    dataset_id = models.ForeignKey('Dataset', on_delete=models.CASCADE, null=True, blank=True,
                                   help_text='Dataset linked to this custom values')

    allowed_value = models.TextField(null=True, blank=True,
                                     help_text='List or string of allowed values for this combination of '
                                               'property-model[-dataset]')
    default_value = models.TextField(null=True, blank=True,
                                     help_text='The default value for this combination of property-model[-dataset]')

    class Meta:
        constraints = [
            UniqueConstraint(fields=['property_id', 'model_id', 'dataset_id'],
                             name='unique_with_optional'),
            UniqueConstraint(fields=['property_id', 'model_id'], condition=Q(dataset_id=None),
                             name='unique_without_optional'),
        ]
        ordering = ['model_id', 'id']
        verbose_name_plural = "Allowed properties"

    def __str__(self):
        return f'{self.model_id.name} - {self.property_id.name}'


class Dataset(models.Model):
    name = models.CharField(max_length=32, help_text='The name of the dataset')
    path = models.CharField(max_length=2048, help_text='Absolute path to the YAML file of the dataset')
    classes = models.TextField(null=True, blank=True,
                               help_text='List of values (comma separated) which represents the classes of the dataset')

    is_single_image = models.BooleanField(default=False,
                                          help_text='Whether the dataset has been generated for InferenceSingle')
    public = models.BooleanField(default=False)
    ctype = models.CharField(choices=ColorTypes.choices, max_length=5, default=ColorTypes.RGB,
                             help_text='The ColorType of the images')
    ctype_gt = models.CharField(choices=ColorTypes.choices, max_length=5, null=True, blank=True,
                                help_text='The ColorType of the ground truth images')

    task_id = models.ForeignKey('Task', on_delete=models.PROTECT, help_text='The task which the dataset refers to')
    users = models.ManyToManyField(settings.AUTH_USER_MODEL, through='DatasetPermission',
                                   help_text='List of users who have rights to access this dataset')

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
    modelweights_id = models.ForeignKey('ModelWeights', on_delete=models.CASCADE,
                                        help_text='Weight used for the inference')
    dataset_id = models.ForeignKey(Dataset, on_delete=models.CASCADE, help_text='Dataset used for the inference')
    stats = models.CharField(max_length=2048, help_text='File used for store statistics')

    # Celery generates a random uuid4
    celery_id = models.CharField(max_length=50, null=True, blank=True, help_text='UUID4 indicating the celery task')
    # logfile = models.FilePathField(path=opjoin(settings.INFERENCE_DIR, 'logs'), null=True, blank=True)
    logfile = models.CharField(max_length=2048, null=True, blank=True,
                               help_text='Absolute path to the file which stores the log of the inference process')
    outputfile = models.CharField(max_length=2048, null=True, blank=True,
                                  help_text='Absolute path of the file which store the prediction of the inference')

    project_id = models.ForeignKey('Project', on_delete=models.CASCADE,
                                   help_text='Project on which this inference has been lanched')

    class Meta:
        indexes = [models.Index(fields=['celery_id'])]
        ordering = ['id']

    def __str__(self):
        return f'{self.modelweights_id.name} - {self.dataset_id.name}'


class Model(models.Model):
    name = models.CharField(max_length=255, help_text='Name of a neural network model family')
    task_id = models.ForeignKey('Task', on_delete=models.PROTECT, help_text='Task which the model refers to')

    class Meta:
        indexes = [models.Index(fields=['name'])]
        ordering = ['id']

    def __str__(self):
        return self.name


class ModelWeights(models.Model):
    location = models.CharField(max_length=2048, help_text='Absolute path of the ONNX weight')
    name = models.CharField(max_length=200, help_text='The name of the model weight')
    layer_to_remove = models.CharField(max_length=200, null=True, blank=True,
                                       help_text='Name of the ONNX layer which will be removed when finetuning')
    is_active = models.BooleanField('active', default=False,
                                    help_text='Boolean which tells if the weight is ready to be used or not')
    classes = models.TextField(null=True, blank=True,
                               help_text='List of classes of the dataset from which the weight has been trained')

    model_id = models.ForeignKey('Model', on_delete=models.CASCADE,
                                 help_text='The model family which this weight belong to')
    dataset_id = models.ForeignKey('Dataset', on_delete=models.CASCADE, null=True, blank=True,
                                   help_text='Dataset on which this has weight has been trained on')
    pretrained_on = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True,
                                      help_text='The parent weight used as pretraining')
    public = models.BooleanField(default=False, help_text='Wheter the weight is public or private')

    users = models.ManyToManyField(settings.AUTH_USER_MODEL, through='ModelWeightsPermission',
                                   help_text='List of users who have rights to access this weight')

    # A new model onnx is uploaded to backend async with celery.
    process_id = models.CharField(max_length=50, null=True, blank=True,
                                  help_text='Optional UUID4 used for storing the celery task id when the'
                                            ' weight is downloaded from url')

    class Meta:
        ordering = ('model_id', 'id',)
        verbose_name_plural = "Model Weights"

    def __str__(self):
        return self.name


class ModelWeightsPermission(models.Model):
    modelweight = models.ForeignKey(ModelWeights, on_delete=models.CASCADE, related_name='permission',
                                    help_text='Weight for which specify a permission')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             help_text='User which grants a permission')
    permission = models.CharField(choices=Perm.choices, max_length=4, default=Perm.OWNER,
                                  help_text='Kind of permission')


class Project(models.Model):
    name = models.CharField(max_length=255, help_text='Name of a user project')
    task_id = models.ForeignKey('backend_app.Task', on_delete=models.PROTECT,
                                help_text='The project focuses on a fixed task')

    users = models.ManyToManyField(settings.AUTH_USER_MODEL, through='ProjectPermission', related_name='projects',
                                   help_text='Users who can manage and interact within the project')

    class Meta:
        indexes = [models.Index(fields=['name'])]
        ordering = ['id']

    def __str__(self):
        return self.name


class ProjectPermission(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='permission')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    permission = models.CharField(choices=Perm.choices, max_length=4, default=Perm.OWNER)


class PropertyType(models.TextChoices):
    int = 'INT', 'Integer'
    float = 'FLT', 'Float'
    list = 'LST', 'List'
    string = 'STR', 'String'


class Property(models.Model):
    name = models.CharField(max_length=50, help_text='Name of a property')
    type = models.CharField(choices=PropertyType.choices, max_length=3, help_text='Type of the property')
    default = models.TextField(null=True, blank=True, help_text='The default value of the property')
    values = models.TextField(null=True, blank=True, help_text='Values which the property could get')

    class Meta:
        indexes = [models.Index(fields=['name'])]
        ordering = ['id']
        verbose_name_plural = "Properties"

    def __str__(self):
        return self.name


class Task(models.Model):
    name = models.CharField(max_length=255, help_text='The name of a task (e.g. classification)')

    class Meta:
        indexes = [models.Index(fields=['name'])]
        ordering = ['id']

    def __str__(self):
        return self.name


class Training(models.Model):
    logfile = models.CharField(max_length=2048, null=True, blank=True,
                               help_text='The absolute path to the training log file')
    # Celery generates a random uuid4
    celery_id = models.CharField(max_length=50, null=True, blank=True, help_text='Celery UUID4 task process id')

    modelweights_id = models.ForeignKey('ModelWeights', on_delete=models.CASCADE,
                                        help_text='Weight produced in this learning process')
    project_id = models.ForeignKey('Project', on_delete=models.CASCADE,
                                   help_text='Project in which this training has been launched')

    class Meta:
        indexes = [models.Index(fields=['celery_id'])]
        # ordering = ['id']
        verbose_name_plural = "Trainings"

    # def __str__(self):
    #     return self.modelweights_id.name


class TrainingSetting(models.Model):
    training_id = models.ForeignKey(Training, on_delete=models.CASCADE,
                                    help_text='Training process for which stores settings')
    property_id = models.ForeignKey(Property, on_delete=models.CASCADE,
                                    help_text='The property of whom store the value')
    value = models.TextField(help_text='The value used for the training')

    class Meta:
        ordering = ['id']
        unique_together = ["training_id", "property_id"]

    def __str__(self):
        return self.value


@receiver(pre_delete, sender=Model, dispatch_uid='model_delete_signal')
@receiver(pre_delete, sender=ModelWeights, dispatch_uid='modelweight_delete_signal')
@receiver(pre_delete, sender=Training, dispatch_uid='training_delete_signal')
@receiver(pre_delete, sender=Inference, dispatch_uid='inference_delete_signal')
@receiver(pre_delete, sender=Dataset, dispatch_uid='dataset_delete_signal')
def delete_files(sender, instance, **kwargs):
    fields_to_delete = ['logfile', 'outputfile', 'location', 'path']
    for f in fields_to_delete:
        if hasattr(instance, f):
            f = eval(f'instance.{f}')
            if os.path.isfile(f):
                os.remove(f)
