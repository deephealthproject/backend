# migration 03
from django.db import models


class Task(models.Model):
    name = models.CharField(max_length=32, null=False)

    class Meta:
        indexes = [models.Index(fields=['name'])]
        ordering = ['id']

    def __str__(self):
        return self.name


class Dataset(models.Model):
    name = models.CharField(max_length=32)
    path = models.CharField(max_length=2048)
    ispretraining = models.BooleanField()

    task_id = models.ForeignKey(Task, on_delete=models.PROTECT)

    class Meta:
        indexes = [models.Index(fields=['name'])]
        ordering = ['id']

    def __str__(self):
        return self.name


class Model(models.Model):
    name = models.CharField(max_length=32)
    location = models.CharField(max_length=2048)

    task_id = models.ForeignKey(Task, on_delete=models.PROTECT)

    class Meta:
        indexes = [models.Index(fields=['name'])]
        ordering = ['id']

    def __str__(self):
        return self.name


class ModelWeights(models.Model):
    location = models.CharField(max_length=2048)

    model_id = models.ForeignKey(Model, on_delete=models.PROTECT)
    task_id = models.ForeignKey(Task, on_delete=models.PROTECT)
    pretraining_id = models.ForeignKey(Dataset, related_name='pretraining_id', on_delete=models.PROTECT)
    finetuning_id = models.ForeignKey(Dataset, related_name='finetuning_id', on_delete=models.PROTECT, null=True,
                                      blank=True)

    class Meta:
        ordering = ['id']
        verbose_name_plural = "Model weights"

    def __str__(self):
        # TODO maybe a name field?
        return self.location


class Project(models.Model):
    name = models.CharField(max_length=32)

    task_id = models.ForeignKey(Task, on_delete=models.PROTECT)
    modelweights_id = models.ForeignKey(ModelWeights, on_delete=models.PROTECT, null=True, blank=True)

    class Meta:
        indexes = [models.Index(fields=['name'])]
        ordering = ['id']

    def __str__(self):
        return self.name


class Property(models.Model):
    name = models.CharField(max_length=50)
    type = models.CharField(max_length=50)
    default = models.CharField(max_length=50)
    values = models.CharField(max_length=1000, null=True, blank=True)

    class Meta:
        indexes = [models.Index(fields=['name'])]
        ordering = ['id']
        verbose_name_plural = "Properties"

    def __str__(self):
        return self.name


class PropertyInstance(models.Model):
    property_id = models.ForeignKey(Property, on_delete=models.PROTECT)
    modelweights_id = models.ForeignKey(ModelWeights, on_delete=models.PROTECT)

    value = models.CharField(max_length=50)

    class Meta:
        ordering = ['id']

    def __str__(self):
        # TODO maybe a name field?
        return self.value
