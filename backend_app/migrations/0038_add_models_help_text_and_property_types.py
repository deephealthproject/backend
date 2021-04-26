# Generated by Django 3.1.7 on 2021-04-22 16:10

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('backend_app', '0037_add_classes_weights'),
    ]

    operations = [
        migrations.AlterField(
            model_name='allowedproperty',
            name='allowed_value',
            field=models.TextField(blank=True, help_text='List or string of allowed values for this combination of property-model[-dataset]', null=True),
        ),
        migrations.AlterField(
            model_name='allowedproperty',
            name='dataset_id',
            field=models.ForeignKey(blank=True, help_text='Dataset linked to this custom values', null=True, on_delete=django.db.models.deletion.CASCADE, to='backend_app.dataset'),
        ),
        migrations.AlterField(
            model_name='allowedproperty',
            name='default_value',
            field=models.TextField(blank=True, help_text='The default value for this combination of property-model[-dataset]', null=True),
        ),
        migrations.AlterField(
            model_name='allowedproperty',
            name='model_id',
            field=models.ForeignKey(help_text='Model linked to this custom values', on_delete=django.db.models.deletion.CASCADE, to='backend_app.model'),
        ),
        migrations.AlterField(
            model_name='allowedproperty',
            name='property_id',
            field=models.ForeignKey(help_text='The property for which specify a custom values', on_delete=django.db.models.deletion.PROTECT, to='backend_app.property'),
        ),
        migrations.AlterField(
            model_name='dataset',
            name='classes',
            field=models.TextField(blank=True, help_text='List of values (comma separated) which represents the classes of the dataset', null=True),
        ),
        migrations.AlterField(
            model_name='dataset',
            name='ctype',
            field=models.CharField(choices=[('NONE', 'ecvl.ColorType.none'), ('GRAY', 'ecvl.ColorType.GRAY'), ('RGB', 'ecvl.ColorType.RGB'), ('RGBA', 'ecvl.ColorType.RGBA'), ('BGR', 'ecvl.ColorType.BGR'), ('HSV', 'ecvl.ColorType.HSV'), ('YCBCR', 'ecvl.ColorType.YCbCr')], default='RGB', help_text='The ColorType of the images', max_length=5),
        ),
        migrations.AlterField(
            model_name='dataset',
            name='ctype_gt',
            field=models.CharField(blank=True, choices=[('NONE', 'ecvl.ColorType.none'), ('GRAY', 'ecvl.ColorType.GRAY'), ('RGB', 'ecvl.ColorType.RGB'), ('RGBA', 'ecvl.ColorType.RGBA'), ('BGR', 'ecvl.ColorType.BGR'), ('HSV', 'ecvl.ColorType.HSV'), ('YCBCR', 'ecvl.ColorType.YCbCr')], help_text='The ColorType of the ground truth images', max_length=5, null=True),
        ),
        migrations.AlterField(
            model_name='dataset',
            name='is_single_image',
            field=models.BooleanField(default=False, help_text='Whether the dataset has been generated for InferenceSingle'),
        ),
        migrations.AlterField(
            model_name='dataset',
            name='name',
            field=models.CharField(help_text='The name of the dataset', max_length=32),
        ),
        migrations.AlterField(
            model_name='dataset',
            name='path',
            field=models.CharField(help_text='Absolute path to the YAML file of the dataset', max_length=2048),
        ),
        migrations.AlterField(
            model_name='dataset',
            name='task_id',
            field=models.ForeignKey(help_text='The task which the dataset refers to', on_delete=django.db.models.deletion.PROTECT, to='backend_app.task'),
        ),
        migrations.AlterField(
            model_name='dataset',
            name='users',
            field=models.ManyToManyField(help_text='List of users who have rights to access this dataset', through='backend_app.DatasetPermission', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='inference',
            name='celery_id',
            field=models.CharField(blank=True, help_text='UUID4 indicating the celery task', max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='inference',
            name='dataset_id',
            field=models.ForeignKey(help_text='Dataset used for the inference', on_delete=django.db.models.deletion.CASCADE, to='backend_app.dataset'),
        ),
        migrations.AlterField(
            model_name='inference',
            name='logfile',
            field=models.CharField(blank=True, help_text='Absolute path to the file which stores the log of the inference process', max_length=2048, null=True),
        ),
        migrations.AlterField(
            model_name='inference',
            name='modelweights_id',
            field=models.ForeignKey(help_text='Weight used for the inference', on_delete=django.db.models.deletion.CASCADE, to='backend_app.modelweights'),
        ),
        migrations.AlterField(
            model_name='inference',
            name='outputfile',
            field=models.CharField(blank=True, help_text='Absolute path of the file which store the prediction of the inference', max_length=2048, null=True),
        ),
        migrations.AlterField(
            model_name='inference',
            name='project_id',
            field=models.ForeignKey(help_text='Project on which this inference has been lanched', on_delete=django.db.models.deletion.CASCADE, to='backend_app.project'),
        ),
        migrations.AlterField(
            model_name='inference',
            name='stats',
            field=models.CharField(help_text='File used for store statistics', max_length=2048),
        ),
        migrations.AlterField(
            model_name='model',
            name='name',
            field=models.CharField(help_text='Name of a neural network model family', max_length=255),
        ),
        migrations.AlterField(
            model_name='model',
            name='task_id',
            field=models.ForeignKey(help_text='Task which the model refers to', on_delete=django.db.models.deletion.PROTECT, to='backend_app.task'),
        ),
        migrations.AlterField(
            model_name='modelweights',
            name='classes',
            field=models.TextField(blank=True, help_text='List of classes of the dataset from which the weight has been trained', null=True),
        ),
        migrations.AlterField(
            model_name='modelweights',
            name='dataset_id',
            field=models.ForeignKey(blank=True, help_text='Dataset on which this has weight has been trained on', null=True, on_delete=django.db.models.deletion.CASCADE, to='backend_app.dataset'),
        ),
        migrations.AlterField(
            model_name='modelweights',
            name='is_active',
            field=models.BooleanField(default=False, help_text='Boolean which tells if the weight is ready to be used or not', verbose_name='active'),
        ),
        migrations.AlterField(
            model_name='modelweights',
            name='layer_to_remove',
            field=models.CharField(blank=True, help_text='Name of the ONNX layer which will be removed when finetuning', max_length=200, null=True),
        ),
        migrations.AlterField(
            model_name='modelweights',
            name='location',
            field=models.CharField(help_text='Absolute path of the ONNX weight', max_length=2048),
        ),
        migrations.AlterField(
            model_name='modelweights',
            name='model_id',
            field=models.ForeignKey(help_text='The model family which this weight belong to', on_delete=django.db.models.deletion.CASCADE, to='backend_app.model'),
        ),
        migrations.AlterField(
            model_name='modelweights',
            name='name',
            field=models.CharField(help_text='The name of the model weight', max_length=200),
        ),
        migrations.AlterField(
            model_name='modelweights',
            name='pretrained_on',
            field=models.ForeignKey(blank=True, help_text='The parent weight used as pretraining', null=True, on_delete=django.db.models.deletion.CASCADE, to='backend_app.modelweights'),
        ),
        migrations.AlterField(
            model_name='modelweights',
            name='process_id',
            field=models.CharField(blank=True, help_text='Optional UUID4 used for storing the celery task id when the weight is downloaded from url', max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='modelweights',
            name='public',
            field=models.BooleanField(default=False, help_text='Wheter the weight is public or private'),
        ),
        migrations.AlterField(
            model_name='modelweights',
            name='users',
            field=models.ManyToManyField(help_text='List of users who have rights to access this weight', through='backend_app.ModelWeightsPermission', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='modelweightspermission',
            name='modelweight',
            field=models.ForeignKey(help_text='Weight for which specify a permission', on_delete=django.db.models.deletion.CASCADE, related_name='permission', to='backend_app.modelweights'),
        ),
        migrations.AlterField(
            model_name='modelweightspermission',
            name='permission',
            field=models.CharField(choices=[('OWN', 'Can change and delete'), ('VIEW', 'Can view')], default='OWN', help_text='Kind of permission', max_length=4),
        ),
        migrations.AlterField(
            model_name='modelweightspermission',
            name='user',
            field=models.ForeignKey(help_text='User which grants a permission', on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='project',
            name='name',
            field=models.CharField(help_text='Name of a user project', max_length=255),
        ),
        migrations.AlterField(
            model_name='project',
            name='task_id',
            field=models.ForeignKey(help_text='The project focuses on a fixed task', on_delete=django.db.models.deletion.PROTECT, to='backend_app.task'),
        ),
        migrations.AlterField(
            model_name='project',
            name='users',
            field=models.ManyToManyField(help_text='Users who can manage and interact within the project', related_name='projects', through='backend_app.ProjectPermission', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='property',
            name='default',
            field=models.TextField(blank=True, help_text='The default value of the property', null=True),
        ),
        migrations.AlterField(
            model_name='property',
            name='name',
            field=models.CharField(help_text='Name of a property', max_length=50),
        ),
        migrations.AlterField(
            model_name='property',
            name='type',
            field=models.CharField(choices=[('INT', 'Integer'), ('FLT', 'Float'), ('LST', 'List'), ('STR', 'String')], help_text='Type of the property', max_length=3),
        ),
        migrations.AlterField(
            model_name='property',
            name='values',
            field=models.TextField(blank=True, help_text='Values which the property could get', null=True),
        ),
        migrations.AlterField(
            model_name='task',
            name='name',
            field=models.CharField(help_text='The name of a task (e.g. classification)', max_length=255),
        ),
        migrations.AlterField(
            model_name='training',
            name='celery_id',
            field=models.CharField(blank=True, help_text='Celery UUID4 task process id', max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='training',
            name='logfile',
            field=models.CharField(blank=True, help_text='The absolute path to the training log file', max_length=2048, null=True),
        ),
        migrations.AlterField(
            model_name='training',
            name='modelweights_id',
            field=models.ForeignKey(help_text='Weight produced in this learning process', on_delete=django.db.models.deletion.CASCADE, to='backend_app.modelweights'),
        ),
        migrations.AlterField(
            model_name='training',
            name='project_id',
            field=models.ForeignKey(help_text='Project in which this training has been launched', on_delete=django.db.models.deletion.CASCADE, to='backend_app.project'),
        ),
        migrations.AlterField(
            model_name='trainingsetting',
            name='property_id',
            field=models.ForeignKey(help_text='The property of whom store the value', on_delete=django.db.models.deletion.CASCADE, to='backend_app.property'),
        ),
        migrations.AlterField(
            model_name='trainingsetting',
            name='training_id',
            field=models.ForeignKey(help_text='Training process for which stores settings', on_delete=django.db.models.deletion.CASCADE, to='backend_app.training'),
        ),
        migrations.AlterField(
            model_name='trainingsetting',
            name='value',
            field=models.TextField(help_text='The value used for the training'),
        ),
    ]