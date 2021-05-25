# Generated by Django 3.1.7 on 2021-05-25 17:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backend_app', '0039_add_timestamps_trainings_inferences'),
    ]

    operations = [
        migrations.AlterField(
            model_name='allowedproperty',
            name='allowed_value',
            field=models.TextField(blank=True, help_text='Comma-separated string of allowed values for this combination of property-model[-dataset]', null=True),
        ),
        migrations.AlterField(
            model_name='inference',
            name='created',
            field=models.DateTimeField(auto_now_add=True, help_text='Creation date of a training'),
        ),
        migrations.AlterField(
            model_name='inference',
            name='updated',
            field=models.DateTimeField(auto_now=True, help_text='Date and time of last modification'),
        ),
        migrations.AlterField(
            model_name='training',
            name='created',
            field=models.DateTimeField(auto_now_add=True, help_text='Creation date of a training'),
        ),
        migrations.AlterField(
            model_name='training',
            name='updated',
            field=models.DateTimeField(auto_now=True, help_text='Date and time of last modification'),
        ),
    ]
