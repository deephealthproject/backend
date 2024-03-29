# Generated by Django 3.1.7 on 2021-04-13 15:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backend_app', '0033_add_layer_to_remove_weight'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='allowedproperty',
            options={'ordering': ['model_id', 'id'], 'verbose_name_plural': 'Allowed properties'},
        ),
        migrations.AlterModelOptions(
            name='modelweights',
            options={'ordering': ('model_id', 'id'), 'verbose_name_plural': 'Model Weights'},
        ),
        migrations.AlterField(
            model_name='allowedproperty',
            name='allowed_value',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='allowedproperty',
            name='default_value',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='property',
            name='default',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='property',
            name='values',
            field=models.TextField(blank=True, null=True),
        ),
    ]
