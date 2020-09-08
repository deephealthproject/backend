# Generated by Django 3.0.7 on 2020-09-08 09:17

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('backend_app', '0021_add_modelweights_permission'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingsetting',
            name='training_id',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, to='backend_app.Training'),
            preserve_default=False,
        ),
        migrations.AlterUniqueTogether(
            name='trainingsetting',
            unique_together={('training_id', 'property_id')},
        ),
        migrations.RemoveField(
            model_name='trainingsetting',
            name='modelweights_id',
        ),
    ]