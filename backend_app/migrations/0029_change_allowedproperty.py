# Generated by Django 3.1.3 on 2021-01-07 11:13

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('backend_app', '0028_rename_owners_to_users'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='modelweights',
            options={'ordering': ['id'], 'verbose_name_plural': 'Model Weights'},
        ),
        migrations.AddField(
            model_name='allowedproperty',
            name='dataset_id',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='backend_app.dataset'),
        ),
        migrations.AlterField(
            model_name='datasetpermission',
            name='dataset',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='permission', to='backend_app.dataset'),
        ),
        migrations.AlterField(
            model_name='modelweightspermission',
            name='modelweight',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='permission', to='backend_app.modelweights'),
        ),
        migrations.AlterField(
            model_name='projectpermission',
            name='project',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='permission', to='backend_app.project'),
        ),
        migrations.AlterUniqueTogether(
            name='allowedproperty',
            unique_together=set(),
        ),
        migrations.AddConstraint(
            model_name='allowedproperty',
            constraint=models.UniqueConstraint(fields=('property_id', 'model_id', 'dataset_id'), name='unique_with_optional'),
        ),
        migrations.AddConstraint(
            model_name='allowedproperty',
            constraint=models.UniqueConstraint(condition=models.Q(dataset_id=None), fields=('property_id', 'model_id'), name='unique_without_optional'),
        ),
    ]