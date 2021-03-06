# Generated by Django 3.1.2 on 2020-11-20 13:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backend_app', '0027_edit_permission_choices'),
    ]

    operations = [
        migrations.RenameField(
            model_name='dataset',
            old_name='owners',
            new_name='users',
        ),
        migrations.RenameField(
            model_name='modelweights',
            old_name='owners',
            new_name='users',
        ),
        migrations.AlterField(
            model_name='datasetpermission',
            name='permission',
            field=models.CharField(choices=[('OWN', 'Can change and delete'), ('VIEW', 'Can view')], default='OWN', max_length=4),
        ),
        migrations.AlterField(
            model_name='modelweightspermission',
            name='permission',
            field=models.CharField(choices=[('OWN', 'Can change and delete'), ('VIEW', 'Can view')], default='OWN', max_length=4),
        ),
        migrations.AlterField(
            model_name='projectpermission',
            name='permission',
            field=models.CharField(choices=[('OWN', 'Can change and delete'), ('VIEW', 'Can view')], default='OWN', max_length=4),
        ),
    ]
