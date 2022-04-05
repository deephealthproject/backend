# Generated by Django 3.2.6 on 2022-03-29 15:49

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='SFSSH',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('username', models.CharField(help_text='Username needed to connect with Occam environment', max_length=255)),
                ('hostname', models.CharField(help_text='Hostname of the HPC facility', max_length=255)),
                ('type', models.CharField(choices=[('DCK', 'SFDocker'), ('DCC', 'SFDockerCompose'), ('SSH', 'SFSSH'), ('HLM', 'SFHelm'), ('SLM', 'SFSlurm')], default='SSH', max_length=3)),
                ('ssh_key', models.CharField(help_text='Path to the SSH key needed to connect with Slurm environment', max_length=4096)),
                ('file', models.CharField(blank=True, help_text='Path to a file containing a Jinja2 template, describing how the StreamFlow command should be executed in the remote environment', max_length=4096, null=True)),
                ('ssh_key_passphrase', models.CharField(blank=True, help_text='Passphrase protecting the SSH key', max_length=255, null=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['id'],
            },
        ),
        migrations.CreateModel(
            name='SFHelm',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('type', models.CharField(choices=[('DCK', 'SFDocker'), ('DCC', 'SFDockerCompose'), ('SSH', 'SFSSH'), ('HLM', 'SFHelm'), ('SLM', 'SFSlurm')], default='HLM', max_length=3)),
                ('atomic', models.BooleanField(blank=True, help_text='If set, installation process purges chart on fail (also sets wait flag)', null=True)),
                ('caFile', models.CharField(blank=True, help_text='Verify certificates of HTTPS-enabled servers using this CA bundle', max_length=255, null=True)),
                ('certFile', models.CharField(blank=True, help_text='Identify HTTPS client using this SSL certificate file', max_length=255, null=True)),
                ('chart', models.CharField(help_text='A chart archive. This can be a chart reference, a path to a packaged chart, a path to an unpacked chart directory or a URL', max_length=255)),
                ('chartVersion', models.CharField(blank=True, default='latest', help_text='Specify the exact chart version to install', max_length=255, null=True)),
                ('commandLineValues', models.CharField(blank=True, help_text='Set values on the command line. Can separate values with commas: key1=val1,key2=val2', max_length=255, null=True)),
                ('debug', models.BooleanField(blank=True, help_text=' Enable verbose output', null=True)),
                ('depUp', models.BooleanField(blank=True, help_text='Run helm dependency update before installing the chart', null=True)),
                ('devel', models.BooleanField(blank=True, help_text='Use development versions, too (equivalent to version `>0.0.0-0`). If version is set, this is ignored', null=True)),
                ('fileValues', models.CharField(blank=True, help_text='Set values from respective files. Can separate values with commas: key1=path1,key2=path2', max_length=255, null=True)),
                ('inCluster', models.BooleanField(blank=True, help_text='If true, the Helm connector will use a ServiceAccount to connect to the Kubernetes cluster. This is useful when StreamFlow runs directly inside a Kubernetes Pod', null=True)),
                ('keepHistory', models.BooleanField(blank=True, help_text='Remove all associated resources and mark the release as deleted, but retain the release history', null=True)),
                ('keyFile', models.CharField(blank=True, help_text='Identify HTTPS client using this SSL key file', max_length=255, null=True)),
                ('keyring', models.CharField(blank=True, default='${HOME}/.gnupg/pubring.gpg', help_text='Location of public keys used for verification', max_length=255, null=True)),
                ('kubeContext', models.CharField(blank=True, help_text='Name of the kubeconfig context to use', max_length=255, null=True)),
                ('kubeconfig', models.CharField(blank=True, help_text='Absolute path of the kubeconfig file to be used', max_length=255, null=True)),
                ('namespace', models.CharField(blank=True, default='Current kube config namespace', help_text='Namespace to install the release into', max_length=255, null=True)),
                ('nameTemplate', models.CharField(blank=True, help_text='Specify template used to name the release', max_length=255, null=True)),
                ('noHooks', models.BooleanField(blank=True, help_text='Prevent hooks from running during install', null=True)),
                ('password', models.CharField(blank=True, help_text=' Chart repository password where to locate the requested chart', max_length=255, null=True)),
                ('registryConfig', models.CharField(blank=True, default='${HOME}/.config/helm/registry.json', help_text='Path to the registry config file', max_length=255, null=True)),
                ('repositoryCache', models.CharField(blank=True, default='${HOME}/.cache/helm/repository', help_text='Path to the file containing cached repository indexes', max_length=255, null=True)),
                ('repositoryConfig', models.CharField(blank=True, default='${HOME}/.config/helm/repositories.yaml', help_text='Path to the file containing repository names and URLs', max_length=255, null=True)),
                ('releaseName', models.CharField(blank=True, help_text='The release name. If unspecified, it will autogenerate one for you', max_length=255, null=True)),
                ('renderSubchartNotes', models.BooleanField(blank=True, help_text='Render subchart notes along with the parent', null=True)),
                ('repo', models.CharField(blank=True, help_text='Chart repository url where to locate the requested chart', max_length=255, null=True)),
                ('skipCrds', models.BooleanField(blank=True, help_text='If set, no CRDs will be installed', null=True)),
                ('stringValues', models.CharField(blank=True, help_text='Set string values. Can separate values with commas: key1=val1,key2=val2', max_length=255, null=True)),
                ('timeout', models.CharField(blank=True, default='1000m', help_text='Time to wait for any individual Kubernetes operation', max_length=255, null=True)),
                ('transferBufferSize', models.IntegerField(blank=True, default='32MiB - 1B', help_text='Buffer size allocated for local and remote data transfers', null=True)),
                ('username', models.CharField(blank=True, help_text='Chart repository username where to locate the requested chart', max_length=255, null=True)),
                ('yamlValues', models.TextField(blank=True, help_text='Specify values in a list of YAML files and/or URLs', null=True)),
                ('verify', models.BooleanField(blank=True, help_text='Verify the package before installing it', null=True)),
                ('wait', models.BooleanField(blank=True, default='True', help_text='If set, will wait until all Pods, PVCs, Services, and minimum number of Pods of a Deployment are in a ready state before marking the release as successful. It will wait for as long as timeout', null=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['id'],
            },
        ),
    ]
