from django.conf import settings
from django.db import models


class SFEnvironment(models.Model):
    """StreamFlow basic environment"""

    class SFConfigType(models.TextChoices):
        DOCKER = 'DCK', 'docker'
        DOCKER_COMPOSE = 'DCC', 'docker-compose'
        SSH = 'SSH', 'ssh'
        HELM = 'HLM', 'helm'
        SLURM = 'SLM', 'slurm'

    name = models.CharField(max_length=255)
    type = models.CharField(choices=SFConfigType.choices, max_length=3)

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    class Meta:
        abstract = True


class SFSSH(SFEnvironment):
    """StreamFlow SSH Model"""
    username = models.CharField(max_length=255, help_text="Username needed to connect with Occam environment")
    hostname = models.CharField(max_length=255, help_text="Hostname of the HPC facility")

    type = models.CharField(choices=SFEnvironment.SFConfigType.choices, max_length=3,
                            default=SFEnvironment.SFConfigType.SSH)

    # FilePathField could be a better choice but requires `path` which is the absolute filesystem
    # path to a directory from which this FilePathField should get its choices.
    ssh_key = models.CharField(max_length=4096,
                               help_text="Path to the SSH key needed to connect with Slurm environment")
    file = models.CharField(max_length=4096, null=True, blank=True,
                            help_text="Path to a file containing a Jinja2 template, describing how the StreamFlow"
                                      " command should be executed in the remote environment")
    # TODO Encrypted with user password?
    ssh_key_passphrase = models.CharField(max_length=255, null=True, blank=True,
                                          help_text="Passphrase protecting the SSH key")

    class Meta:
        ordering = ['id']
