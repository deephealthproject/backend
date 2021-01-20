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


class SFHelm(SFEnvironment):
    """StreamFlow Helm Model"""
    type = models.CharField(choices=SFEnvironment.SFConfigType.choices, max_length=3,
                            default=SFEnvironment.SFConfigType.HELM)

    atomic = models.BooleanField(null=True, blank=True,
                                 help_text='If set, installation process purges chart on fail (also sets wait flag)')
    caFile = models.CharField(max_length=255, null=True, blank=True,
                              help_text='Verify certificates of HTTPS-enabled servers using this CA bundle')
    certFile = models.CharField(max_length=255, null=True, blank=True,
                                help_text='Identify HTTPS client using this SSL certificate file')
    chart = models.CharField(max_length=255,
                             help_text='A chart archive. This can be a chart reference, a path to a packaged chart, '
                                       'a path to an unpacked chart directory or a URL')
    chartVersion = models.CharField(max_length=255, default='latest', null=True, blank=True,
                                    help_text='Specify the exact chart version to install')
    commandLineValues = models.CharField(max_length=255, null=True, blank=True,
                                         help_text='Set values on the command line. Can separate values with commas: '
                                                   'key1=val1,key2=val2')
    debug = models.BooleanField(null=True, blank=True, help_text=' Enable verbose output')
    depUp = models.BooleanField(null=True, blank=True,
                                help_text='Run helm dependency update before installing the chart')
    devel = models.BooleanField(null=True, blank=True,
                                help_text='Use development versions, too (equivalent to version `>0.0.0-0`). If '
                                          'version is set, this is ignored')
    fileValues = models.CharField(max_length=255, null=True, blank=True,
                                  help_text='Set values from respective files. Can separate values with commas: '
                                            'key1=path1,key2=path2')
    inCluster = models.BooleanField(null=True, blank=True,
                                    help_text='If true, the Helm connector will use a ServiceAccount to connect to '
                                              'the Kubernetes cluster. This is useful when StreamFlow runs directly '
                                              'inside a Kubernetes Pod')
    keepHistory = models.BooleanField(null=True, blank=True,
                                      help_text='Remove all associated resources and mark the release as deleted, '
                                                'but retain the release history')
    keyFile = models.CharField(max_length=255, null=True, blank=True,
                               help_text='Identify HTTPS client using this SSL key file')
    keyring = models.CharField(max_length=255, default='${HOME}/.gnupg/pubring.gpg', null=True, blank=True,
                               help_text='Location of public keys used for verification')
    kubeContext = models.CharField(max_length=255, null=True, blank=True,
                                   help_text='Name of the kubeconfig context to use')
    kubeconfig = models.CharField(max_length=255, null=True, blank=True,
                                  help_text='Absolute path of the kubeconfig file to be used')
    namespace = models.CharField(max_length=255, default='Current kube config namespace', null=True, blank=True,
                                 help_text='Namespace to install the release into')
    nameTemplate = models.CharField(max_length=255, null=True, blank=True,
                                    help_text='Specify template used to name the release')
    noHooks = models.BooleanField(null=True, blank=True, help_text='Prevent hooks from running during install')
    password = models.CharField(max_length=255, null=True, blank=True,
                                help_text=' Chart repository password where to locate the requested chart')
    registryConfig = models.CharField(max_length=255, default='${HOME}/.config/helm/registry.json', null=True,
                                      blank=True, help_text='Path to the registry config file')
    repositoryCache = models.CharField(max_length=255, default='${HOME}/.cache/helm/repository', null=True, blank=True,
                                       help_text='Path to the file containing cached repository indexes')
    repositoryConfig = models.CharField(max_length=255, default='${HOME}/.config/helm/repositories.yaml', null=True,
                                        blank=True, help_text='Path to the file containing repository names and URLs')
    releaseName = models.CharField(max_length=255, null=True, blank=True,
                                   help_text='The release name. If unspecified, it will autogenerate one for you')
    renderSubchartNotes = models.BooleanField(null=True, blank=True,
                                              help_text='Render subchart notes along with the parent')
    repo = models.CharField(max_length=255, null=True, blank=True,
                            help_text='Chart repository url where to locate the requested chart')
    skipCrds = models.BooleanField(null=True, blank=True, help_text='If set, no CRDs will be installed')
    stringValues = models.CharField(max_length=255, null=True, blank=True,
                                    help_text='Set string values. Can separate values with commas: key1=val1,key2=val2')
    timeout = models.CharField(max_length=255, default='1000m', null=True, blank=True,
                               help_text='Time to wait for any individual Kubernetes operation')
    transferBufferSize = models.IntegerField(default='32MiB - 1B', null=True, blank=True,
                                             help_text='Buffer size allocated for local and remote data transfers')
    username = models.CharField(max_length=255, null=True, blank=True,
                                help_text='Chart repository username where to locate the requested chart')
    yamlValues = models.TextField(null=True, blank=True, help_text='Specify values in a list of YAML files and/or URLs')
    verify = models.BooleanField(null=True, blank=True, help_text='Verify the package before installing it')
    wait = models.BooleanField(default='True', null=True, blank=True,
                               help_text='If set, will wait until all Pods, PVCs, Services, and minimum number of '
                                         'Pods of a Deployment are in a ready state before marking the release as '
                                         'successful. It will wait for as long as timeout')

    class Meta:
        ordering = ['id']
