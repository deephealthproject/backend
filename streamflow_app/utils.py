import json

data = """{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "helm3.json",
  "type": "object",
  "properties": {
    "atomic": {
      "type": "boolean",
      "description": "If set, installation process purges chart on fail (also sets wait flag)"
    },
    "caFile": {
      "type": "string",
      "description": "Verify certificates of HTTPS-enabled servers using this CA bundle"
    },
    "certFile": {
      "type": "string",
      "description": "Identify HTTPS client using this SSL certificate file"
    },
    "chart": {
      "type": "string",
      "description": "A chart archive. This can be a chart reference, a path to a packaged chart, a path to an unpacked chart directory or a URL"
    },
    "chartVersion": {
      "type": "string",
      "description": "Specify the exact chart version to install",
      "default": "latest"
    },
    "commandLineValues": {
      "type": "string",
      "description": "Set values on the command line. Can separate values with commas: key1=val1,key2=val2"
    },
    "debug": {
      "type": "boolean",
      "description": " Enable verbose output"
    },
    "depUp": {
      "type": "boolean",
      "description": "Run helm dependency update before installing the chart"
    },
    "devel": {
      "type": "boolean",
      "description": " Use development versions, too (equivalent to version `>0.0.0-0`). If version is set, this is ignored"
    },
    "fileValues": {
      "type": "string",
      "description": "Set values from respective files. Can separate values with commas: key1=path1,key2=path2"
    },
    "inCluster": {
      "type": "boolean",
      "description": "If true, the Helm connector will use a ServiceAccount to connect to the Kubernetes cluster. This is useful when StreamFlow runs directly inside a Kubernetes Pod",
      "default": false
    },
    "keepHistory": {
      "type": "boolean",
      "description": "Remove all associated resources and mark the release as deleted, but retain the release history",
      "default": false
    },
    "keyFile": {
      "type": "string",
      "description": "Identify HTTPS client using this SSL key file"
    },
    "keyring": {
      "type": "string",
      "description": "Location of public keys used for verification",
      "default": "${HOME}/.gnupg/pubring.gpg"
    },
    "kubeContext": {
      "type": "string",
      "description": "Name of the kubeconfig context to use"
    },
    "kubeconfig": {
      "type": "string",
      "description": "Absolute path of the kubeconfig file to be used"
    },
    "namespace": {
      "type": "string",
      "description": "Namespace to install the release into",
      "default": "Current kube config namespace"
    },
    "nameTemplate": {
      "type": "string",
      "description": "Specify template used to name the release"
    },
    "noHooks": {
      "type": "boolean",
      "description": "Prevent hooks from running during install"
    },
    "password": {
      "type": "string",
      "description": " Chart repository password where to locate the requested chart"
    },
    "registryConfig": {
      "type": "string",
      "description": "Path to the registry config file",
      "default": "${HOME}/.config/helm/registry.json"
    },
    "repositoryCache": {
      "type": "string",
      "description": "Path to the file containing cached repository indexes",
      "default": "${HOME}/.cache/helm/repository"
    },
    "repositoryConfig": {
      "type": "string",
      "description": "Path to the file containing repository names and URLs",
      "default": "${HOME}/.config/helm/repositories.yaml"
    },
    "releaseName": {
      "type": "string",
      "description": "The release name. If unspecified, it will autogenerate one for you"
    },
    "renderSubchartNotes": {
      "type": "boolean",
      "description": "Render subchart notes along with the parent"
    },
    "repo": {
      "type": "string",
      "description": "Chart repository url where to locate the requested chart"
    },
    "skipCrds": {
      "type": "boolean",
      "description": "If set, no CRDs will be installed",
      "default": false
    },
    "stringValues": {
      "type": "string",
      "description": "Set string values. Can separate values with commas: key1=val1,key2=val2"
    },
    "timeout": {
      "type": "string",
      "description": "Time to wait for any individual Kubernetes operation",
      "default": "1000m"
    },
    "transferBufferSize": {
      "type": "integer",
      "description": "Buffer size allocated for local and remote data transfers",
      "default": "32MiB - 1B",
      "$comment": "Kubernetes Python client talks with its server counterpart, written in Golang, via Websocket protocol. The standard websocket package in Golang defines DefaultMaxPayloadBytes equal to 32 MB. Nevertheless, since kubernetes-client prepends channel number to the actual payload (which is always 0 for STDIN), we must reserve 1 byte for this purpose"
    },
    "username": {
      "type": "string",
      "description": "Chart repository username where to locate the requested chart"
    },
    "yamlValues": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Specify values in a list of YAML files and/or URLs",
      "default": []
    },
    "verify": {
      "type": "boolean",
      "description": "Verify the package before installing it"
    },
    "wait": {
      "type": "boolean",
      "description": "If set, will wait until all Pods, PVCs, Services, and minimum number of Pods of a Deployment are in a ready state before marking the release as successful. It will wait for as long as timeout",
      "default": true
    }
  },
  "required": [
    "chart"
  ],
  "additionalProperties": false
}
"""


def streamflow_to_django(data):
    data = json.loads(data)
    output = ""
    required_fields = data.get('required')
    for k, v in data['properties'].items():
        lhs = k
        rhs = ''
        help_text = v['description']
        default = v.get('default')
        if v['type'] == "boolean":
            rhs = f'models.BooleanField('
        elif v['type'] == 'string':
            rhs = f'models.CharField(max_length=255,'
        elif v['type'] == 'integer':
            rhs = f'models.IntegerField('
        elif v['type'] == 'array':
            rhs = f'models.TextField('

        if default:
            rhs += f' default=\'{default}\','

        if lhs not in required_fields:
            rhs += ' null=True, blank=True,'

        rhs += f' help_text=\'{help_text}\')'
        output += f'{lhs} = {rhs}\n'
    print(output)


if __name__ == '__main__':
    streamflow_to_django(data)
