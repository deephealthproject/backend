from django.contrib import admin

from streamflow_app import models


class SFSSHAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.SFSSH._meta.fields]


class SFHelmAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.SFHelm._meta.fields]


admin.site.register(models.SFSSH, SFSSHAdmin)
admin.site.register(models.SFHelm, SFHelmAdmin)
