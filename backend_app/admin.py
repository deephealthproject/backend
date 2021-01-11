from django.contrib import admin
from django.contrib.auth import get_user_model
from django.contrib.auth.admin import UserAdmin

from backend_app import models


class AllowedPropertyAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.AllowedProperty._meta.fields]


class DatasetPermissionPlain(admin.ModelAdmin):
    list_display = [f.name for f in models.DatasetPermission._meta.fields]


class DatasetPermission(admin.TabularInline):
    model = models.DatasetPermission
    extra = 1


class DatasetAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.Dataset._meta.fields]
    inlines = (DatasetPermission,)


class InferenceAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.Inference._meta.fields]


class ModelAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.Model._meta.fields]


class ModelWeightsPermissionPlain(admin.ModelAdmin):
    list_display = [f.name for f in models.ModelWeightsPermission._meta.fields]


class ModelWeightsPermission(admin.TabularInline):
    model = models.ModelWeightsPermission
    extra = 1


class ModelWeightsAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.ModelWeights._meta.fields]
    inlines = (ModelWeightsPermission,)


class ProjectPermissionPlain(admin.ModelAdmin):
    list_display = [f.name for f in models.ProjectPermission._meta.fields]


class ProjectPermission(admin.TabularInline):
    model = models.ProjectPermission
    extra = 1


class ProjectAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.Project._meta.fields]
    inlines = (ProjectPermission,)


class PropertyAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.Property._meta.fields]


class TaskAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.Task._meta.fields]


class TrainingAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.Training._meta.fields]


class TrainingSettingAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.TrainingSetting._meta.fields]


class CustomUserAdmin(UserAdmin):
    ordering = ('date_joined',)
    list_display = ('username', 'email', 'date_joined', 'first_name', 'last_name', 'is_staff')


# Replace the default UserAdmin with CustomUserAdmin
admin.site.unregister(get_user_model())
admin.site.register(get_user_model(), CustomUserAdmin)

admin.site.register(models.AllowedProperty, AllowedPropertyAdmin)
admin.site.register(models.Dataset, DatasetAdmin)
admin.site.register(models.DatasetPermission, DatasetPermissionPlain)
admin.site.register(models.Inference, InferenceAdmin)
admin.site.register(models.Model, ModelAdmin)
admin.site.register(models.ModelWeightsPermission, ModelWeightsPermissionPlain)
admin.site.register(models.ModelWeights, ModelWeightsAdmin)
admin.site.register(models.ProjectPermission, ProjectPermissionPlain)
admin.site.register(models.Project, ProjectAdmin)
admin.site.register(models.Property, PropertyAdmin)
admin.site.register(models.Task, TaskAdmin)
admin.site.register(models.Training, TrainingAdmin)
admin.site.register(models.TrainingSetting, TrainingSettingAdmin)

admin.site.site_header = "DeepHealth Back-End Administration"
admin.site.site_title = "DeepHealth Admin Portal"
admin.site.index_title = "DeepHealth"
