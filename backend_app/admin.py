from django.contrib import admin

from backend_app import models


class AllowedPropertyAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.AllowedProperty._meta.fields]


class DatasetAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.Dataset._meta.fields]


class InferenceAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.Inference._meta.fields]


class ModelAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.Model._meta.fields]


class ModelWeightsAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.ModelWeights._meta.fields]


class ProjectAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.Project._meta.fields]


class PropertyAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.Property._meta.fields]


class TaskAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.Task._meta.fields]


class TrainingSettingAdmin(admin.ModelAdmin):
    list_display = [f.name for f in models.TrainingSetting._meta.fields]


admin.site.register(models.AllowedProperty, AllowedPropertyAdmin)
admin.site.register(models.Dataset, DatasetAdmin)
admin.site.register(models.Inference, InferenceAdmin)
admin.site.register(models.Model, ModelAdmin)
admin.site.register(models.ModelWeights, ModelWeightsAdmin)
admin.site.register(models.Project, ProjectAdmin)
admin.site.register(models.Property, PropertyAdmin)
admin.site.register(models.Task, TaskAdmin)
admin.site.register(models.TrainingSetting, TrainingSettingAdmin)

admin.site.site_header = "DeepHealth Back-End Administration"
admin.site.site_title = "DeepHealth Admin Portal"
admin.site.index_title = "DeepHealth"
