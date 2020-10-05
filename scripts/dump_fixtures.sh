#!/bin/bash
mkdir -p backend_app/fixtures
./manage.py dumpdata --indent 2 --output backend_app/fixtures/tasks.json backend_app.task
./manage.py dumpdata --indent 2 --output backend_app/fixtures/property.json backend_app.property
./manage.py dumpdata --indent 2 --output backend_app/fixtures/allowedproperty.json backend_app.allowedproperty
./manage.py dumpdata --indent 2 --output backend_app/fixtures/model.json backend_app.model
./manage.py dumpdata --indent 2 --output backend_app/fixtures/modelweights.json backend_app.modelweights
./manage.py dumpdata --indent 2 --output backend_app/fixtures/dataset.json backend_app.dataset
