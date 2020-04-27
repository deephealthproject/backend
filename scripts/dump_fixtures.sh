#!/bin/bash
mkdir -p backend_app/fixture
./manage.py dumpdata --indent 2 --output backend_app/fixtures/tasks backend_app.task
./manage.py dumpdata --indent 2 --output backend_app/fixtures/property backend_app.property
./manage.py dumpdata --indent 2 --output backend_app/fixtures/allowedproperty backend_app.allowedproperty
./manage.py dumpdata --indent 2 --output backend_app/fixtures/model backend_app.model
./manage.py dumpdata --indent 2 --output backend_app/fixtures/dataset backend_app.dataset
