#!/usr/bin/env bash

./manage.py graph_models -a -g -o imgs/db_schema_full.svg
./manage.py graph_models backend_app auth_app streamflow_app auth -X Group,Permission,AbstractUser -o imgs/db_schema_base.svg