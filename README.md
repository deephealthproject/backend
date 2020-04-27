# DeepHealth Toolkit back-end

The DeepHealth back-end interacts with the front-end, serving various APIs;

It receives a configuration from the front-end then runs a deep learning pipeline based on PyECVL and PyEDDL.  

## Installation

### Requirements
- Python3.6+
- curl -- `sudo apt install curl`
- PyEDDL 0.4.0+ and PyECVL 0.1.0



Clone and install back-end with:

```bash
cd ~
git clone https://github.com/deephealthproject/backend.git
cd backend
pip install -r requirements.txt
```
Generate a new SECRET_KEY with:

```bash
python -c 'from django.core.management.utils import get_random_secret_key;print(get_random_secret_key())'
```

Edit the `~/config` file to configure the application (SECRET_KEY, DB and RabbitMQ connection and other optional Django settings).


##### Celery
Install with: `sudo apt install rabbitmq-server` 
and run the celery deamon with: `python manage.py celery`.



## First run

The back-end is a web-server based on Django, so it must be initialized like any Django project.


```bash
cd ~/backend

# Apply all the migrations
python manage.py migrate

# Creating an admin user
python manage.py createsuperuser

# Load db default entries
python manage.py loaddata tasks.json property.json allowedproperty.json dataset.json model.json

# Start the development server
python manage.py runserver <my-server>:<my-server-port>

# Start celery
python manage.py celery
```
