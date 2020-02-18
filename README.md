# DeepHealth Toolkit back-end

The DeepHealth back-end interacts with the front-end, serving various APIs;

It receives a configuration from the front-end then runs a deep learning pipeline based on PyECVL and PyEDDL.  

## Installation
### Requirements
- Python3.6+
- PyECVL 0.1.0+
- PyEDDL 0.2.0+

Clone and install back-end with:
```bash
cd ~
git clone https://github.com/deephealthproject/backend.git
cd backend
pip install -r requirements.txt
```
Create a `secrets.json` file in `~/backend/` with the following structure:
```json
{
  "SECRET_KEY": "<key>",
  "CELERY_BROKER_URL": "amqp://<username>:<password>@localhost"
}
```
Generate a new "SECRET_KEY" with `python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())'
` and set username e password for celery. 


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

# Start the development server
python manage.py runserver <my-server>:<my-server-port>
```
