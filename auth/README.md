# OAuth2 Authentication

## Configuration

### 1. Backend Internal provider - Django OAuth Toolkit (DOT)
__Scenario__: You want to make your own Authorization Server to issue access tokens to client applications for a certain API. This configuration can be employed in REST only and web browser modes.

#### Register an application
To obtain a valid access_token first we must register an application.
Point your browser at: [http://mysite/backend/auth/applications/](http://mysite/backend/auth/applications/)

Click on the link to create a new application and fill the form with the following data:
```
Name: just a name of your choice
Client id: keep default value
Client secret: keep default value
Client Type: public
Authorization Grant Type: Resource owner password-based
Redirect uris: keep default value
```
Keep note of client id (and client secret if you set client type as _confidential_) and save your app.

#### Get your token and use APIs
At this point we’re ready to request an _access_token_. Open your shell:
<!-- ```shell script
curl -X POST -d "grant_type=password&username=<user_name>&password=<password>" -u"<client_id>:<client_secret>" http://mysite/backend/auth/token/
``` -->
```shell script
curl -X POST -H "Content-Type: application/x-www-form-urlencoded" -d "grant_type=password&username=<user_name>&password=<password>&client_id=<client_id>" http://mysite/backend/auth/token/
```
The user_name and password are the credential of the users registered in your Authorization Server, like any user created in Step 2. Response should be something like:

```json
{
    "access_token": "<your_access_token>",
    "token_type": "Bearer",
    "expires_in": 36000,
    "refresh_token": "<your_refresh_token>",
    "scope": "read write"
}
```

Grab your access_token and start using the test API:
```shell script
curl -H "Authorization: Bearer <your_access_token>" http://mysite/backend/auth/testUser/
```
Example:
```shell script
# Request
curl -H "Authorization: Bearer HqddkWJBvqjrNLwA8iiy9IYQqZENW2" https://jenkins-master-deephealth-unix01.ing.unimore.it/backend/auth/testUser/

# Response
[{"username": "dhtest", "email": "", "first_name": "", "last_name": ""}]
```

Some time has passed and your access token is about to expire, you can get renew the access token issued using the refresh token:
```shell script
# Refresh token request
curl -X POST -H "Content-Type: application/x-www-form-urlencoded" -d "grant_type=refresh_token&refresh_token=<your_refresh_token>&client_id=<client_id>" http://mysite/backend/auth/token/
```
Your response should be exactly the same your first access_token request, containing a new access_token and refresh_token.

### 2. GitHub

__Scenario__: You want to provide user registration and login using social sites credentials (GitHub). This configuration needs a web browser.

Log in to your GitHub account, go to Settings. In the left menu you will see Developer settings. Click on OAuth applications.

In the OAuth applications screen click on Register a new application or navigate at [github.com/settings/applications/new](https://github.com/settings/applications/new).

The relevant step here is that the Authorization callback URL must be `http://mysite/backend/auth/complete/github/`.

Register the application and grab the information we need:
```
Client ID
44fd4145a8d85fda4ff1
Client Secret
2de7904bdefe32d315805d3b7daec7906cc0e9e7
```
Now we update the secret config file:
```
SOCIAL_AUTH_GITHUB_KEY='44fd4145a8d85fda4ff1'
SOCIAL_AUTH_GITHUB_SECRET='2de7904bdefe32d315805d3b7daec7906cc0e9e7'
```
Navigate through browser to [http://mysite/backend/auth/login/](http://mysite/backend/auth/login/) and click _Login with GitHub_ link.

<!-- ## Flow example

1. The client knows the _client_id_ of the backend application.
1. An user request to authenticate against the backend authentication provider
-->

### Create users

The `/auth/create/` API lets to create new users providing _username_ and _password_.
Example:
```shell script
# Request
curl -X POST -H "Content-Type: application/x-www-form-urlencoded" -d "username=<username>&password=<password>" --header "Authorization: Bearer <access_token>" "http://mysite/backend/auth/create/"

# Response
> {"id": 1, "username": "test",}
```


## Testing

Backend provides the `/auth/testUser/` API for testing authentication, which returns some information about the _dhtest_ user. __Authorization token must be included in every request__, otherwise "HTTP 401 Unauthorized" will be thrown.

Example:
```shell script
# Request
curl "http://mysite/backend/auth/testUser/" --header "Authorization: Bearer <access_token>"

# Response
> {"username": "dhtest"}
```
