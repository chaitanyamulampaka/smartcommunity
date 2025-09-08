# app/apps.py
from django.apps import AppConfig

class UsersConfig(AppConfig):   # <-- Change AppConfig to UsersConfig
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'

    def ready(self):
        import app.signals  # Make sure this points to your signals
