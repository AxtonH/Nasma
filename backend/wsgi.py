from backend.app import create_app

# WSGI entrypoint for production servers like Gunicorn
app = create_app()


