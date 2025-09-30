# gunicorn_config.py

# Server Socket
bind = "0.0.0.0:5001"

# Worker Processes
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"

# Worker timeout (in seconds)
# This helps prevent workers from being killed during long uploads
timeout = 120

# Maximum request body size (in bytes)
# 100 * 1024 * 1024 = 100 MB
limit_request_body = 104857600

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
