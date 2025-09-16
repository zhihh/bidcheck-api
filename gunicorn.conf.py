# Gunicorn 生产环境配置
import multiprocessing
import os

# 服务器套接字
bind = "0.0.0.0:8000"
backlog = 2048

# 工作进程配置
workers = int(os.environ.get("WORKERS", min(multiprocessing.cpu_count(), 4)))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# 超时配置
timeout = 120
keepalive = 5

# 日志配置
accesslog = "logs/access.log"
errorlog = "logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 进程名称
proc_name = "bidcheck-api"

# 重启配置
preload_app = True
reload = False

# 安全配置
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190
