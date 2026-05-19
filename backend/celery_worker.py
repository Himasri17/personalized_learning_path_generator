"""
celery_worker.py
----------------
Celery application entry point.

Starting the worker
-------------------
Development (single worker, all queues):
    celery -A celery_worker.celery worker --loglevel=info

Production (separate workers per queue for isolation):
    celery -A celery_worker.celery worker -Q ingest  --loglevel=info --concurrency=2
    celery -A celery_worker.celery worker -Q qgen    --loglevel=info --concurrency=4
    celery -A celery_worker.celery worker -Q grade   --loglevel=info --concurrency=4

Monitoring (Flower dashboard on :5555):
    celery -A celery_worker.celery flower

Beat scheduler (periodic tasks):
    celery -A celery_worker.celery beat --loglevel=info
"""

from __future__ import annotations

import logging
import os

from app import create_app
from app.extensions import celery

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Create the Flask application (pushes an app context for every task)
# ---------------------------------------------------------------------------

env      = os.getenv("APP_ENV", "development")
flask_app = create_app(env)

# The factory already called _init_celery(app), which wraps every task so it
# runs inside a Flask app context.  We just expose `celery` for the CLI.

logger.info("Celery worker booted [env=%s, broker=%s]", env, flask_app.config["REDIS_URL"])


# ---------------------------------------------------------------------------
# Auto-discover tasks from every registered ml.tasks module
# ---------------------------------------------------------------------------

celery.autodiscover_tasks(
    [
        "ml.tasks.ingest_task",
        "ml.tasks.qgen_task",
        "ml.tasks.grade_task",
    ],
    force=True,
)


# ---------------------------------------------------------------------------
# Periodic tasks (Celery Beat)
# ---------------------------------------------------------------------------

from celery.schedules import crontab  # noqa: E402

celery.conf.beat_schedule = {
    # Clean up orphaned sessions older than 24 h (status stuck at "generating")
    "cleanup-stale-sessions": {
        "task":     "ml.tasks.cleanup_task.cleanup_stale_sessions",
        "schedule": crontab(hour=3, minute=0),   # daily at 03:00 UTC
        "args":     (),
    },
}


# ---------------------------------------------------------------------------
# Optional: global task lifecycle signals for observability
# ---------------------------------------------------------------------------

from celery.signals import (   # noqa: E402
    task_prerun,
    task_postrun,
    task_failure,
    worker_ready,
)


@worker_ready.connect
def _on_worker_ready(sender, **kwargs):
    logger.info("Celery worker is ready. Queues: ingest | qgen | grade")


@task_prerun.connect
def _on_task_prerun(task_id, task, args, kwargs, **extra):
    logger.debug("TASK START  [%s] id=%s", task.name, task_id)


@task_postrun.connect
def _on_task_postrun(task_id, task, args, kwargs, retval, state, **extra):
    logger.debug("TASK FINISH [%s] id=%s state=%s", task.name, task_id, state)


@task_failure.connect
def _on_task_failure(task_id, exception, traceback, sender, **kwargs):
    logger.error(
        "TASK FAILED [%s] id=%s error=%s",
        sender.name, task_id, exception,
        exc_info=(type(exception), exception, traceback),
    )