"""
backend/app/blueprints/docs/storage.py

Storage helpers for uploading, retrieving, and managing files in
S3 (production) or MinIO (local dev).

All callers interact only with the three public helpers:
  - upload_to_storage(file_obj, key, content_type)
  - get_file_url(key, expiry_seconds)
  - delete_from_storage(key)
  - allowed_file(filename, allowed_extensions)

The backend (S3 vs MinIO) is selected at runtime via the
STORAGE_BACKEND env var ("s3" | "minio").  Both use boto3 with
different endpoint configurations.
"""

import os
import logging
from functools import lru_cache
from typing import BinaryIO

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from flask import current_app

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _get_storage_config() -> dict:
    """
    Pull storage credentials from Flask app config (which reads .env).
    Centralised here so every helper uses the same source of truth.
    """
    return {
        "backend":           current_app.config.get("STORAGE_BACKEND", "s3"),
        "bucket":            current_app.config["STORAGE_BUCKET"],
        "region":            current_app.config.get("AWS_REGION", "us-east-1"),
        "access_key":        current_app.config.get("AWS_ACCESS_KEY_ID"),
        "secret_key":        current_app.config.get("AWS_SECRET_ACCESS_KEY"),
        # MinIO only
        "endpoint_url":      current_app.config.get("MINIO_ENDPOINT_URL"),   # e.g. http://minio:9000
        "public_base_url":   current_app.config.get("MINIO_PUBLIC_BASE_URL"),
    }


def _build_s3_client(cfg: dict):
    """
    Build a boto3 S3 client.
    For MinIO, endpoint_url overrides the AWS service endpoint.
    """
    kwargs = {
        "region_name":            cfg["region"],
        "aws_access_key_id":      cfg["access_key"],
        "aws_secret_access_key":  cfg["secret_key"],
    }
    if cfg["backend"] == "minio" and cfg.get("endpoint_url"):
        kwargs["endpoint_url"] = cfg["endpoint_url"]

    return boto3.client("s3", **kwargs)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """
    Return True if *filename* has a dot and its extension is in the
    allowed set (case-insensitive).

    Example:
        allowed_file("notes.PDF", {"pdf", "docx"})  →  True
        allowed_file("exploit", {"pdf"})             →  False
    """
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in allowed_extensions


def upload_to_storage(
    file_obj: BinaryIO,
    key: str,
    content_type: str = "application/octet-stream",
) -> str:
    """
    Upload *file_obj* to the configured bucket under *key*.

    Args:
        file_obj:     File-like object (already seeked to 0).
        key:          Object key inside the bucket, e.g.
                      "documents/<user_id>/<uuid>.pdf"
        content_type: MIME type passed to S3 for correct browser handling.

    Returns:
        The S3/MinIO object key (same as *key*).

    Raises:
        RuntimeError: on upload failure (wraps boto3 exceptions so callers
                      don't need to import botocore).
    """
    cfg = _get_storage_config()
    client = _build_s3_client(cfg)

    try:
        client.upload_fileobj(
            Fileobj=file_obj,
            Bucket=cfg["bucket"],
            Key=key,
            ExtraArgs={"ContentType": content_type},
        )
        logger.info("Uploaded object to bucket=%s key=%s", cfg["bucket"], key)
        return key

    except NoCredentialsError:
        logger.error("Storage credentials are not configured correctly.")
        raise RuntimeError("Storage credentials missing.")
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        logger.error("S3 ClientError [%s] uploading %s: %s", error_code, key, exc)
        raise RuntimeError(f"File upload failed ({error_code}).")


def get_file_url(key: str, expiry_seconds: int = 3600) -> str:
    """
    Generate a pre-signed URL for downloading *key*.

    For MinIO in local dev, a plain public URL is returned if
    MINIO_PUBLIC_BASE_URL is set (avoids Docker network hostname issues).

    Args:
        key:            Object key inside the bucket.
        expiry_seconds: URL validity window (default 1 hour).

    Returns:
        A time-limited HTTPS URL string.
    """
    cfg = _get_storage_config()

    # MinIO local dev shortcut: return direct public URL
    if cfg["backend"] == "minio" and cfg.get("public_base_url"):
        return f"{cfg['public_base_url'].rstrip('/')}/{cfg['bucket']}/{key}"

    client = _build_s3_client(cfg)
    try:
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": cfg["bucket"], "Key": key},
            ExpiresIn=expiry_seconds,
        )
        return url
    except ClientError as exc:
        logger.error("Failed to generate pre-signed URL for %s: %s", key, exc)
        raise RuntimeError("Could not generate file URL.")


def delete_from_storage(key: str) -> None:
    """
    Permanently delete an object from the bucket.

    Called by the scheduled cleanup job after soft-deleted documents
    have aged past the retention window.

    Args:
        key: Object key to delete.

    Raises:
        RuntimeError: if deletion fails.
    """
    cfg = _get_storage_config()
    client = _build_s3_client(cfg)

    try:
        client.delete_object(Bucket=cfg["bucket"], Key=key)
        logger.info("Deleted object bucket=%s key=%s", cfg["bucket"], key)
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        logger.error("Failed to delete %s [%s]: %s", key, error_code, exc)
        raise RuntimeError(f"File deletion failed ({error_code}).")


def ensure_bucket_exists() -> None:
    """
    Create the configured bucket if it does not already exist.
    Safe to call on every app startup — a no-op if the bucket is present.

    Intended for MinIO local dev. In production the bucket is
    pre-created via Terraform / CDK.
    """
    cfg = _get_storage_config()
    if cfg["backend"] != "minio":
        logger.debug("Skipping bucket creation check (backend=%s)", cfg["backend"])
        return

    client = _build_s3_client(cfg)
    bucket = cfg["bucket"]

    try:
        client.head_bucket(Bucket=bucket)
        logger.debug("Bucket '%s' already exists.", bucket)
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code in ("404", "NoSuchBucket"):
            client.create_bucket(Bucket=bucket)
            logger.info("Created MinIO bucket '%s'.", bucket)
        else:
            logger.error("Unexpected error checking bucket '%s': %s", bucket, exc)
            raise


def get_object_metadata(key: str) -> dict:
    """
    Return HEAD metadata for an object (size, content-type, last-modified).

    Useful for integrity checks after upload.

    Returns:
        dict with keys: content_length, content_type, last_modified, etag
    """
    cfg = _get_storage_config()
    client = _build_s3_client(cfg)

    try:
        resp = client.head_object(Bucket=cfg["bucket"], Key=key)
        return {
            "content_length": resp.get("ContentLength"),
            "content_type":   resp.get("ContentType"),
            "last_modified":  resp.get("LastModified"),
            "etag":           resp.get("ETag", "").strip('"'),
        }
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code == "404":
            raise FileNotFoundError(f"Object '{key}' not found in storage.")
        logger.error("head_object failed for %s [%s]: %s", key, error_code, exc)
        raise RuntimeError(f"Metadata fetch failed ({error_code}).")