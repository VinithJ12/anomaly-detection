#!/usr/bin/env python3
import io
import time
import random
import logging
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from botocore.exceptions import BotoCoreError, ClientError
import os

#  Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()   # print to screen (this runs on your laptop)
    ]
)
logger = logging.getLogger(__name__)

#  S3 + config
try:
    s3 = boto3.client("s3")
    BUCKET_NAME = os.environ["BUCKET_NAME"]
    logger.info(f"Producer started. Target bucket: {BUCKET_NAME}")
except KeyError:
    logger.error("BUCKET_NAME environment variable is not set. Exiting.")
    exit(1)


def generate_batch(n_rows: int = 100, inject_anomalies: bool = True) -> pd.DataFrame:
    try:
        base_time = datetime.utcnow()

        data = {
            "timestamp": [
                (base_time + timedelta(minutes=i)).isoformat() for i in range(n_rows)
            ],
            "temperature": np.random.normal(loc=22.0, scale=1.5, size=n_rows).round(2),
            "humidity":    np.random.normal(loc=55.0, scale=5.0, size=n_rows).round(2),
            "pressure":    np.random.normal(loc=1013.0, scale=3.0, size=n_rows).round(2),
            "wind_speed":  np.abs(np.random.normal(loc=10.0, scale=2.5, size=n_rows)).round(2),
        }

        df = pd.DataFrame(data)

        # Inject a few obvious anomalies so the detector can catch them
        if inject_anomalies and n_rows > 10:
            anomaly_indices = random.sample(range(n_rows), k=max(1, n_rows // 20))
            for idx in anomaly_indices:
                col = random.choice(["temperature", "humidity", "pressure", "wind_speed"])
                direction = random.choice([-1, 1])
                df.at[idx, col] = df[col].mean() + direction * df[col].std() * random.uniform(5, 8)
            logger.info(f"Injected {len(anomaly_indices)} anomalies into batch.")

        return df

    except Exception as e:
        logger.error(f"Failed to generate batch: {e}")
        raise


def upload_batch(df: pd.DataFrame):
    try:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        key = f"raw/sensors_{timestamp}.csv"

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)

        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=csv_buffer.getvalue(),
            ContentType="text/csv"
        )
        logger.info(f"Uploaded {len(df)} rows → s3://{BUCKET_NAME}/{key}")
        return key

    except ClientError as e:
        # AWS-specific errors (e.g. wrong bucket name, permissions issue)
        logger.error(f"AWS error uploading batch: {e.response['Error']['Message']}")
    except BotoCoreError as e:
        # Lower-level connection/config errors
        logger.error(f"BotoCore error uploading batch: {e}")
    except Exception as e:
        logger.error(f"Unexpected error uploading batch: {e}")


if __name__ == "__main__":
    interval = int(os.getenv("INTERVAL_SECONDS", "60"))
    batch_num = 0
    logger.info(f"Producing batches every {interval}s. Ctrl+C to stop.")

    while True:
        try:
            batch_num += 1
            logger.info(f"--- Generating batch #{batch_num} ---")
            df = generate_batch(n_rows=100, inject_anomalies=True)
            upload_batch(df)
        except Exception as e:
            # Don't crash the loop — log and keep going
            logger.error(f"Batch #{batch_num} failed unexpectedly: {e}")

        # Sleep separately so Ctrl+C is always caught cleanly,
        # whether it fires during work or during the wait.
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Producer stopped by user.")
            break