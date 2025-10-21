#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    logger.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)
    
    logger.info(f"Original dataset shape: {df.shape}")
    
    # Drop outliers
    min_price = args.min_price
    max_price = args.max_price
    logger.info(f"Filtering prices between ${min_price} and ${max_price}")
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    
    # Filter out listings with minimum_nights > 365 (long-term leases, not short-term rentals)
    logger.info("Filtering out long-term leases (minimum_nights > 365)")
    df = df[df['minimum_nights'] <= 365].copy()
    
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Filter by geolocation (NYC bounds)
    logger.info("Filtering by NYC geographic bounds")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    
    logger.info(f"Cleaned dataset shape: {df.shape}")
    
    # Save the cleaned file
    filename = "clean_sample.csv"
    df.to_csv(filename, index=False)

    # Log the new data
    logger.info("Logging artifact to W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)
    run.log_artifact(artifact)
    
    logger.info("Done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")
  
    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact (raw data) to download from W&B",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact (cleaned data) to upload to W&B",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price threshold for filtering outliers",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price threshold for filtering outliers",
        required=True
    )

    args = parser.parse_args()

    go(args)
