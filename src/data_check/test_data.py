import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data: pd.DataFrame) -> None:
    """Test if the DataFrame has the expected column names.
    
    Args:
        data: Input DataFrame to test
    """
    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_neighborhood_names(data: pd.DataFrame) -> None:
    """Test if neighborhood names are within expected values.
    
    Args:
        data: Input DataFrame to test
    """
    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float) -> None:
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    
    Args:
        data: Current dataset to test
        ref_data: Reference dataset to compare against
        kl_threshold: Maximum allowed KL divergence threshold
        
    Raises:
        AssertionError: If KL divergence exceeds the threshold
    """
    # Use newer pandas value_counts with normalize=True for probability distribution
    dist1 = data['neighbourhood_group'].value_counts(normalize=True).sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts(normalize=True).sort_index()
    
    # Ensure distributions sum to 1 and have matching indices
    assert np.isclose(dist1.sum(), 1.0)
    assert np.isclose(dist2.sum(), 1.0)
    assert dist1.index.equals(dist2.index)

    # Calculate KL divergence with improved numerical stability
    kl_div = scipy.stats.entropy(dist1, dist2, base=2)
    assert np.isfinite(kl_div) and kl_div < kl_threshold


def test_row_count(data: pd.DataFrame) -> None:
    """
    Test that the dataset has a reasonable number of rows (not too few after cleaning).
    We expect at least 1000 rows to have a meaningful dataset.
    
    Args:
        data: Input DataFrame to test
        
    Raises:
        AssertionError: If the dataset has too few rows
    """
    assert len(data) > 1000, f"Dataset has only {len(data)} rows, expected > 1000"

# My added functions here:
def test_price_range(data: pd.DataFrame, min_price: float, max_price: float) -> None:
    """
    Test that all prices are within the expected range (min_price to max_price).
    This ensures the cleaning step properly filtered outliers.
    
    Args:
        data: Input DataFrame to test
        min_price: Minimum acceptable price
        max_price: Maximum acceptable price
        
    Raises:
        AssertionError: If any prices are outside the valid range
    """
    # Check that all prices are within the valid range
    assert data['price'].between(min_price, max_price).all(), \
        f"Found prices outside range [{min_price}, {max_price}]"
    
    # Also verify no negative or zero prices slipped through
    assert (data['price'] > 0).all(), "Found non-positive prices in the dataset"
