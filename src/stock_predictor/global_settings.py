import os
import logging
from typing import Any
from dotenv import load_dotenv

load_dotenv()

# Path to the root directory of the project
ROOT_DIR = (
    os.path.dirname(os.path.abspath(__file__)).split("stock_predictor")[0]
    + "stock_predictor/src/"
)
# Path to the directory containing the data
DATA_DIR = ROOT_DIR + "data/"
# Path to main package
PACKAGE_DIR = ROOT_DIR + "stock_predictor/"

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER")
BASE_URL = os.getenv("BASE_URL")
PRODUCTION = os.getenv("PRODUCTION")
DATABASE_FILE = os.getenv("DATABASE_FILE")
VERBOSE = int(os.getenv("VERBOSE"))

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

DEFAULT_AVG_REDUCTION = 0.2
# Configure basic logging
logging.basicConfig(
    level=logging.INFO if PRODUCTION == "False" else logging.WARNING,
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


FILTER_TECHNICALS = (
    "[cyan2]Filtering [bold]{start_count}[/bold] stocks for technicals..."
)
FILTER_RECOMMENDATIONS = (
    "[bright_magenta]Filtering [bold]{start_count}[/bold] stocks for recommendations..."
)
FILTER_PREDICTIONS = "[green1]Predicting {symbol} for future gains, [bold]{start_count}[/bold] stocks left to scan..."
FILTER_NEWS = "[yellow1]Filtering news for {symbol} with Openai..."


def chunk_list(seq: list, size: int) -> list[list]:
    """
    Splits a list into chunks of specified size.

    Args:
        seq (list): The list to be chunked.
        size (int): The size of each chunk.

    Returns:
        list of lists
    """
    return [seq[pos : pos + size] for pos in range(0, len(seq), size)]


def check_for_float_or_int(value: Any) -> bool:
    if isinstance(value, float) or isinstance(value, int):
        return True
    else:
        return False


####################################################################################################
# Add the industry_sorter function to the global_values.py file
####################################################################################################
def industry_sorter_check(sort_by: str) -> str:
    """
    Sorts the industries based on the specified criteria.

    Args:
        sort_by (str): The criteria to sort the industries by. Must be one of the following:
            - "industry"
            - "count"
            - "gross_margin_avg"
            - "net_margin_avg"
            - "trailing_pe_avg"
            - "forward_pe_avg"

    Returns:
        str: The sorted industries based on the specified criteria.

    Raises:
        ValueError: If the specified sort_by criteria is not one of the allowed values.
    """
    allowed_sort_by = [
        "industry",
        "count",
        "gross_margin_avg",
        "net_margin_avg",
        "trailing_pe_avg",
        "forward_pe_avg",
    ]
    if sort_by not in allowed_sort_by:
        raise ValueError(f"sort_by must be one of {allowed_sort_by}")


def stock_sorter_check(sort_by: str) -> str:
    """
    Check if the given sort_by value is allowed.

    Args:
        sort_by (str): The value to be checked.
        Allowed values are "name", "symbol", "gross_margin_pct", "net_margin_pct", "forward_pe_pct", "trailing_pe_pct", "industry".

    Returns:
        str: The sort_by value if it is allowed.

    Raises:
        ValueError: If the sort_by value is not allowed.
    """
    allowed_sort_by = [
        "name",
        "symbol",
        "market_cap",
        "gross_margin_pct",
        "net_margin_pct",
        "forward_pe_pct",
        "trailing_pe_pct",
        "industry",
    ]
    if sort_by not in allowed_sort_by:
        raise ValueError(f"sort_by must be one of {allowed_sort_by}")
    return sort_by
