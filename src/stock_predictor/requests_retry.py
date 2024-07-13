# src/stock_trader/http_module/requests_retry.py
# This module contains the RequestsRetry class, which is used to send HTTP requests with retry logic.
from typing import Dict, Optional, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


class RequestsRetry:
    def __init__(
        self, total_retries: int = 3, backoff_factor: float = 2.0, status_forcelist=None
    ) -> None:
        """
        Initializes the RequestsRetry class with a retry strategy.

        Args:
            total_retries (int): The total number of retry attempts. Defaults to 3.
            backoff_factor (float): A backoff factor to apply between attempts after the second try. Defaults to 2.0.
            status_forcelist (list): A list of HTTP status codes that we should force a retry on. Defaults to [429, 500, 502, 503, 504].
        """
        status_forcelist = status_forcelist or [429, 500, 502, 503, 504]

        self.retry_strategy = Retry(
            total=total_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
        )
        self.adapter = HTTPAdapter(max_retries=self.retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", self.adapter)
        self.session.mount("http://", self.adapter)

    ############################
    # Send HTTP request
    ############################
    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Union[str, bool, float]]] = None,
        json: Optional[Dict[str, Union[str, bool, float]]] = None,
    ) -> requests.Response:
        """
        Sends a HTTP request with retry logic.

        Args:
            method (str): The HTTP method to use for the request.
            url (str): The URL to send the request to.
            headers (Optional[Dict[str, str]]): Headers to include in the request.
            params (Optional[Dict[str, Union[str, bool, float]]]): Query parameters to include in the request.
            json (Optional[Dict[str, Union[str, bool, float]]]): JSON payload to include in the request.

        Returns:
            requests.Response: The HTTP response received from the server.

        Raises:
            requests.HTTPError: If the response status code is not one of the acceptable statuses (200, 204, 207).
        """
        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json,
            )
            response.raise_for_status()  # Raise an HTTPError if the response status is 4xx, 5xx
        except requests.RequestException as e:
            raise requests.HTTPError(f"Request failed: {e}") from e

        return response
