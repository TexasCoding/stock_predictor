import os
from dotenv import load_dotenv
import requests

load_dotenv()


class FmpBase:
    def __init__(self) -> None:
        self.api_key = os.getenv("FMP_API_KEY", "")
        self.v3_base_url = "https://financialmodelingprep.com/api/v3/"
        self.v4_base_url = "https://financialmodelingprep.com/api/v4/"

    def get_request(self, url: str) -> dict:
        """Make a GET request to the given URL.

        Args:
            url (str): The URL to make the GET request to.

        Returns:
            dict: The JSON response from the GET request.
        """
        response = requests.get(url)
        return response.json()
