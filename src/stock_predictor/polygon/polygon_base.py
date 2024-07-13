from stock_predictor.global_settings import POLYGON_API_KEY
from stock_predictor.requests_retry import RequestsRetry as http


class PolygonBase:
    def __init__(self) -> None:
        self.http = http()
        self.polyv2 = "https://api.polygon.io/v2/"
        self.polyv3 = "https://api.polygon.io/v3/"
        self.polyvx = "https://api.polygon.io/vX/"
        self.headers = {"Authorization": f"Bearer {POLYGON_API_KEY}"}

    def request(self, method: str, url: str, params: dict = None, json: dict = None):
        """
        Send a request to the Polygon API.

        Args:
            method (str): The HTTP method to use for the request.
            url (str): The URL to send the request to.
            headers (dict): Headers to include in the request.
            params (dict): Query parameters to include in the request.
            json (dict): JSON payload to include in the request.

        Returns:
            requests.Response: The HTTP response received from the server.
        """
        response = self.http.request(
            method=method, url=url, headers=self.headers, params=params, json=json
        )
        return response.json()
