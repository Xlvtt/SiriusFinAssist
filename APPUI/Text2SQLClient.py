import requests

class Text2SQLClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def execute_query(self, query: str):
        url = f"{self.base_url}/text2data"
        try:
            response = requests.post(url, json={"query": query})
            response.raise_for_status()
            return response.json()["data"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ошибка при вызове API: {e}")