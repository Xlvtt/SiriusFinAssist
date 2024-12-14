import requests
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class apiResponse:
    def Json(json_message: str):
        json_parsed = json.loads(json_message)
        return apiResponse(success=json_parsed['success'],message=json_parsed['message'], error=json_parsed['error'])

    message: str
    success: bool
    error: Optional[str] = None

class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def send_message(self, message: str) -> apiResponse:
        try:
            print(f"{self.base_url}/chat")
            response = requests.get(
                f"{self.base_url}/chat",
                params = {'message': message},
                headers={
                    'Content-Type' : 'application/json',
                    'Accept' : 'application/json'
                }
            )
            print(f"{response}")
            if response.status_code == 200:
                return apiResponse.Json(response.text)
            else:
                return apiResponse(success=False,message=None, error=f"THHP ERROR: {response.status_code}")
        except requests.exceptions.RequestException as e:
            return apiResponse(success=False,message=None, error=f"THHP ERROR: {str(e)}")
            
        