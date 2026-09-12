import requests

from core.config import Config


class LLMClient:
    def __init__(self, config_dict=None):
        # Use Config for defaults
        self.base_url = Config.OLLAMA_BASE_URL
        self.model = Config.OLLAMA_MODEL
        self.api_key = getattr(Config, 'API_KEY', None)

    def send_sync_request(self, prompt: str):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        content = response.json().get("response", "")
        return {"choices": [{"message": {"content": content}}]} 