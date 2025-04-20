import json
import os
from typing import Any, Dict, Optional

import httpx
from google import genai

from error import AuthenticationError, ProviderError

class Gemini:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise AuthenticationError("Gemini", message="Missing Gemini API key")
        self.gemini = genai.Client(api_key=self.api_key)
        self.model = model or "gemini-2.5-pro-preview-03-25"

    def generate(self, message: str, params: Optional[Dict] = None):
        try:
            params = params or {}
            response = self.gemini.models.generate_content(
                model=self.model,
                contents=message,
                **params
            )
            return response
        except Exception as e:
            raise ProviderError("Gemini", message=f"Error generating content: {str(e)}")


class OpenRouter:
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise AuthenticationError("OpenRouter")

        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _make_api_request(self, endpoint: str, params: Dict[str, Any]):
        try:
            with httpx.Client() as client:
                response = client.post(
                    endpoint,
                    headers=self.headers,
                    data=json.dumps(params),
                    timeout=45.0,
                )
                if response.status_code in [401, 403]:
                    raise AuthenticationError(
                        "OpenRouter",
                        message="Invalid API key or unauthorized access",
                        status_code=response.status_code,
                        response=response.json() if response.content else None,
                    )

                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            if e.response.status_code in [401, 403]:
                raise AuthenticationError(
                    "OpenRouter",
                    message="Invalid API key or unauthorized access",
                    status_code=e.response.status_code,
                    response=e.response.json() if e.response.content else None,
                )
            raise ProviderError(
                "OpenRouter",
                message=f"API request failed: {str(e)}",
                status_code=e.response.status_code,
                response=e.response.json() if e.response.content else None,
            )
        except Exception as e:
            raise ProviderError("OpenRouter", message=f"Request failed: {str(e)}")

    def generate(
        self,
        model: str,
        messages: str,
        params: Optional[Dict[str, Any]] = None,
    ):
        try:
            if params is None:
                params = {}

            if params.get("response_format"):
                params["response_format"] = None

            messages = [{"role": "user", "content": messages}]
            body = {"model": model, "messages": messages, **params}
            res = self._make_api_request(endpoint=self.url, params=body)

            return res
        except (AuthenticationError, ProviderError) as e:
            raise e
        except Exception as e:
            raise ProviderError(
                "OpenRouter", message=f"Error generating completion: {str(e)}"
            )
