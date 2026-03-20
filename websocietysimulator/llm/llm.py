import json
import logging
import math
import threading
import time
from typing import Dict, List, Optional, Union

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .infinigence_embeddings import InfinigenceEmbeddings

logger = logging.getLogger("websocietysimulator")


def _sanitize_text(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = value.encode("utf-8", "replace").decode("utf-8")
    return "".join(ch if ch in "\n\r\t" or ord(ch) >= 32 else " " for ch in value)


def _sanitize_jsonable(value):
    if isinstance(value, dict):
        return {_sanitize_text(k): _sanitize_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_jsonable(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


def _build_chat_payload(messages, model, temperature, max_tokens, stop_strs, n):
    payload = {
        "model": model,
        "messages": _sanitize_jsonable(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": n,
    }
    if stop_strs:
        payload["stop"] = _sanitize_jsonable(stop_strs)
    json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str)
    return payload

class LLMBase:
    def __init__(self, model: str = "qwen2.5-72b-instruct"):
        """
        Initialize LLM base class
        
        Args:
            model: Model name, defaults to deepseek-chat
        """
        self.model = model
        
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call LLM to get response
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        raise NotImplementedError("Subclasses need to implement this method")
    
    def get_embedding_model(self):
        """
        Get the embedding model for text embeddings
        
        Returns:
            OpenAIEmbeddings: An instance of OpenAI's text embedding model
        """
        raise NotImplementedError("Subclasses need to implement this method")

class InfinigenceLLM(LLMBase):
    def __init__(self, api_key: str, model: str = "qwen2.5-72b-instruct"):
        """
        Initialize Deepseek LLM
        
        Args:
            api_key: Deepseek API key
            model: Model name, defaults to qwen2.5-72b-instruct
        """
        super().__init__(model)
        self.api_key = api_key
        self.base_url = "https://cloud.infini-ai.com/maas/v1"
        self._thread_local = threading.local()
        self.embedding_model = InfinigenceEmbeddings(api_key=api_key)

    def _get_client(self, refresh: bool = False):
        client = None if refresh else getattr(self._thread_local, "client", None)
        if client is None:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self._thread_local.client = client
        return client
        
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=300),  # 等待时间从10秒开始，指数增长，最长300秒
        stop=stop_after_attempt(10)  # 最多重试10次
    )
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call Infinigence AI API to get response with rate limit handling
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        payload = _build_chat_payload(
            messages=messages,
            model=model or self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_strs=stop_strs,
            n=n,
        )
        try:
            response = self._get_client().chat.completions.create(**payload)
            
            if n == 1:
                return response.choices[0].message.content
            else:
                return [choice.message.content for choice in response.choices]
        except Exception as e:
            if "429" in str(e):
                logger.warning("Rate limit exceeded")
            else:
                logger.error(f"Other LLM Error: {e}")
            raise e
    
    def get_embedding_model(self):
        return self.embedding_model

class OpenAILLM(LLMBase):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI LLM
        
        Args:
            api_key: OpenAI API key
            model: Model name, defaults to gpt-3.5-turbo
        """
        super().__init__(model)
        self.api_key = api_key
        self._thread_local = threading.local()
        self.embedding_model = OpenAIEmbeddings(api_key=api_key)

    def _get_client(self, refresh: bool = False):
        client = None if refresh else getattr(self._thread_local, "client", None)
        if client is None:
            client = OpenAI(api_key=self.api_key)
            self._thread_local.client = client
        return client

    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call OpenAI API to get response
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        payload = _build_chat_payload(
            messages=messages,
            model=model or self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_strs=stop_strs,
            n=n,
        )
        last_error = None
        for attempt in range(3):
            try:
                response = self._get_client(refresh=attempt > 0).chat.completions.create(**payload)
                if n == 1:
                    return response.choices[0].message.content
                return [choice.message.content for choice in response.choices]
            except Exception as e:
                last_error = e
                if "parse the JSON body of your request" in str(e):
                    logger.warning(
                        "OpenAI request body parse failure on attempt %s/3; rebuilding the client and retrying.",
                        attempt + 1,
                    )
                    time.sleep(attempt + 1)
                    continue
                raise
        raise last_error
    
    def get_embedding_model(self):
        return self.embedding_model 
