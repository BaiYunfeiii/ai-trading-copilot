from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Literal, Optional, Sequence, Tuple, Union, Dict, Any


try:
    # 兼容 openai 官方 SDK v1+（OpenAI 类）与 openai>=1.0 的新接口
    from openai import OpenAI  # type: ignore
except Exception as exc:  # pragma: no cover - optional runtime dependency
    OpenAI = None  # type: ignore


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class OpenAIConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # OpenAI-compatible endpoint 基址，例如自建/代理服务
    model: str = "gpt-4o-mini"  # 默认模型，可被调用时覆盖
    timeout: Optional[float] = 60.0


Message = Dict[str, str]


class OpenAIClient:
    """
    对 OpenAI 兼容 Chat Completions API 的轻量封装，支持：
    - 非流式：一次性返回完整文本
    - 流式：增量返回 token 片段（yield）

    通过传入 base_url 可对接 OpenAI-compatible 服务（如本地代理/第三方）。
    """

    def __init__(self, config: Optional[OpenAIConfig] = None) -> None:
        self.config = config or OpenAIConfig()
        if OpenAI is None:  # pragma: no cover
            raise RuntimeError("未安装 openai 包，请先: pip install openai>=1.0.0")
        # 初始化 SDK 客户端
        client_kwargs: Dict[str, Any] = {}
        if self.config.api_key is not None:
            client_kwargs["api_key"] = self.config.api_key
        if self.config.base_url is not None:
            client_kwargs["base_url"] = self.config.base_url

        self._client = OpenAI(**client_kwargs)

    @staticmethod
    def _normalize_messages(messages: Sequence[Message]) -> List[Message]:
        normalized: List[Message] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if not role or not content:
                raise ValueError("每条消息必须包含 'role' 与 'content'")
            normalized.append({"role": role, "content": content})
        return normalized

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Iterator[str]]:
        """
        进行对话。

        参数：
        - messages: 形如 {"role": "user"|"system"|"assistant", "content": "..."}
        - model: 覆盖默认模型
        - temperature/top_p/max_tokens: 采样参数
        - stream: 是否流式返回；若为 True，返回 Iterator[str]
        - extra_params: 传递给底层 API 的附加参数（如 presence_penalty 等）
        """

        payload: Dict[str, Any] = {
            "model": model or self.config.model,
            "messages": self._normalize_messages(messages),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_params:
            payload.update(extra_params)

        if not stream:
            resp = self._client.chat.completions.create(**payload)
            # 兼容标准返回结构 choices[0].message.content
            choices = getattr(resp, "choices", None)
            if not choices:
                return ""
            message = choices[0].message  # type: ignore[attr-defined]
            return getattr(message, "content", "") or ""

        def _streaming() -> Iterator[str]:
            stream_resp = self._client.chat.completions.create(stream=True, **payload)
            for chunk in stream_resp:  # type: ignore[assignment]
                try:
                    delta = chunk.choices[0].delta  # type: ignore[attr-defined]
                    piece = getattr(delta, "content", None)
                    if piece:
                        yield piece
                except Exception:
                    # 忽略不含文本的片段（如 role/tool_calls 等）
                    continue

        return _streaming()


__all__ = [
    "OpenAIClient",
    "OpenAIConfig",
]

