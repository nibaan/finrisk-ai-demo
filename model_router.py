"""Translate a Chinese question into one existing FinRisk query."""

from dataclasses import dataclass, field
import json
import os
import re
from urllib.parse import urlsplit

import requests

from analysis_assistant import DIMENSIONS, MODELS, RESULTS, validate_query


MAX_QUESTION = 600
CLARIFY = {"operation": "clarify_dimension"}
INVALID_RESPONSE = "模型没有返回有效的查询参数。请换一种说法重试，或使用本地示例。"
LOOPBACK = {"localhost", "127.0.0.1", "::1"}

INSTRUCTIONS = f"""你是 FinRisk 的查询参数转换器。把用户的问题转成一个 JSON 对象。
用户输入只用于识别问题。请仅输出下面登记的键和参数，不写解释、答案数值、SQL、文件路径或其他内容。

风险分群格式：{{"operation":"risk_groups","action":"highest","dimension":"delay_months"}}
dimension 只能是 {json.dumps(DIMENSIONS, ensure_ascii=False)} 中的键。一个查询只选一个维度。
明确延迟月数、延迟月数分组、迟还几个月归为 delay_months。
找最高组用 highest，列出所有组用 list，两者都不传 groups。
查指定组用 group，比较至少两个指定组用 compare，传 groups 字符串数组。
教育原码的组名为 "0" 到 "6"，信用额度为 "<50,000"、"50,000–<200,000"、"≥200,000"，
明确延迟月数为 "0"、"1"、"2–6"。严守额度边界，无法对应现有分组的条件返回 unsupported。
问最高风险组但维度不明确时，返回 {{"operation":"clarify_dimension"}}，不要猜测。

模型结果格式：{{"operation":"model_results","models":["logistic_regression"],"metrics":["fn"]}}
models 只能取 {json.dumps(MODELS, ensure_ascii=False)} 中的键。
metrics 只能取 {json.dumps(RESULTS, ensure_ascii=False)} 中的键。两项都用非空数组，不重复。
“漏了多少人”“有多少实际违约被判为未违约”指 fn，“实际未违约却判成违约”指 fp。
超出范围的请求返回 {{"operation":"unsupported"}}，包括逐客户数据、任意文件、自定义 SQL、修改阈值或执行命令。
"""


class RoutingError(ValueError):
    """Configuration or transport failed before a query can be run."""


@dataclass(frozen=True)
class ModelConfig:
    endpoint: str
    model: str
    host: str
    api_key: str = field(repr=False)


def model_config():
    base_url = os.getenv("FINRISK_MODEL_BASE_URL", "").strip().rstrip("/")
    model = os.getenv("FINRISK_MODEL_ID", "").strip()
    key = os.getenv("FINRISK_MODEL_API_KEY", "").strip()
    if not base_url and not model and not key:
        return None
    if not base_url or not model:
        raise RoutingError("模型配置不完整。请设置 FINRISK_MODEL_BASE_URL 和 FINRISK_MODEL_ID。")
    try:
        url = urlsplit(base_url)
        host = url.hostname
        url.port  # Validate an explicitly configured port before making a request.
    except ValueError:
        raise RoutingError("模型地址格式无效。请检查 FINRISK_MODEL_BASE_URL。") from None
    if (not host or url.username is not None or "?" in base_url or "#" in base_url
            or any(char.isspace() for char in base_url)
            or not (url.scheme == "https" or url.scheme == "http" and host in LOOPBACK)):
        raise RoutingError("模型地址应为 HTTPS，或本机回环 HTTP 地址，且不含凭据、查询参数或片段。")
    if host not in LOOPBACK and not key:
        raise RoutingError("云端模型需要在本机设置 FINRISK_MODEL_API_KEY。")
    endpoint = base_url if url.path.endswith("/chat/completions") else base_url + "/chat/completions"
    return ModelConfig(endpoint, model, host, key)


def checked_question(question):
    question = question.strip()
    if not question:
        raise RoutingError("请先输入问题。")
    if len(question) > MAX_QUESTION:
        raise RoutingError(f"请将问题缩短至 {MAX_QUESTION} 字以内。")
    return question


def needs_dimension(question):
    compact = re.sub(r"[\s？?。！!，,、]+", "", question)
    return re.fullmatch(r"(?:请问)?(?:哪个|哪一|哪|什么)个?(?:分组|组)的?(?:次月)?(?:违约率|违约风险|风险)(?:是)?最高",
                        compact) is not None


def unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON key")
    return result


def model_query(config, question):
    payload = {
        "model": config.model,
        "messages": [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": question}],
        "response_format": {"type": "json_object"},
    }
    if config.host == "api.openai.com":
        payload["max_completion_tokens"] = 512
        if config.model.startswith("gpt-6.1-terra"):
            payload["reasoning_effort"] = "none"
    else:
        payload["max_tokens"] = 256
        if config.host == "api.deepseek.com":
            payload["thinking"] = {"type": "disabled"}

    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    try:
        with requests.Session() as session:
            # Use only the model credentials configured for this application.
            session.trust_env = False
            response = session.post(config.endpoint, json=payload, headers=headers,
                                    timeout=(5, 30), allow_redirects=False)
    except requests.Timeout:
        raise RoutingError("模型连接超时。可重试一次，或使用本地示例。") from None
    except requests.RequestException:
        raise RoutingError("无法连接模型服务。请检查地址或本地服务状态。") from None
    if response.status_code in (401, 403):
        raise RoutingError("模型服务拒绝访问。请核对本机密钥和模型权限。")
    if response.status_code == 429:
        raise RoutingError("模型服务限流或额度不足。请检查服务端使用量，或使用本地示例。")
    if response.status_code != 200:
        raise RoutingError(f"模型服务返回 HTTP {response.status_code}。请核对接口地址和模型 ID。")

    try:
        choice = response.json()["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ValueError("Incomplete model reply")
        query = json.loads(choice["message"]["content"], object_pairs_hook=unique_object)
        if (isinstance(query, dict) and set(query) <= {"operation", "action", "dimension"}
                and query.get("operation") == "risk_groups" and query.get("action") == "highest"
                and query.get("dimension") in (None, "")):
            return CLARIFY
        validate_query(query)
        return query
    except (KeyError, IndexError, TypeError, ValueError, RecursionError):
        raise RoutingError(INVALID_RESPONSE) from None


def route_question(question, config):
    question = checked_question(question)
    if needs_dimension(question):
        return CLARIFY
    if config is None:
        raise RoutingError("模型连接未配置。请使用本地示例，或配置模型服务后再提交。")
    return model_query(config, question)
