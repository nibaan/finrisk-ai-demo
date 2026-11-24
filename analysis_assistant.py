"""Validate query parameters and answer from three fixed summary files."""

import csv
from dataclasses import dataclass
from fractions import Fraction
import json
from math import isclose


DIMENSIONS = {"education": "教育原码", "credit_limit": "信用额度", "delay_months": "明确延迟月份数"}
MODELS = {"dummy_prior": "Dummy 基准", "logistic_regression": "逻辑回归"}
METRICS = {"roc_auc": "ROC-AUC", "average_precision": "AP", "precision": "Precision", "recall": "Recall", "f1": "F1"}
RESULTS = {**METRICS, "fn": "漏判", "fp": "误报"}
GROUPS_FILE = "data/processed/g3_results/train_risk_groups.csv"
METRICS_FILE = "data/processed/g4_results/metrics.json"
MATRIX_FILE = "data/processed/g4_results/confusion_matrix.csv"
SUPPORTED = "支持按教育原码、信用额度或明确延迟月数查询风险分群，以及查询 Dummy 与逻辑回归的测试结果。"

EXAMPLES = (
    ("按明确延迟月数分组，哪个组违约率最高？",
     {"operation": "risk_groups", "action": "highest", "dimension": "delay_months"}),
    ("低于 50,000 新台币的客户违约率是多少？",
     {"operation": "risk_groups", "action": "group", "dimension": "credit_limit", "groups": ["<50,000"]}),
    ("教育原码 0 的风险怎么样？",
     {"operation": "risk_groups", "action": "group", "dimension": "education", "groups": ["0"]}),
    ("逻辑回归漏了多少人？又把多少实际未违约客户判成了违约？",
     {"operation": "model_results", "models": ["logistic_regression"], "metrics": ["fn", "fp"]}),
    ("两个模型的 ROC-AUC 和 AP 分别是多少？",
     {"operation": "model_results", "models": list(MODELS), "metrics": ["roc_auc", "average_precision"]}),
)


class QueryError(ValueError):
    """A user-facing error that leaves the query without an answer."""


@dataclass(frozen=True)
class Answer:
    text: str
    sources: tuple[str, ...] = ()


def allowed_items(values, allowed):
    return (type(values) is list and bool(values)
            and all(type(value) is str and value in allowed for value in values)
            and len(set(values)) == len(values))


def validate_query(query):
    if type(query) is not dict or type(query.get("operation")) is not str:
        raise QueryError(SUPPORTED)
    operation = query["operation"]
    if operation == "risk_groups":
        action = query.get("action")
        if action not in ("list", "group", "compare", "highest"):
            raise QueryError(SUPPORTED)
        expected = {"operation", "action", "dimension"}
        if action in ("group", "compare"):
            expected.add("groups")
            groups = query.get("groups")
            if (type(groups) is not list or not 1 <= len(groups) <= 13
                    or any(type(name) is not str or not 1 <= len(name) <= 64
                           or name != name.strip() or any(c in name for c in ('/', '\\', ':', '\n', '\r'))
                           or ".." in name for name in groups)
                    or len(set(groups)) != len(groups)
                    or (action == "group" and len(groups) != 1)
                    or (action == "compare" and len(groups) < 2)):
                raise QueryError(SUPPORTED)
        if set(query) - expected:
            raise QueryError(SUPPORTED)
        dimension = query.get("dimension")
        if dimension is None or dimension == "":
            raise QueryError("请先选择一个分组维度。可选教育原码、信用额度、明确延迟月数。")
        if type(dimension) is not str or dimension not in DIMENSIONS:
            raise QueryError(SUPPORTED)
    elif operation == "model_results":
        if (set(query) != {"operation", "models", "metrics"}
                or not allowed_items(query.get("models"), MODELS)
                or not allowed_items(query.get("metrics"), RESULTS)):
            raise QueryError(SUPPORTED)
    elif operation in ("clarify_dimension", "unsupported") and set(query) == {"operation"}:
        return
    else:
        raise QueryError(SUPPORTED)


def read_rows(root, source, fields):
    try:
        with (root / source).open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, strict=True)
            header = reader.fieldnames or []
            if not set(fields).issubset(header) or len(header) != len(set(header)):
                raise QueryError(f"来源不可用。{source} 缺少所需字段或表头重复。")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error):
        raise QueryError(f"来源不可用。无法读取 {source}。") from None
    if not rows:
        raise QueryError(f"来源不可用。{source} 是空表。")
    if any(None in row or any(row[field] is None for field in fields) for row in rows):
        raise QueryError(f"来源不可用。{source} 存在缺字段的记录。")
    return rows


def risk_answer(root, query):
    dimension = query["dimension"]
    label = DIMENSIONS[dimension]
    rows = read_rows(root, GROUPS_FILE, ("dimension", "group", "valid_label_customers", "events", "event_rate", "small_group"))
    groups = {}
    for row in rows:
        if row["dimension"] != dimension:
            continue
        group = row["group"]
        if group in groups:
            raise QueryError(f"来源数据需核对。{label}存在重复分组「{group}」。")
        try:
            customers, events = int(row["valid_label_customers"]), int(row["events"])
            small = row["small_group"]
            if not group or not 0 <= events <= customers or small not in ("True", "False"):
                raise ValueError
            rate = Fraction(events, customers) if customers else None
            if rate is not None and not isclose(float(row["event_rate"]), float(rate), rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError
        except ValueError:
            raise QueryError(f"来源数据需核对。{label}组「{group}」的人数、事件率或小样本标记无效。") from None
        groups[group] = (customers, events, rate, small == "True")

    if not groups:
        raise QueryError(f"没有匹配的{label}分组。")
    selected = query.get("groups", list(groups))
    for name in selected:
        if name not in groups:
            raise QueryError(f"没有匹配的{label}分组「{name}」。")
    action = query["action"]
    heading = f"冻结训练集（{label}分组汇总）共 {sum(v[0] for v in groups.values()):,} 位有效标签客户，" \
              f"其中 {sum(v[1] for v in groups.values()):,} 人次月违约。"
    sortable = [name for name in selected if groups[name][2] is not None]
    unavailable = [name for name in selected if groups[name][2] is None]
    if action in ("compare", "highest"):
        sortable.sort(key=lambda name: groups[name][2], reverse=True)
        selected = sortable + unavailable
    if action == "highest":
        if sortable:
            best = groups[sortable[0]][2]
            selected = [name for name in sortable if groups[name][2] == best] + unavailable
            heading += "\n以下为违约率最高的分组。"
            if unavailable:
                heading += "分母为零的分组另列。"
        else:
            heading += "\n所有组的分母为零，无法比较最高违约率。"
    lines = [heading]
    for name in selected:
        customers, events, rate, small = groups[name]
        if dimension == "delay_months":
            group_label = f"{name} 个月组"
        elif dimension == "education":
            group_label = f"教育原码 {name} 组"
        else:
            group_label = f"信用额度 {name} 新台币组"
        rate_text = f"事件率 {float(rate):.2%}（{events:,} ÷ {customers:,}）" if rate is not None else "分母为零，事件率无法计算"
        note = "。小样本，观察结果不宜外推" if small else ""
        lines.append(f"{group_label}：{customers:,} 位有效标签客户，观察到 {events:,} 人违约，{rate_text}{note}。")
    if dimension == "credit_limit" and "<50,000" in selected and "50,000–<200,000" in groups:
        lines.append("低额度组严格小于 50,000，额度恰好 50,000 的客户属于下一组。")
    return Answer("\n".join(lines), (GROUPS_FILE,))


def required(mapping, key):
    if not isinstance(mapping, dict) or key not in mapping:
        raise QueryError(f"来源不可用。{METRICS_FILE} 缺少字段 {key}。")
    return mapping[key]


def model_answer(root, query):
    try:
        raw = json.loads((root / METRICS_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise QueryError(f"来源不可用。无法读取 {METRICS_FILE}。") from None
    rows = read_rows(root, MATRIX_FILE, ("model", "tn", "fp", "fn", "tp"))
    models = required(raw, "models")
    n = required(required(raw, "split_counts"), "test")
    threshold = required(raw, "threshold")
    zero = required(raw, "zero_division")
    if (required(raw, "test_split") != "test" or type(n) is not int or n <= 0
            or type(threshold) not in (int, float) or not 0 <= threshold <= 1 or zero != 0):
        raise QueryError("来源数据需核对。冻结测试集、阈值或零分母口径无效。")
    summaries = []
    actual_positives = None
    for name in query["models"]:
        metrics = required(models, name)
        matches = [row for row in rows if row["model"] == name]
        if not matches:
            raise QueryError(f"来源不可用。没有 {MODELS[name]} 的混淆矩阵。")
        if len(matches) > 1:
            raise QueryError(f"来源数据需核对。{MODELS[name]} 的混淆矩阵重复。")
        try:
            tn, fp, fn, tp = counts = [int(matches[0][key]) for key in ("tn", "fp", "fn", "tp")]
        except ValueError:
            raise QueryError(f"来源数据需核对。{MODELS[name]} 的混淆矩阵包含无效人数。") from None
        actual, predicted = fn + tp, fp + tp
        reported_counts = [required(metrics, key) for key in ("test_customers", "actual_positives", "predicted_positives")]
        if (min(counts) < 0 or any(type(count) is not int for count in reported_counts)
                or sum(counts) != n or [n, actual, predicted] != reported_counts
                or (actual_positives is not None and actual != actual_positives)):
            raise QueryError(f"来源数据需核对。{MODELS[name]} 的矩阵总人数、实际正类或预测正类与测试指标冲突。")
        actual_positives = actual
        saved = {key: required(metrics, key) for key in METRICS}
        derived = {"precision": tp / predicted if predicted else zero,
                   "recall": tp / actual if actual else zero,
                   "f1": 2 * tp / (2 * tp + fp + fn) if tp or fp or fn else zero}
        if (any(type(value) not in (int, float) or not 0 <= value <= 1 for value in saved.values())
                or any(not isclose(saved[key], value, rel_tol=1e-12, abs_tol=1e-15) for key, value in derived.items())):
            raise QueryError(f"来源数据需核对。{MODELS[name]} 的指标无效或与混淆矩阵冲突。")
        formatted = {key: f"{value:.4f}" for key, value in saved.items()}
        formatted.update(fn=f"{fn:,} 人", fp=f"{fp:,} 人")
        summary = "，".join(f"{RESULTS[key]} {formatted[key]}" for key in query["metrics"])
        note = "。预测正类为 0，Precision 按零分母规则记为 0" if not predicted and "precision" in query["metrics"] else ""
        summaries.append(f"{MODELS[name]}：{summary}{note}。")
    lines = [f"冻结测试集共 {n:,} 人，实际次月违约 {actual_positives:,} 人。", *summaries,
             f"分类阈值 {threshold:g}，用于人数、Precision、Recall 和 F1。ROC-AUC 和 AP 使用连续预测分数。"]
    return Answer("\n".join(lines), (METRICS_FILE, MATRIX_FILE))


def answer_query(root, query):
    validate_query(query)
    operation = query["operation"]
    if operation == "risk_groups":
        return risk_answer(root, query)
    if operation == "model_results":
        return model_answer(root, query)
    if operation == "clarify_dimension":
        raise QueryError("请先选择一个分组维度。可选教育原码、信用额度、明确延迟月数。")
    raise QueryError(SUPPORTED)
