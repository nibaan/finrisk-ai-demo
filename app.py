"""Read the verified G1–G4 artifacts into the FinRisk AI local dashboard."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis_assistant import DIMENSIONS, EXAMPLES, METRICS, MODELS, QueryError, answer_query
from model_router import CLARIFY, MAX_QUESTION, RoutingError, model_config, route_question

ROOT = Path(__file__).resolve().parent
DATA = Path("data/processed")
G1 = DATA / "uci_g1_check.json"
G2 = DATA / "g2_results/quality.json"
G3 = DATA / "g3_results"
G4 = DATA / "g4_results"
LABEL = "default payment next month"
CORE_FIELDS = {"ID": "客户 ID", "LIMIT_BAL": "信用额度", "SEX": "性别", "EDUCATION": "教育",
               "MARRIAGE": "婚姻", "AGE": "年龄", "PAY_0": "最近月还款状态",
               "BILL_AMT1": "最近月账单", "PAY_AMT1": "最近月前期付款", LABEL: "次月违约标签"}
TEAL, BLUE, AMBER, SLATE = "#08786B", "#275994", "#B05D0E", "#637993"


class ArtifactError(ValueError):
    pass


class NoData(ValueError):
    pass


@st.cache_data(max_entries=24, show_spinner=False)
def read_file(path, modified_ns):
    if path.suffix == ".csv":
        return pd.read_csv(path)
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if path.suffix == ".json" else text


def artifact(relative, *fields):
    path = ROOT / relative
    label = relative.as_posix()
    try:
        value = read_file(path, path.stat().st_mtime_ns)
    except FileNotFoundError as error:
        raise ArtifactError(f"缺少文件 `{label}`。请按项目交接重新生成对应阶段的产物。") from error
    except pd.errors.EmptyDataError as error:
        raise NoData(f"暂无数据。`{label}` 没有记录。") from error
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        raise ArtifactError(f"无法读取 `{label}`，请检查文件格式或重新生成。") from error
    if (value.empty if isinstance(value, pd.DataFrame) else not value):
        raise NoData(f"暂无数据。`{label}` 没有记录。")
    for field in fields:
        current = value
        for key in field.split("."):
            if not isinstance(current, (dict, pd.DataFrame)) or key not in current:
                raise ArtifactError(f"`{label}` 缺少字段 `{field}`。请重新生成对应阶段的产物。")
            current = current[key]
    return value


def chart(figure, key, alt, *, height=310, **layout):
    figure.update_layout(
        height=height, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#14263C", size=12),
        margin=dict(l=8, r=16, t=10, b=8),
        legend=dict(orientation="h", y=-0.20), **layout,
    )
    st.plotly_chart(figure, key=key, theme=None, alt=alt,
                    config={"displayModeBar": False, "scrollZoom": False})


def overview():
    raw = artifact(G1, "data_rows", "data_columns", "id_unique_count", "duplicate_id_rows", "id_missing_count",
                   "target_positive_rate", "monthly_rows_per_customer_counts", "category_counts." + LABEL + ".0",
                   "category_counts." + LABEL + ".1", *("missing_by_column." + name for name in CORE_FIELDS))
    db = artifact(G2, "master_rows", "history_rows", "months_per_customer", "month_mapping", "unexplained_codes",
                  "query_row_counts.q4", "q4_rows_ids_labels_preserved", "negative_bill_rows", "source_citation")
    dictionary = artifact(Path("G2_DATA_DICTIONARY.md"))
    n = raw["data_rows"]
    if not n:
        raise NoData("暂无数据。G1 质量记录中没有客户。")
    labels = raw["category_counts"][LABEL]
    missing = raw["missing_by_column"]
    st.caption(f"全体客户 · 2005 年台湾信用卡样本 · G1 原始数据与 G2 SQLite 质量记录 · {n:,} 人")
    for column, name, value in zip(st.columns(3), ("全体客户", "次月违约客户", "次月违约事件率"),
                                   (f"{n:,}", f"{labels['1']:,}", f"{raw['target_positive_rate']:.2%}")):
        column.metric(name, value, border=True)

    quality, target = st.columns(2, gap="large")
    with quality:
        st.subheader("核心字段缺失率")
        fields = [f"{name} · {CORE_FIELDS[name]}" for name in CORE_FIELDS]
        rates = [missing[name] / n for name in CORE_FIELDS]
        figure = go.Figure(go.Bar(x=rates, y=fields, orientation="h", text=[f"{rate:.2%}" for rate in rates],
                                  textposition="outside", cliponaxis=False, marker_color=TEAL,
                                  hovertemplate="%{y} · %{x:.2%}<extra></extra>"))
        figure.update_xaxes(range=[0, max(0.01, max(rates) * 1.22)], tickformat=".1%", title="各字段缺失数 ÷ 全体客户数")
        figure.update_yaxes(autorange="reversed", automargin=True)
        chart(figure, "field_missing", "G1 原始数据核心字段缺失率", height=360)
    with target:
        st.subheader("次月违约标签分布")
        figure = go.Figure(go.Bar(x=["0 · 未违约", "1 · 次月违约"], y=[labels["0"], labels["1"]],
                                  texttemplate="%{y:,}", textposition="outside", cliponaxis=False,
                                  marker_color=[BLUE, AMBER], hovertemplate="%{x} · %{y:,} 人<extra></extra>"))
        figure.update_yaxes(title="客户数", range=[0, max(labels.values()) * 1.15])
        chart(figure, "label_distribution", "G1 全体客户的正负标签计数", height=327)
        invalid = n - labels["0"] - labels["1"] - missing[LABEL]
        st.caption(f"有效标签 {labels['0'] + labels['1']:,} 人 · 缺失 {missing[LABEL]:,} 人 · 异常 {invalid:,} 人")

    with st.expander("核心字段缺失明细"):
        st.dataframe(pd.DataFrame({"原始字段": list(CORE_FIELDS), "缺失客户数": [missing[k] for k in CORE_FIELDS],
                                   "缺失率": [f"{value:.2%}" for value in rates]}), hide_index=True)

    st.subheader("两表关联与历史覆盖")
    months = db["months_per_customer"]
    covered = raw["monthly_rows_per_customer_counts"].get(str(months), 0)
    st.markdown(f"客户主表 **{db['master_rows']:,} 行** · 月度历史表 **{db['history_rows']:,} 行** · "
                f"完整覆盖 {months} 个月 **{covered:,}／{n:,} 人（{covered / n:.2%}）**")
    joined = db["query_row_counts"]["q4"]
    join_check = "行数、ID 与标签保持一致" if db["q4_rows_ids_labels_preserved"] else "行数、ID 或标签一致性检查未通过"
    history_months = "、".join(row["month"] for row in db["month_mapping"] if "month" in row)
    st.caption(f"历史月份 {history_months}。月表先按客户聚合再左关联，Q4 返回 {joined:,} 行，{join_check}。"
               f"G1 核对唯一 ID {raw['id_unique_count']:,} 个，重复 ID 行 {raw['duplicate_id_rows']:,}，"
               f"缺失 ID 行 {raw['id_missing_count']:,}。")

    st.subheader("未解释编码")
    fields_zh = {"education": "教育", "marriage": "婚姻", "repayment_status": "还款状态"}
    codes = [{"字段": fields_zh.get(field, field), "原码": code, "记录数": count,
              "单位": "客户月" if field == "repayment_status" else "客户"}
             for field, counts in db["unexplained_codes"].items() for code, count in counts.items()]
    if codes:
        st.dataframe(pd.DataFrame(codes), hide_index=True)
    else:
        st.info("暂无未解释编码记录。")
    st.caption(f"未解释编码保留原值并单列。原始 XLS 保留，项目处理包括列名规范化、按月宽转长、按客户聚合与划分。"
               f"{db['negative_bill_rows']:,} 条负账单月记录保留原值。")
    st.markdown(f"{db['source_citation']} · [UCI 数据来源](https://doi.org/10.24432/C55S3H) · "
                "[CC BY 4.0 许可](https://creativecommons.org/licenses/by/4.0/)")
    with st.expander("G2 数据字典与月份映射"):
        st.markdown(dictionary)


def risk():
    quality = artifact(G3 / "quality.json", "split_counts.train", "training_events")
    groups = artifact(G3 / "train_risk_groups.csv", "dimension", "group", "valid_label_customers", "events",
                      "event_rate", "customer_share", "small_group")
    history = artifact(G3 / "train_history_distribution.csv", "month_count", "explained_delay_customers", "unexplained_status_customers")
    train_n = quality["split_counts"]["train"]
    st.caption(f"冻结训练集 · {train_n:,} 位客户 · {quality['training_events']:,} 个次月违约事件 · G3 客户级统计")
    scope = st.segmented_control("分群维度", list(DIMENSIONS), default="education", required=True,
                                 format_func=DIMENSIONS.__getitem__, key="risk_dimension")
    selected = groups.loc[groups.dimension.eq(scope)]
    group_col, history_col = st.columns(2, gap="large")
    with group_col:
        st.subheader("分群次月违约事件率")
        valid = selected.loc[selected.valid_label_customers.gt(0) & selected.event_rate.notna()]
        if valid.empty:
            st.info("所选分群维度暂无数据。")
        else:
            figure = go.Figure(go.Bar(x=valid.group, y=valid.event_rate,
                                      texttemplate="%{y:.2%}", textposition="outside", cliponaxis=False,
                                      marker_color=[AMBER if small else TEAL for small in valid.small_group],
                                      customdata=valid[["valid_label_customers", "events"]],
                                      hovertemplate="%{x}<br>%{y:.2%} · %{customdata[1]:,}／%{customdata[0]:,} 人<extra></extra>"))
            figure.update_xaxes(type="category", title=DIMENSIONS[scope], automargin=True)
            figure.update_yaxes(tickformat=".0%", range=[0, max(0.10, min(1.12, valid.event_rate.max() * 1.18))])
            chart(figure, "group_rate", "训练客户分组的次月违约事件数除以有效标签客户数")
            st.caption("事件率 = 组内次月违约人数 ÷ 有效标签客户数。橙色标记少于 30 人的非空组。")
    with history_col:
        st.subheader("历史还款状态分布")
        figure = go.Figure()
        for field, name, color in (("explained_delay_customers", "明确延迟（1–9）", TEAL),
                                   ("unexplained_status_customers", "未解释状态（-2／0）", SLATE)):
            figure.add_bar(x=history.month_count, y=history[field], name=name, marker_color=color,
                           hovertemplate="%{x} 个月 · %{y:,} 人")
        figure.update_xaxes(title="六个月观察窗内的月份数", dtick=1)
        figure.update_yaxes(title="客户数")
        chart(figure, "history_distribution", "训练客户中明确延迟和未解释状态的月份数分布", barmode="group")
        st.caption("两条分布分别统计同一批训练客户。0 个月明确延迟只表示没有记录到状态 1–9。")

    if not selected.empty:
        st.dataframe(pd.DataFrame({"分组": selected.group, "有效标签客户数": selected.valid_label_customers,
                                   "事件数": selected.events, "事件率": selected.event_rate.map(rate_text),
                                   "训练客户占比": selected.customer_share.map(rate_text),
                                   "样本提示": selected.small_group.map({True: "少于 30 人", False: ""})}), hide_index=True)
    comparable = valid.loc[~valid.small_group]
    if len(comparable) > 1:
        lowest = comparable.loc[comparable.event_rate.idxmin()]
        highest = comparable.loc[comparable.event_rate.idxmax()]
        if lowest.event_rate < highest.event_rate:
            st.markdown(f"在至少 30 人的组中，**{highest.group}** 组事件率为 **{highest.event_rate:.2%}**，"
                        f"**{lowest.group}** 组为 **{lowest.event_rate:.2%}**。差异限于该历史训练样本。")


def rate_text(value):
    return "无数据" if pd.isna(value) else f"{value:.2%}"


def model():
    metric_fields = [f"models.{name}.{field}" for name in MODELS
                     for field in (*METRICS, "test_customers", "actual_positives", "no_predicted_positives")]
    result = artifact(G4 / "metrics.json", "test_split", "threshold", "zero_division", "dummy_prior.training_prior_positive",
                      "logistic_regression.reference_categories", *metric_fields)
    roc = artifact(G4 / "roc_curve.csv", "model", "fpr", "tpr")
    matrix = artifact(G4 / "confusion_matrix.csv", "model", "tn", "fp", "fn", "tp")
    coefficients = artifact(G4 / "lr_coefficients.csv", "encoded_feature", "source_feature", "feature_type", "category",
                            "reference_category", "training_scale", "coefficient", "odds_ratio")
    metrics = result["models"]
    n, positives = (metrics["logistic_regression"][key] for key in ("test_customers", "actual_positives"))
    threshold = result["threshold"]
    if not n:
        raise NoData("暂无数据。G4 固定测试集没有客户记录。")
    for name in MODELS:
        if roc.loc[roc.model.eq(name)].empty or matrix.loc[matrix.model.eq(name)].empty:
            raise NoData(f"暂无数据。G4 缺少 {MODELS[name]} 的 ROC 或混淆矩阵记录。")
        current = matrix.loc[matrix.model.eq(name), ["tn", "fp", "fn", "tp"]]
        if (len(current) != 1 or current.sum(axis=1).iloc[0] != n or
                current[["fn", "tp"]].sum(axis=1).iloc[0] != positives or
                metrics[name]["test_customers"] != n or metrics[name]["actual_positives"] != positives or result["test_split"] != "test"):
            raise ArtifactError("G4 固定测试集与模型指标或混淆矩阵的样本数不一致。请重新验证 G4 产物。")

    st.caption("冻结测试集 · G4 留出评价 · 正类为次月违约还款")
    for column, label, value in zip(st.columns(3), ("测试客户", "实际正类", "固定阈值"),
                                    (f"{n:,}", f"{positives:,}（{positives / n:.2%}）", f"{threshold:g}")):
        column.metric(label, value, border=True)

    roc_col, matrix_col = st.columns(2, gap="large")
    with roc_col:
        st.subheader("ROC 曲线")
        figure = go.Figure()
        for (name, label), color in zip(MODELS.items(), (SLATE, TEAL)):
            data = roc.loc[roc.model.eq(name)]
            figure.add_scatter(x=data.fpr, y=data.tpr, mode="lines", name=f"{label} · {metrics[name]['roc_auc']:.4f}",
                               line=dict(color=color, width=2.5, dash="dash" if name == "dummy_prior" else "solid"),
                               hovertemplate="FPR %{x:.2%}<br>TPR %{y:.2%}")
        figure.update_xaxes(title="假正例率（FPR）", range=[0, 1.01], tickformat=".0%")
        figure.update_yaxes(title="真正例率（TPR）", range=[0, 1.02], tickformat=".0%")
        chart(figure, "roc_curves", "同一固定测试集的两条 ROC 曲线", height=320)
    with matrix_col:
        st.subheader("混淆矩阵")
        name = st.segmented_control("矩阵模型", list(MODELS), default="logistic_regression", required=True,
                                    format_func=MODELS.__getitem__, key="matrix_model", label_visibility="collapsed")
        active = matrix.loc[matrix.model.eq(name)].iloc[0]
        counts = [[active.tn, active.fp], [active.fn, active.tp]]
        figure = go.Figure(go.Heatmap(
            x=["0 · 未违约", "1 · 违约"], y=["0 · 未违约", "1 · 违约"], z=counts, zmin=0, zmax=n,
            text=[[f"TN · {active.tn:,}", f"FP · {active.fp:,}"], [f"FN · {active.fn:,}", f"TP · {active.tp:,}"]],
            texttemplate="%{text}", colorscale=[[0, "#ECF4F5"], [1, "#55B3A4"]], showscale=False,
            hovertemplate="真实 %{y}<br>预测 %{x}<br>%{z:,} 人<extra></extra>",
        ))
        figure.update_xaxes(title="预测类别")
        figure.update_yaxes(title="真实类别", autorange="reversed", automargin=True)
        chart(figure, "confusion_matrix", f"{MODELS[name]} 混淆矩阵，TN {active.tn:,}、FP {active.fp:,}、FN {active.fn:,}、TP {active.tp:,}", height=270)
        st.caption(f"{MODELS[name]} · 合计 {sum(map(sum, counts)):,} 人 · 阈值 {threshold:g}")

    st.subheader("模型评价指标对比")
    figure = go.Figure()
    for (name, label), color in zip(MODELS.items(), (SLATE, TEAL)):
        figure.add_bar(x=list(METRICS.values()), y=[metrics[name][field] for field in METRICS], name=label,
                       marker_color=color, texttemplate="%{y:.4f}", textposition="outside", cliponaxis=False,
                       hovertemplate="%{x} · %{y:.4f}")
    figure.update_yaxes(range=[0, 1.08], tickformat=".0%")
    chart(figure, "metric_comparison", "Dummy 与逻辑回归的五项测试集评价指标", height=288, barmode="group")
    table = [{"模型": label, **{column: f"{metrics[name][key]:.4f}" for key, column in METRICS.items()}}
             for name, label in MODELS.items()]
    st.dataframe(pd.DataFrame(table), hide_index=True)

    if metrics["dummy_prior"]["no_predicted_positives"]:
        prior = result["dummy_prior"]["training_prior_positive"]
        st.info(f"Dummy 给每位客户相同的训练集正类先验 {prior:.4f}，低于 {threshold:g}，因此全判为负类。"
                f"Precision 的分母为零，按项目规则记为 {result['zero_division']}，Recall 和 F1 均为 0。")
    logistic = matrix.loc[matrix.model.eq("logistic_regression")].iloc[0]
    st.markdown(f"**逻辑回归识别出 {logistic.tp:,} 个真实正类，漏掉 {logistic.fn:,} 个。** "
                f"在 {positives:,} 个实际正类中，召回率为 {metrics['logistic_regression']['recall']:.2%}。")
    st.caption("ROC-AUC 与 AP 使用连续得分。AP 指 average precision。Precision、Recall、F1 按固定阈值计算。")

    st.subheader("逻辑回归系数")
    st.markdown("其他特征固定时，数值系数对应增加一个训练标准差，类别系数与参考原码比较。"
                "正系数表示模型对数胜算较高，负系数表示较低。")
    example = coefficients.loc[coefficients.source_feature.eq("explained_delay_months") & coefficients.feature_type.eq("numeric")]
    if not example.empty:
        row = example.iloc[0]
        st.markdown(f"明确延迟月份数增加约 **{row.training_scale:.2f} 个月**，模型对数胜算变化 **{row.coefficient:+.4f}**，"
                    f"对应胜算乘 **{row.odds_ratio:.2f}**。")
    field_names = {"sex": "性别", "education": "教育", "marriage": "婚姻", "latest_repayment_status": "最近月还款状态"}
    refs = result["logistic_regression"]["reference_categories"]
    st.caption("参考原码 · " + " · ".join(f"{field_names.get(key, key)} {code}" for key, code in refs.items()))
    with st.expander("查看编码后的系数明细"):
        st.dataframe(coefficients[["encoded_feature", "source_feature", "category", "reference_category", "coefficient", "odds_ratio"]],
                     hide_index=True, column_config={"encoded_feature": "编码列", "source_feature": "原始字段", "category": "原码",
                                                    "reference_category": "参考原码", "coefficient": st.column_config.NumberColumn("系数", format="%.4f"),
                                                    "odds_ratio": st.column_config.NumberColumn("胜算比", format="%.4f")})
    st.caption("使用范围 · 2005 年台湾信用卡历史样本的随机客户留出。得分未经校准，跨时期和人群表现有待验证。"
               "系数用于解释模型中的条件关联。")


def assistant():
    with st.container(key="finrisk_assistant_header", horizontal=True, wrap=False,
                      horizontal_alignment="distribute", vertical_alignment="center"):
        st.subheader("AI 分析助手", anchor=False)
        st.button("关闭", key="assistant_close", icon=":material/close:",
                  on_click=set_assistant_open, args=(False,))
    body = st.container(key="finrisk_assistant_body", gap="medium")
    with st.container(key="finrisk_assistant_composer"):
        with st.form("assistant_question_form", border=False):
            question = st.text_input("用中文提问", key="assistant_question", max_chars=MAX_QUESTION,
                                     placeholder="例如：逻辑回归漏掉了多少人？")
            submitted = st.form_submit_button("提交问题", key="assistant_ask", type="primary")

    with body:
        assistant_body(question, submitted)


def assistant_body(question, submitted):
    config = None
    try:
        config = model_config()
    except RoutingError as error:
        st.warning(str(error))
    else:
        if config is None:
            st.info("模型连接未配置。本地示例可直接查询。")
        else:
            st.caption(f"已配置模型 · {config.model} · 提交时发送问题与查询说明")

    example = st.selectbox("本地示例问题", range(len(EXAMPLES)),
                           format_func=lambda index: EXAMPLES[index][0], key="assistant_example")
    example_selected = st.button("查询本地结果", key="assistant_run")
    query = None
    if submitted or example_selected:
        st.session_state.pop("assistant_result", None)
        st.session_state.pop("assistant_dimension", None)
        st.session_state["assistant_clarify"] = False
        try:
            if example_selected:
                query, source = EXAMPLES[example][1], "本地示例"
            else:
                query = route_question(question, config)
                if query == CLARIFY:
                    st.session_state["assistant_clarify"] = True
                    query = None
                else:
                    source = f"模型识别（{config.model}） · 本地取数"
        except RoutingError as error:
            st.warning(str(error))

    if st.session_state.get("assistant_clarify"):
        st.info("请先选择一个分组维度，再比较违约率。")
        dimension = st.selectbox("选择分组维度", list(DIMENSIONS), index=None, placeholder="选择一个维度",
                                 format_func=DIMENSIONS.__getitem__, key="assistant_dimension")
        st.button("按此维度查询", key="assistant_clarify_run", disabled=dimension is None,
                  on_click=apply_assistant_dimension)

    if query is not None:
        st.session_state["assistant_result"] = resolve_assistant(query, source)
    if "assistant_result" in st.session_state:
        answer, source = st.session_state["assistant_result"]
        if answer is None:
            st.warning(source)
        else:
            with st.container(key="finrisk_assistant_answer"):
                st.caption(f"回答来源 · {source}")
                for line in answer.text.splitlines():
                    st.text(line)
                for path in answer.sources:
                    st.caption(f"来源文件 · {path}")


def resolve_assistant(query, source):
    try:
        return answer_query(ROOT, query), source
    except QueryError as error:
        return None, str(error)


def apply_assistant_dimension():
    query = {"operation": "risk_groups", "action": "highest", "dimension": st.session_state["assistant_dimension"]}
    st.session_state["assistant_result"] = resolve_assistant(query, "按所选维度本地查询")
    st.session_state["assistant_clarify"] = False


def set_assistant_open(is_open):
    st.session_state["assistant_open"] = is_open


@st.fragment
def assistant_entry():
    if st.session_state.get("assistant_open"):
        with st.container(key="finrisk_assistant_drawer", gap=None):
            assistant()
    else:
        with st.container(key="finrisk_assistant_launcher"):
            st.button("AI 分析助手", key="assistant_open_button", on_click=set_assistant_open, args=(True,))


def main():
    st.set_page_config(page_title="FinRisk AI · 信用卡违约分析", layout="wide")
    st.html("""<style>
        .block-container {max-width: 1320px; padding-top: 3.75rem;}
        .finrisk-header {padding: 1.45rem 1.8rem; margin-bottom: 1rem; border-radius: 14px;
            background: linear-gradient(112deg, #14263c 52%, #164b56); color: #fff;}
        .finrisk-header h1 {margin: 0; padding: 0; color: #fff; font-size: 2rem; letter-spacing: -.03em;}
        .finrisk-header p {margin: .33rem 0 0; color: #d4e7ee;}
        [data-testid="stMetric"] {background: #fff; border-radius: 12px;}
        [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
        [data-testid="stTabs"] button[role="tab"] p {font-size: 1rem; font-weight: 600;}
        </style>
        <header class="finrisk-header"><h1>FinRisk AI</h1><p>信用卡次月违约分析</p></header>""")
    st.html(ROOT / "assistant_drawer.css")
    tabs = st.tabs(["数据总览", "风险分析", "模型分析"], key="dashboard_module", on_change="rerun")
    for tab, render in zip(tabs, (overview, risk, model)):
        if tab.open:
            with tab:
                try:
                    render()
                except ArtifactError as error:
                    st.error(str(error))
                except NoData as error:
                    st.info(str(error))

    assistant_entry()


if __name__ == "__main__":
    main()
