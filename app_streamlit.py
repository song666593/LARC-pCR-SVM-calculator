# -*- coding: utf-8 -*-
"""
Streamlit 版在线计算器（便于部署到 Streamlit Community Cloud）。
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import calculator_preprocessing  # noqa: F401

HERE = Path(__file__).resolve().parent
JOB = HERE / "svm_pcr_pipeline.joblib"
META = HERE / "model_meta.json"

st.set_page_config(
    page_title="基于炎症代谢相关指标预测局部进展期直肠癌新辅助放化疗疗效预测器",
    page_icon="🩺",
    layout="centered",
)

st.markdown(
    """
<style>
  /* 页面整体 */
  .block-container {
    padding-top: 1.6rem !important;
    padding-bottom: 2rem !important;
    max-width: 820px !important;
  }

  /* 标题：略收，避免压过输入区 */
  h1 {
    font-size: 1.55rem !important;
    font-weight: 750 !important;
    line-height: 1.4 !important;
    color: #0f3d4c !important;
    text-align: center !important;
    margin: 0 0 1.35rem 0 !important;
  }

  /* 自定义变量名标签 */
  .var-label {
    font-size: 1.08rem;
    font-weight: 700;
    color: #1e293b;
    line-height: 1.45;
    margin: 0.15rem 0 0.35rem 0;
  }
  .var-hint {
    font-size: 0.86rem;
    font-weight: 500;
    color: #64748b;
    margin-left: 0.15rem;
  }

  /* 输入框数字加大、高度统一 */
  div[data-testid="stNumberInput"] label { display: none !important; }
  div[data-testid="stNumberInput"] input {
    font-size: 1.18rem !important;
    font-weight: 600 !important;
    color: #0f172a !important;
    min-height: 2.75rem !important;
    padding-top: 0.45rem !important;
    padding-bottom: 0.45rem !important;
  }
  div[data-testid="stNumberInput"] {
    margin-bottom: 0.85rem !important;
  }

  /* 按钮：医疗青绿，降低纯红压迫感 */
  div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #0f5c6b, #1d8a7a) !important;
    border: none !important;
    color: #fff !important;
    min-height: 3rem !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 16px rgba(15, 92, 107, 0.18) !important;
  }
  div[data-testid="stButton"] button p {
    font-size: 1.12rem !important;
    font-weight: 700 !important;
  }
  div[data-testid="stButton"] button:hover {
    filter: brightness(1.05);
  }

  /* 结果区 */
  .result-card {
    margin-top: 0.9rem;
    padding: 1.05rem 1rem 0.95rem;
    border-radius: 14px;
    background: linear-gradient(180deg, #f3fbf9 0%, #ffffff 100%);
    border: 1px solid #cfe8e3;
    text-align: center;
  }
  .result-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #334155;
  }
  .result-value {
    font-size: 2.85rem;
    font-weight: 800;
    color: #0f5c6b;
    line-height: 1.15;
    margin: 0.35rem 0 0.55rem;
  }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    if not JOB.is_file():
        return None, None
    model = joblib.load(JOB)
    meta = json.loads(META.read_text(encoding="utf-8")) if META.is_file() else {}
    return model, meta


def field_label(title: str, code: str) -> None:
    st.markdown(
        f'<div class="var-label">{title}<span class="var-hint">（{code}）</span></div>',
        unsafe_allow_html=True,
    )


model, meta = load_artifacts()

st.title("基于炎症代谢相关指标预测局部进展期直肠癌新辅助放化疗疗效预测器")

if model is None:
    st.error("未找到 svm_pcr_pipeline.joblib。请先运行：python export_svm_joblib.py")
    st.stop()

thr = float(meta.get("clinical_probability_threshold", 0.21))
order = meta.get("feature_order") or [
    "pre-PLR",
    "post-LMR",
    "post-LYM",
    "pre-GLB",
    "pre-ALB",
    "∆TC",
]

# 展示名：中文为主，英文代号作辅，避免一行过长显得碎
DISPLAY = {
    "pre-PLR": ("新辅助放化疗前血小板/淋巴细胞比值", "pre-PLR"),
    "post-LMR": ("术前淋巴细胞/单核细胞比值", "post-LMR"),
    "post-LYM": ("术前淋巴细胞计数", "post-LYM"),
    "pre-GLB": ("新辅助放化疗前球蛋白", "pre-GLB, g/L"),
    "pre-ALB": ("新辅助放化疗前白蛋白", "pre-ALB, g/L"),
    "∆TC": ("总胆固醇变化量", "ΔTC"),
}

vals = {}
left, right = st.columns(2, gap="large")
half = (len(order) + 1) // 2
for i, key in enumerate(order):
    col = left if i < half else right
    title, code = DISPLAY.get(key, (key, key))
    with col:
        field_label(title, code)
        vals[key] = st.number_input(
            label=key,
            value=None,
            placeholder="请输入数值",
            format="%.4f",
            key=f"in_{i}",
            label_visibility="collapsed",
        )

st.write("")  # 一点间距
if st.button("计算 pCR 概率", type="primary", use_container_width=True):
    if any(v is None for v in vals.values()):
        st.warning("请完整填写 6 项指标。")
    else:
        X = pd.DataFrame([[vals[k] for k in order]], columns=order)
        proba = float(model.predict_proba(X)[0, 1])
        pred = proba >= thr
        badge = (
            f'<div style="display:inline-block;margin-top:.2rem;padding:.45rem .9rem;'
            f'border-radius:999px;background:#d1fae5;color:#065f46;font-weight:700;font-size:1.02rem;">'
            f'按阈值 {thr:.2f}：判为「预测 pCR 阳性」</div>'
            if pred
            else f'<div style="display:inline-block;margin-top:.2rem;padding:.45rem .9rem;'
            f'border-radius:999px;background:#e2e8f0;color:#334155;font-weight:700;font-size:1.02rem;">'
            f'按阈值 {thr:.2f}：判为「预测 pCR 阴性」</div>'
        )
        st.markdown(
            f"""
<div class="result-card">
  <div class="result-title">预测 pCR 概率</div>
  <div class="result-value">{proba * 100:.2f}%</div>
  {badge}
</div>
""",
            unsafe_allow_html=True,
        )
