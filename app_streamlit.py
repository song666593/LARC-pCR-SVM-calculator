# -*- coding: utf-8 -*-
"""
Streamlit 版在线计算器（便于部署到 Streamlit Community Cloud，获得公网 URL）。

本地运行：
  pip install streamlit joblib scikit-learn pandas numpy
  streamlit run app_streamlit.py
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


@st.cache_resource
def load_artifacts():
    if not JOB.is_file():
        return None, None
    model = joblib.load(JOB)
    meta = json.loads(META.read_text(encoding="utf-8")) if META.is_file() else {}
    return model, meta


model, meta = load_artifacts()

st.title("基于炎症代谢相关指标预测局部进展期直肠癌新辅助放化疗疗效预测器")

if model is None:
    st.error("未找到 svm_pcr_pipeline.joblib。请先运行：python export_svm_joblib.py")
    st.stop()

thr = float(meta.get("clinical_probability_threshold", 0.21))
labels = meta.get("feature_labels_zh") or {}
order = meta.get("feature_order") or [
    "pre-PLR",
    "post-LMR",
    "post-LYM",
    "pre-GLB",
    "pre-ALB",
    "∆TC",
]

vals = {}
cols = st.columns(2)
for i, key in enumerate(order):
    with cols[i % 2]:
        vals[key] = st.number_input(
            labels.get(key, key),
            value=None,
            placeholder="请输入数值",
            format="%.6f",
            key=f"in_{i}",
        )

if st.button("计算 pCR 概率", type="primary", use_container_width=True):
    if any(v is None for v in vals.values()):
        st.warning("请完整填写 6 项指标。")
    else:
        X = pd.DataFrame([[vals[k] for k in order]], columns=order)
        proba = float(model.predict_proba(X)[0, 1])
        pred = proba >= thr
        st.markdown(
            f"""
<div style="text-align:center; margin: 0.8rem 0 0.4rem;">
  <div style="font-size:1.15rem; color:#334155; font-weight:600;">预测 pCR 概率</div>
  <div style="font-size:3.2rem; font-weight:800; color:#0f5c6b; line-height:1.15; margin-top:0.25rem;">
    {proba * 100:.2f}%
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        if pred:
            st.success(f"按阈值 {thr:.2f}：判为「预测 pCR 阳性」")
        else:
            st.warning(f"按阈值 {thr:.2f}：判为「预测 pCR 阴性」")
