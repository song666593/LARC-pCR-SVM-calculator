# -*- coding: utf-8 -*-
"""
Streamlit 版在线计算器（便于部署到 Streamlit Community Cloud，获得公网 URL）。

本地运行：
  pip install streamlit joblib scikit-learn pandas numpy
  streamlit run app_streamlit.py

部署前请先在本目录生成 svm_pcr_pipeline.joblib：
  python export_svm_joblib.py
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
    page_title="LARC nCRT后pCR预测",
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

st.title("局部进展期直肠癌新辅助放化疗后 pCR 预测工具")
st.caption("基于 LASSO 入选的 6 项炎症—代谢相关指标与径向基核支持向量机（RBF-SVM）")

st.info(
    "本工具仅供科研与术前辅助分层参考，不能替代 MRI/内镜再分期、病理诊断及多学科临床决策。"
    "模型尚未完成多中心外部验证，外推需谨慎。"
)

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

with st.expander("模型说明（可写入论文方法）", expanded=False):
    st.markdown(
        f"""
- **输入变量（6）**：治疗前 PLR、ALB、GLB；术前 LMR、LYM；总胆固醇变化量 ΔTC  
- **预处理**：中位数插补 → Winsor 1%–99% → 标准化  
- **算法**：RBF-SVM（C=1.6，gamma=scale，class_weight=balanced）  
- **临床工作点阈值**：{thr:.2f}（与验证集报告的灵敏度/特异度口径一致）  
- **拟合样本**：fit_on={meta.get('fit_on')}，n_fit={meta.get('n_fit')}  
- **内部核对**：训练 AUC={meta.get('auc_train')}，验证 AUC={meta.get('auc_val')}
"""
    )

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
        st.metric("预测 pCR 概率", f"{proba * 100:.2f}%", help=f"原始概率={proba:.4f}")
        if pred:
            st.success(f"按阈值 {thr:.2f}：判为「预测 pCR 阳性」")
        else:
            st.warning(f"按阈值 {thr:.2f}：判为「预测 pCR 阴性」")
        st.caption("阳性仅表示模型输出概率达到预设工作点，不等于病理完全缓解。")

st.divider()
st.markdown(
    "**适用场景**：nCRT 完成、术前再分期之后、病理结果揭晓之前，用常规化验补充全身炎症—营养信息。"
    "不适用：初始方案选择、替代影像评估、单独启动观察等待。"
)
