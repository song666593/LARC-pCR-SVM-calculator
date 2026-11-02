# LARC nCRT后 pCR 在线预测工具（RBF-SVM）

基于炎症—代谢相关双时相实验室指标，预测局部进展期直肠癌新辅助放化疗后病理完全缓解（pCR）概率。

**仅供科研与术前辅助分层参考，不能替代病理诊断、MRI/内镜评估及多学科临床决策。**

## Streamlit Cloud 部署

1. 将本仓库连接到 [Streamlit Community Cloud](https://share.streamlit.io)
2. Main file path 填写：`app_streamlit.py`
3. Deploy 后获得公网链接，写入论文方法部分

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

## 输入变量（6）

- pre-PLR：新辅助放化疗前血小板/淋巴细胞比值
- post-LMR：术前淋巴细胞/单核细胞比值
- post-LYM：术前淋巴细胞计数
- pre-GLB：新辅助放化疗前球蛋白（g/L）
- pre-ALB：新辅助放化疗前白蛋白（g/L）
- ∆TC：总胆固醇变化量

工作点阈值：0.21（与内部验证报告口径一致）
