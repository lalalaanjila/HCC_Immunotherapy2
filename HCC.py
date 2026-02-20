import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# =========================
# 0. 基本配置
# =========================
st.set_page_config(page_title="HCC 疗效预测（临床+3D DL联合模型）", layout="wide")

MODEL_PATH = "DL_Clinical_XGBoost.pkl"
DATA_PATH = "HCC_DL_clinical.csv"   # 用于读取 min/max/median，便于输入默认值

# =========================
# 1. 加载模型与数据
# =========================
@st.cache_resource
def load_bundle():
    bundle = joblib.load(MODEL_PATH)
    return bundle

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

bundle = load_bundle()
model = bundle["model"]
feature_names = bundle["feature_names"]

data = load_data()

# 列检查
missing_cols = [c for c in feature_names if c not in data.columns]
if missing_cols:
    st.error(f"数据文件缺少列：{missing_cols}")
    st.stop()

X_all = data[feature_names]
stats = X_all.agg(["min", "max", "median"])

def is_binary_01(s: pd.Series) -> bool:
    vals = sorted(pd.Series(s.dropna().unique()).tolist())
    return len(vals) == 2 and vals == [0, 1]

# =========================
# 2. SHAP（TreeExplainer）
# =========================
@st.cache_resource
def build_shap_explainer_tree(_model):
    return shap.TreeExplainer(_model)

explainer = build_shap_explainer_tree(model)

# =========================
# 3. UI
# =========================
st.title("HCC 疗效预测在线计算器（临床 + 3D DL 特征联合模型）")

st.markdown("""
本工具基于 **XGBoost 联合模型（临床 5 项 + 3D DL PCA 特征 15 项）** 对 HCC 免疫治疗再挑战疗效进行预测。  
输出为 **P(group=1，应答/疗效较好)** 的概率，并提供个体层面 SHAP 解释（waterfall plot，适配 Streamlit Cloud）。
> ⚠️ 仅用于科研/教学，不能替代临床决策。
""")

st.subheader("1. 输入变量")

input_values = {}

# =========================
# （1）临床变量 —— 放在上面
# =========================
st.markdown("### （1）临床变量")

# Alcohol
if "Alcohol" in feature_names:
    if is_binary_01(data["Alcohol"]):
        input_values["Alcohol"] = st.selectbox(
            "Alcohol（饮酒史：0=无 / 1=有）",
            options=[0, 1],
            format_func=lambda x: "0 = 无饮酒史" if x == 0 else "1 = 有饮酒史",
        )
    else:
        input_values["Alcohol"] = st.number_input(
            "Alcohol（数值编码）",
            min_value=float(stats.loc["min", "Alcohol"]),
            max_value=float(stats.loc["max", "Alcohol"]),
            value=float(stats.loc["median", "Alcohol"]),
            step=1.0,
        )

# AFP
input_values["AFP"] = st.number_input(
    "AFP（ng/mL）",
    min_value=float(stats.loc["min", "AFP"]),
    max_value=float(stats.loc["max", "AFP"]),
    value=float(stats.loc["median", "AFP"]),
    step=1.0,
)

# AST_ALT
if is_binary_01(data["AST_ALT"]):
    input_values["AST_ALT"] = st.selectbox(
        "AST_ALT（0=低 / 1=高；按截断后的变量）",
        options=[0, 1],
    )
else:
    input_values["AST_ALT"] = st.number_input(
        "AST/ALT（比值）",
        min_value=float(stats.loc["min", "AST_ALT"]),
        max_value=float(stats.loc["max", "AST_ALT"]),
        value=float(stats.loc["median", "AST_ALT"]),
        step=0.01,
        format="%.4f",
    )

# Ascites
if is_binary_01(data["Ascites"]):
    input_values["Ascites"] = st.selectbox("Ascites（腹水：0=无 / 1=有）", options=[0, 1])
else:
    asc_opts = sorted(pd.Series(data["Ascites"].dropna().unique()).tolist())
    input_values["Ascites"] = st.selectbox("Ascites（腹水分级/编码）", options=asc_opts)

# ECOG_PS
ecog_opts = sorted(pd.Series(data["ECOG_PS"].dropna().unique()).tolist())
input_values["ECOG_PS"] = st.selectbox("ECOG-PS", options=ecog_opts)

# 分隔线：让（2）出现在（1）下面
st.markdown("---")

# =========================
# （2）3D DL 特征 —— 放在下面
# =========================
st.markdown("### （2）3D DL 特征（PCA 主成分）")

dl_cols = [c for c in feature_names if c.startswith("AP_PC") or c.startswith("VP_PC")]

for col in dl_cols:
    input_values[col] = st.number_input(
        label=f"{col}",
        min_value=float(stats.loc["min", col]),
        max_value=float(stats.loc["max", col]),
        value=float(stats.loc["median", col]),
        step=0.01,
        format="%.4f",
    )

# 组装模型输入
features_df = pd.DataFrame([[input_values[c] for c in feature_names]], columns=feature_names)

st.markdown("---")
st.subheader("2. 模型预测结果")

if st.button("点击进行预测"):
    predicted_class = int(model.predict(features_df)[0])
    predicted_proba = model.predict_proba(features_df)[0]

    prob_non = float(predicted_proba[0])
    prob_yes = float(predicted_proba[1])

    st.write(f"**预测类别 (group)：{predicted_class}**")
    st.write(f"- P(group=0，非应答)：{prob_non * 100:.1f}%")
    st.write(f"- P(group=1，应答)：{prob_yes * 100:.1f}%")

    if predicted_class == 1:
        st.success(f"模型提示应答概率较高：P(应答) ≈ {prob_yes * 100:.1f}%")
    else:
        st.warning(f"模型提示应答概率较低：P(应答) ≈ {prob_yes * 100:.1f}%")

    st.markdown("---")
    st.subheader("3. 个体可解释性（SHAP waterfall plot）")

    with st.spinner("计算 SHAP 中..."):
        shap_values = explainer.shap_values(features_df)

        # 二分类兼容：可能返回 list
        if isinstance(shap_values, list):
            shap_val = shap_values[1][0]   # 阳性类
            base_val = explainer.expected_value[1]
        else:
            shap_val = shap_values[0]
            base_val = explainer.expected_value

        explanation = shap.Explanation(
            values=shap_val,
            base_values=base_val,
            data=features_df.iloc[0].values,
            feature_names=features_df.columns.tolist(),
        )

        plt.figure()
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        plt.savefig("shap_waterfall_dl_clinical.png", dpi=300, bbox_inches="tight")
        plt.close()


    st.image("shap_waterfall_dl_clinical.png", caption="当前患者的 SHAP waterfall plot（对 P(group=1) 的贡献）")
