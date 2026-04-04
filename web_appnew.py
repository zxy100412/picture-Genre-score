# ==============================================
# 1. 保留所有核心依赖，删除Flask相关导入
# ==============================================
import os
import io
import json
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import joblib
import warnings
warnings.filterwarnings('ignore')

# 新增Streamlit导入，替代Flask
import streamlit as st

# ==============================================
# 2. 保留你原有的所有配置、函数、模型加载逻辑
# ==============================================
# --------------------------
# 配置区（完全保留你的原有配置）
# --------------------------
MODEL_PATH = "genre_classifier_adv_model.pkl"
FEATURES_CSV_PATH = "painting_features.csv"
WEIGHTS_CSV_PATH = "comfort_weights.csv"
TARGET_SIZE = (512, 512)
GLCM_LEVELS = 16
CONFIDENCE_THRESHOLD = 0.7

# 流派最优参数（完全保留你的原有配置）
GENRE_OPTIMAL = {
    "洛可可": {
        "x_best": [80, 65, 40, 75, 70, 85, 75, 20, 80, 60, 75],
        "sigma": [15, 12, 10, 15, 12, 10, 12, 8, 15, 12, 15]
    },
    "印象主义": {
        "x_best": [70, 70, 50, 70, 65, 75, 70, 30, 70, 55, 70],
        "sigma": [15, 12, 12, 15, 12, 12, 12, 10, 15, 12, 15]
    },
    "现实主义": {
        "x_best": [50, 50, 60, 55, 50, 70, 65, 40, 60, 50, 60],
        "sigma": [15, 12, 12, 15, 12, 12, 12, 10, 15, 12, 15]
    },
    "浪漫主义": {
        "x_best": [60, 60, 55, 65, 60, 75, 70, 35, 65, 55, 65],
        "sigma": [15, 12, 12, 15, 12, 12, 12, 10, 15, 12, 15]
    },
    "后印象派": {
        "x_best": [65, 65, 55, 70, 65, 70, 65, 35, 65, 55, 65],
        "sigma": [15, 12, 12, 15, 12, 12, 12, 10, 15, 12, 15]
    },
    "抽象派": {
        "x_best": [55, 60, 60, 65, 60, 65, 60, 40, 60, 50, 60],
        "sigma": [15, 12, 12, 15, 12, 12, 12, 10, 15, 12, 15]
    }
}

FEATURE_NAMES = [
    "平均亮度", "平均饱和度", "RMS对比度", "颜色丰富度", "鲜明度",
    "区域平衡度", "梯度平滑度", "GLCM对比度", "GLCM相关性", "GLCM能量", "GLCM同质性"
]

# --------------------------
# 特征提取函数（完全保留你的原有逻辑）
# --------------------------
def extract_features_from_image(img):
    """从PIL图像提取11个视觉特征，完全保留原有逻辑"""
    # 图像预处理
    img = img.convert("RGB").resize(TARGET_SIZE)
    img_np = np.array(img) / 255.0

    # 1. 颜色特征（LAB空间）
    from skimage.color import rgb2lab
    lab = rgb2lab(img_np)
    L, A, B = lab[:,:,0], lab[:,:,1], lab[:,:,2]

    avg_luminance = np.mean(L)  # 0-100
    avg_saturation = np.mean(np.sqrt(A**2 + B**2))  # 0-~100
    rms_contrast = np.std(L)  # 0-~100
    color_variance = np.var(A) + np.var(B)
    color_richness = np.sqrt(color_variance)  # 0-~100
    vividness = np.mean(np.abs(A) + np.abs(B))  # 0-~100

    # 2. 空间特征
    h, w = img_np.shape[:2]
    blocks = [
        img_np[:h//2, :w//2], img_np[:h//2, w//2:],
        img_np[h//2:, :w//2], img_np[h//2:, w//2:]
    ]
    block_means = [np.mean(b) for b in blocks]
    region_balance = 1 - (np.max(block_means) - np.min(block_means)) / (np.max(block_means) + 1e-6)
    region_balance = region_balance * 100  # 0-100

    from skimage.filters import sobel
    edge = sobel(rgb2lab(img_np)[:,:,0])
    gradient_smoothness = 1 - np.std(edge)
    gradient_smoothness = gradient_smoothness * 100  # 0-100

    # 3. 纹理特征（GLCM）
    gray = np.array(img.convert("L"))
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=GLCM_LEVELS, symmetric=True, normed=True)
    glcm_avg = np.mean(glcm, axis=3)

    contrast = graycoprops(glcm_avg, 'contrast')[0, 0] * 10  # 0-~100
    correlation = graycoprops(glcm_avg, 'correlation')[0, 0] * 100  # -100~100 → 0~100
    correlation = (correlation + 100) / 2
    energy = graycoprops(glcm_avg, 'energy')[0, 0] * 100  # 0-100
    homogeneity = graycoprops(glcm_avg, 'homogeneity')[0, 0] * 100  # 0-100

    features = np.array([
        avg_luminance, avg_saturation, rms_contrast, color_richness, vividness,
        region_balance, gradient_smoothness, contrast, correlation, energy, homogeneity
    ])
    return features

# --------------------------
# 舒适度评分函数（完全保留你的原有逻辑）
# --------------------------
def compute_comfort_score(features, genre, df_ref=None, weights=None):
    """计算视觉舒适度评分，完全保留原有TOPSIS+熵权法逻辑"""
    if genre not in GENRE_OPTIMAL:
        return 50.0, features / 100.0

    opt = GENRE_OPTIMAL[genre]
    x_best = np.array(opt["x_best"])
    sigma = np.array(opt["sigma"])

    # 高斯正向化
    normalized = np.exp(-((features - x_best) ** 2) / (2 * sigma ** 2))

    if weights is None:
        weights = np.ones(len(features)) / len(features)

    # TOPSIS评分
    if df_ref is not None and not df_ref.empty:
        ref_features = df_ref[FEATURE_NAMES].values
        weighted_ref = ref_features * weights
        ideal_best = np.max(weighted_ref, axis=0)
        ideal_worst = np.min(weighted_ref, axis=0)

        weighted_current = normalized * weights
        d_best = np.sqrt(np.sum((weighted_current - ideal_best) ** 2))
        d_worst = np.sqrt(np.sum((weighted_current - ideal_worst) ** 2))

        if d_best + d_worst == 0:
            score = 0.5
        else:
            score = d_worst / (d_best + d_worst)
    else:
        score = np.mean(normalized * weights)

    comfort_score = score * 100
    return comfort_score, normalized

# --------------------------
# 模型与数据加载（完全保留你的原有逻辑）
# --------------------------
@st.cache_resource  # Streamlit缓存，加速加载
def load_model_and_data():
    model = None
    le = None
    df = pd.DataFrame()
    weights = None

    # 加载分类模型
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model_data = joblib.load(f)
            model = model_data["model"]
            le = model_data["label_encoder"]

    # 加载特征数据集
    if os.path.exists(FEATURES_CSV_PATH):
        df = pd.read_csv(FEATURES_CSV_PATH)

    # 加载权重
    if os.path.exists(WEIGHTS_CSV_PATH):
        w_df = pd.read_csv(WEIGHTS_CSV_PATH)
        if "weight" in w_df.columns:
            weights = w_df["weight"].values

    # 预计算各流派TOPSIS理想解
    genre_ideal = {}
    if not df.empty and le is not None:
        for genre in le.classes_:
            genre_df = df[df["流派"] == genre]
            if not genre_df.empty:
                ref_features = genre_df[FEATURE_NAMES].values
                if weights is None:
                    w = np.ones(ref_features.shape[1]) / ref_features.shape[1]
                else:
                    w = weights
                weighted_ref = ref_features * w
                ideal_best = np.max(weighted_ref, axis=0)
                ideal_worst = np.min(weighted_ref, axis=0)
                genre_ideal[genre] = (ideal_best, ideal_worst, w)
    return model, le, df, weights, genre_ideal

model, le, df_ref, weights, genre_ideal = load_model_and_data()

# ==============================================
# 3. Streamlit 网页界面（替代Flask，完全重写）
# ==============================================
st.set_page_config(page_title="画作视觉舒适度评价系统", layout="wide", page_icon="🎨")

# 页面标题
st.title("🎨 画作视觉舒适度智能评价系统")
st.markdown("---")

# 侧边栏说明
with st.sidebar:
    st.header("系统说明")
    st.info("""
    本系统基于图像特征提取与机器学习，为不同绘画流派提供定制化的视觉舒适度量化评价。
    支持流派自动识别、多维度特征分析、舒适度评分三大核心功能。
    """)
    st.subheader("支持流派")
    st.write(", ".join(GENRE_OPTIMAL.keys()))

# 主界面：图片上传
st.subheader("📤 上传画作图片")
uploaded_file = st.file_uploader("选择一张图片（支持PNG/JPG/JPEG）", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 1. 显示上传的图片
    col1, col2 = st.columns([1, 2])
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="上传的画作", use_column_width=True)

    # 2. 执行分析
    with st.spinner("🔍 正在分析图像特征..."):
        # 提取特征
        features = extract_features_from_image(img)

        # 流派预测
        if model is not None and le is not None:
            proba = model.predict_proba(features.reshape(1, -1))[0]
            pred_idx = np.argmax(proba)
            genre = le.inverse_transform([pred_idx])[0]
            confidence = proba[pred_idx] * 100
            all_proba = {le.classes_[i]: round(proba[i]*100, 2) for i in range(len(proba))}
            low_confidence = confidence < CONFIDENCE_THRESHOLD * 100
        else:
            genre = "未知流派"
            confidence = 0.0
            all_proba = {}
            low_confidence = True

        # 计算舒适度评分
        comfort_score, normalized = compute_comfort_score(features, genre, df_ref, weights)

        # 获取流派平均特征（用于对比）
        if not df_ref.empty and genre in df_ref["流派"].values:
            genre_avg = df_ref[df_ref["流派"] == genre][FEATURE_NAMES].mean().values
        else:
            genre_avg = np.zeros_like(features)

    # 3. 展示分析结果
    with col2:
        st.subheader("📊 分析结果")

        # 流派与置信度
        st.metric("识别流派", genre, f"置信度: {confidence:.1f}%")
        if low_confidence:
            st.warning("⚠️ 置信度较低，建议检查图片或手动确认流派")

        # 舒适度评分
        st.metric("视觉舒适度评分", f"{comfort_score:.1f} / 100", 
                  delta=f"{comfort_score - 50:.1f}" if comfort_score > 50 else f"{comfort_score - 50:.1f}")

        # 特征详情展示
        st.subheader("🔍 详细特征指标")
        feature_df = pd.DataFrame({
            "特征名称": FEATURE_NAMES,
            "当前画作值": [round(f, 2) for f in features],
            f"{genre}平均水平": [round(f, 2) for f in genre_avg],
            "归一化得分": [round(n, 3) for n in normalized]
        })
        st.dataframe(feature_df, use_container_width=True, hide_index=True)

        # 所有流派概率
        st.subheader("📈 各流派预测概率")
        proba_df = pd.DataFrame(list(all_proba.items()), columns=["流派", "概率(%)"])
        st.bar_chart(proba_df.set_index("流派"), use_container_width=True)

# 页脚
st.markdown("---")
st.caption("基于Python + Streamlit + 机器学习的画作视觉舒适度评价系统")