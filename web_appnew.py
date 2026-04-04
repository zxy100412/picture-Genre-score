# ====================== 这是你原来的代码，我只修报错 ======================
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

import streamlit as st

# 你原来的配置
MODEL_PATH = "genre_classifier_adv_model.pkl"
FEATURES_CSV_PATH = "painting_features.csv"
WEIGHTS_CSV_PATH = "comfort_weights.csv"
TARGET_SIZE = (512, 512)
GLCM_LEVELS = 16
CONFIDENCE_THRESHOLD = 0.7

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

# ==============================================
# 👇 这是你原来的函数，我只修复报错，完全不改逻辑
# ==============================================
def extract_features_from_image(img):
    img = img.convert("RGB").resize(TARGET_SIZE)
    img_np = np.array(img) / 255.0

    from skimage.color import rgb2lab
    lab = rgb2lab(img_np)
    L, A, B = lab[:,:,0], lab[:,:,1], lab[:,:,2]

    avg_luminance = np.mean(L)
    avg_saturation = np.mean(np.sqrt(A**2 + B**2))
    rms_contrast = np.std(L)
    color_variance = np.var(A) + np.var(B)
    color_richness = np.sqrt(color_variance)
    vividness = np.mean(np.abs(A) + np.abs(B))

    h, w = img_np.shape[:2]
    blocks = [
        img_np[:h//2, :w//2], img_np[:h//2, w//2:],
        img_np[h//2:, :w//2], img_np[h//2:, w//2:]
    ]
    block_means = [np.mean(b) for b in blocks]
    region_balance = 1 - (np.max(block_means) - np.min(block_means)) / (np.max(block_means) + 1e-6)
    region_balance = region_balance * 100

    from skimage.filters import sobel
    edge = sobel(rgb2lab(img_np)[:,:,0])
    gradient_smoothness = 1 - np.std(edge)
    gradient_smoothness = gradient_smoothness * 100

    # ====================== 我只修这里！！！ ======================
    gray = np.array(img.convert("L"))
    gray = gray.astype(np.uint8)   # 修复类型
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)

    # 👇 完全保留你原来的计算方式，只修维度错误！
    contrast = np.mean(graycoprops(glcm, 'contrast')) * 10
    correlation = np.mean(graycoprops(glcm, 'correlation')) * 100
    correlation = (correlation + 100) / 2
    energy = np.mean(graycoprops(glcm, 'energy')) * 100
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity')) * 100
    # ============================================================

    features = np.array([
        avg_luminance, avg_saturation, rms_contrast, color_richness, vividness,
        region_balance, gradient_smoothness, contrast, correlation, energy, homogeneity
    ])
    return features

# ====================== 你原来的评分函数，完全不动 ======================
def compute_comfort_score(features, genre, df_ref=None, weights=None):
    if genre not in GENRE_OPTIMAL:
        return 50.0, features / 100.0

    opt = GENRE_OPTIMAL[genre]
    x_best = np.array(opt["x_best"])
    sigma = np.array(opt["sigma"])
    normalized = np.exp(-((features - x_best) ** 2) / (2 * sigma ** 2))

    if weights is None:
        weights = np.ones(len(features)) / len(features)

    if df_ref is not None and not df_ref.empty:
        ref_features = df_ref[FEATURE_NAMES].values
        weighted_ref = ref_features * weights
        ideal_best = np.max(weighted_ref, axis=0)
        ideal_worst = np.min(weighted_ref, axis=0)
        weighted_current = normalized * weights
        d_best = np.sqrt(np.sum((weighted_current - ideal_best) ** 2))
        d_worst = np.sqrt(np.sum((weighted_current - ideal_worst) ** 2))
        score = d_worst / (d_best + d_worst) if (d_best + d_worst) != 0 else 0.5
    else:
        score = np.mean(normalized * weights)

    return score * 100, normalized

# ====================== 你原来的模型加载，完全不动 ======================
@st.cache_resource
def load_model_and_data():
    model = None
    le = None
    df = pd.DataFrame()
    weights = None
    if os.path.exists(MODEL_PATH):
        model_data = joblib.load(MODEL_PATH)
        model = model_data["model"]
        le = model_data["label_encoder"]
    if os.path.exists(FEATURES_CSV_PATH):
        df = pd.read_csv(FEATURES_CSV_PATH)
    if os.path.exists(WEIGHTS_CSV_PATH):
        w_df = pd.read_csv(WEIGHTS_CSV_PATH)
        weights = w_df["weight"].values if "weight" in w_df.columns else None
    return model, le, df, weights

model, le, df_ref, weights = load_model_and_data()

# ====================== Streamlit 界面 ======================
st.title("画作评价系统")
uploaded = st.file_uploader("上传图片", type=["png","jpg","jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="上传图片")
    
    with st.spinner("分析中..."):
        feat = extract_features_from_image(img)
        
        if model and le:
            proba = model.predict_proba(feat.reshape(1,-1))[0]
            idx = np.argmax(proba)
            genre = le.inverse_transform([idx])[0]
            conf = proba[idx]
        else:
            genre = "未知"
            conf = 0.0
            
        score, norm = compute_comfort_score(feat, genre, df_ref, weights)
    
    st.success(f"流派：{genre} | 置信度：{conf:.1%} | 舒适度：{score:.1f}")
