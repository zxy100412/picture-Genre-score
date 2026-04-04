# -*- coding: utf-8 -*-
"""
================================================================
  画作分派别视觉舒适度评价 — Web 应用
================================================================
说明：
  Flask 后端，集成特征提取 + 流派分类 + 舒适度评价。
  用户上传一张画作 → 返回流派、置信度、舒适度得分、11 个特征值。

启动方式：python web_app.py
访问地址：http://localhost:5000
================================================================
"""

import os
import io
import json
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, render_template, request, jsonify
from skimage.feature import graycomatrix, graycoprops
import joblib
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 最大 200MB

# ================================================================
# ========================= 配置区 ===============================
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'genre_classifier_adv_model.pkl')
FEATURES_CSV = os.path.join(BASE_DIR, 'painting_features.csv')
TARGET_SIZE = (512, 512)
GLCM_LEVELS = 16
CONFIDENCE_THRESHOLD = 0.7

FEATURE_COLS = [
    '平均亮度', '平均饱和度', 'RMS对比度', '颜色丰富度', '鲜明度',
    '区域平衡度', '梯度平滑度',
    'GLCM对比度', 'GLCM相关性', 'GLCM能量', 'GLCM同质性',
]

# 舒适度最优参数（从 step3 复制）
DEFAULT_OPTIMAL = {
    '平均亮度':   {'x_best': 60,   'sigma': 15},
    '平均饱和度': {'x_best': 40,   'sigma': 10},
    'RMS对比度':  {'x_best': 0.4,  'sigma': 0.15},
    '颜色丰富度': {'x_best': 200,  'sigma': 150},
    '鲜明度':     {'x_best': 25,   'sigma': 10},
    '区域平衡度': {'x_best': 60,   'sigma': 80},
    '梯度平滑度': {'x_best': 50,   'sigma': 30},
    'GLCM对比度': {'x_best': 1.5,  'sigma': 2.0},
    'GLCM相关性': {'x_best': 0.85, 'sigma': 0.15},
    'GLCM能量':   {'x_best': 0.35, 'sigma': 0.2},
    'GLCM同质性': {'x_best': 0.80, 'sigma': 0.15},
}

GENRE_OPTIMAL_PARAMS = {
    "洛可可艺术": {
        '平均亮度':   {'x_best': 100.9343, 'sigma': 30.0},
        '平均饱和度': {'x_best': 100.7258, 'sigma': 20.0},
        'RMS对比度':  {'x_best': 50.9931,  'sigma': 15.0},
        '颜色丰富度': {'x_best': 18.8574,  'sigma': 50.0},
        '鲜明度':     {'x_best': 31.2463,  'sigma': 10.0},
        '区域平衡度': {'x_best': 204.9224, 'sigma': 60.0},
        '梯度平滑度': {'x_best': 202.6575, 'sigma': 50.0},
        'GLCM对比度': {'x_best': 0.5849,   'sigma': 0.32},
        'GLCM相关性': {'x_best': 0.9448,   'sigma': 0.02},
        'GLCM能量':   {'x_best': 0.2806,   'sigma': 0.06},
        'GLCM同质性': {'x_best': 0.8299,   'sigma': 0.05},
    },
    "印象主义": {
        '平均亮度':   {'x_best': 134.5154, 'sigma': 30.0},
        '平均饱和度': {'x_best': 73.8233,  'sigma': 20.0},
        'RMS对比度':  {'x_best': 46.0796,  'sigma': 15.0},
        '颜色丰富度': {'x_best': 20.4687,  'sigma': 50.0},
        '鲜明度':     {'x_best': 27.2453,  'sigma': 10.0},
        '区域平衡度': {'x_best': 213.5707, 'sigma': 60.0},
        '梯度平滑度': {'x_best': 191.2782, 'sigma': 50.0},
        'GLCM对比度': {'x_best': 0.4749,   'sigma': 0.24},
        'GLCM相关性': {'x_best': 0.8826,   'sigma': 0.07},
        'GLCM能量':   {'x_best': 0.3857,   'sigma': 0.08},
        'GLCM同质性': {'x_best': 0.8418,   'sigma': 0.04},
    },
    "现实主义": {
        '平均亮度':   {'x_best': 136.6471, 'sigma': 30.0},
        '平均饱和度': {'x_best': 95.8793,  'sigma': 20.0},
        'RMS对比度':  {'x_best': 49.8605,  'sigma': 15.0},
        '颜色丰富度': {'x_best': 21.5480,  'sigma': 50.0},
        '鲜明度':     {'x_best': 31.6307,  'sigma': 10.0},
        '区域平衡度': {'x_best': 220.3357, 'sigma': 60.0},
        '梯度平滑度': {'x_best': 180.1425, 'sigma': 50.0},
        'GLCM对比度': {'x_best': 0.8199,   'sigma': 0.58},
        'GLCM相关性': {'x_best': 0.9071,   'sigma': 0.05},
        'GLCM能量':   {'x_best': 0.3175,   'sigma': 0.11},
        'GLCM同质性': {'x_best': 0.7999,   'sigma': 0.08},
    },
    "水墨文人画": {
        '平均亮度':   {'x_best': 176.1531, 'sigma': 30.0},
        '平均饱和度': {'x_best': 39.2437,  'sigma': 20.0},
        'RMS对比度':  {'x_best': 35.9652,  'sigma': 15.0},
        '颜色丰富度': {'x_best': 4.4182,   'sigma': 50.0},
        '鲜明度':     {'x_best': 25.2035,  'sigma': 10.0},
        '区域平衡度': {'x_best': 236.1464, 'sigma': 60.0},
        '梯度平滑度': {'x_best': 218.7258, 'sigma': 50.0},
        'GLCM对比度': {'x_best': 0.7299,   'sigma': 0.39},
        'GLCM相关性': {'x_best': 0.8060,   'sigma': 0.06},
        'GLCM能量':   {'x_best': 0.6425,   'sigma': 0.11},
        'GLCM同质性': {'x_best': 0.8796,   'sigma': 0.04},
    },
    "青绿山水画": {
        '平均亮度':   {'x_best': 174.6460, 'sigma': 30.0},
        '平均饱和度': {'x_best': 50.5496,  'sigma': 20.0},
        'RMS对比度':  {'x_best': 38.3564,  'sigma': 15.0},
        '颜色丰富度': {'x_best': 8.6878,   'sigma': 50.0},
        '鲜明度':     {'x_best': 27.0934,  'sigma': 10.0},
        '区域平衡度': {'x_best': 221.3906, 'sigma': 60.0},
        '梯度平滑度': {'x_best': 208.5856, 'sigma': 50.0},
        'GLCM对比度': {'x_best': 0.6390,   'sigma': 0.34},
        'GLCM相关性': {'x_best': 0.8338,   'sigma': 0.04},
        'GLCM能量':   {'x_best': 0.4656,   'sigma': 0.11},
        'GLCM同质性': {'x_best': 0.8384,   'sigma': 0.05},
    },
    "表现主义": DEFAULT_OPTIMAL.copy(),
    "抽象主义": DEFAULT_OPTIMAL.copy(),
}


# ================================================================
# ========== 特征提取（复用 step1 逻辑） =========================
# ================================================================

def _f_nonlinear(t):
    threshold = (6.0 / 29.0) ** 3
    return np.where(
        t > threshold,
        np.cbrt(t),
        (1.0 / 3.0) * (29.0 / 6.0) ** 2 * t + 4.0 / 29.0
    )

def rgb_to_lab(img_rgb):
    img = img_rgb.astype(np.float64) / 255.0
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    X = 0.4125 * R + 0.3576 * G + 0.1805 * B
    Y = 0.2127 * R + 0.7152 * G + 0.0722 * B
    Z = 0.0139 * R + 0.1192 * G + 0.9503 * B
    Xn, Yn, Zn = 95.047, 100.0, 108.883
    fx = _f_nonlinear(X * 100.0 / Xn)
    fy = _f_nonlinear(Y * 100.0 / Yn)
    fz = _f_nonlinear(Z * 100.0 / Zn)
    L_star = 116.0 * fy - 16.0
    a_star = 500.0 * (fx - fy)
    b_star = 200.0 * (fy - fz)
    return np.stack([L_star, a_star, b_star], axis=-1)

def extract_features_from_image(pil_img):
    """从 PIL Image 对象提取 11 个特征"""
    img = pil_img.convert('RGB').resize(TARGET_SIZE, Image.LANCZOS)
    img_rgb = np.array(img)
    img_lab = rgb_to_lab(img_rgb)

    L = img_lab[:, :, 0]
    a = img_lab[:, :, 1]
    b = img_lab[:, :, 2]
    N = L.size

    # 颜色指标
    mean_L = np.mean(L)
    chroma = np.sqrt(a ** 2 + b ** 2)
    mean_C = np.mean(chroma)
    denom = np.sqrt(mean_C ** 2 + mean_L ** 2)
    mean_S = (mean_C / denom * 100) if denom > 0 else 0
    rms_contrast = (np.std(L) / mean_L) if mean_L > 0 else 0
    V_ab = np.var(a) + np.var(b)
    vividness = np.mean(np.sqrt((L - 50) ** 2 + a ** 2 + b ** 2))

    # 空间指标
    h, w = L.shape
    gh, gw = h // 3, w // 3
    rmeans = []
    for row in range(3):
        for col in range(3):
            rs, re = row * gh, ((row + 1) * gh if row < 2 else h)
            cs, ce = col * gw, ((col + 1) * gw if col < 2 else w)
            rmeans.append((np.mean(L[rs:re, cs:ce]), np.mean(a[rs:re, cs:ce]), np.mean(b[rs:re, cs:ce])))
    diffs = []
    for p in range(9):
        for q in range(p + 1, 9):
            d = np.sqrt(sum((rmeans[p][k] - rmeans[q][k]) ** 2 for k in range(3)))
            diffs.append(d)
    balance_B = np.mean(diffs)

    gy, gx = np.gradient(L)
    grad_smooth = np.var(np.sqrt(gx ** 2 + gy ** 2))

    # 纹理指标
    L_q = np.clip(L, 0, 100)
    L_q = (L_q / 100.0 * (GLCM_LEVELS - 1)).astype(np.uint8)
    glcm = graycomatrix(L_q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=GLCM_LEVELS, symmetric=True, normed=True)
    g_con = graycoprops(glcm, 'contrast').mean()
    g_cor = graycoprops(glcm, 'correlation').mean()
    g_ene = graycoprops(glcm, 'energy').mean()
    g_hom = graycoprops(glcm, 'homogeneity').mean()

    return {
        '平均亮度': round(float(mean_L), 4),
        '平均饱和度': round(float(mean_S), 4),
        'RMS对比度': round(float(rms_contrast), 4),
        '颜色丰富度': round(float(V_ab), 4),
        '鲜明度': round(float(vividness), 4),
        '区域平衡度': round(float(balance_B), 4),
        '梯度平滑度': round(float(grad_smooth), 4),
        'GLCM对比度': round(float(g_con), 4),
        'GLCM相关性': round(float(g_cor), 4),
        'GLCM能量': round(float(g_ene), 4),
        'GLCM同质性': round(float(g_hom), 4),
    }


# ================================================================
# ========== 舒适度评价（与 step3 完全一致） =====================
# ================================================================

# 加载预计算的熵权法权重（来自 step3 跑全量数据的结果）
WEIGHTS_CSV = os.path.join(BASE_DIR, 'comfort_weights.csv')
GENRE_WEIGHTS = {}
GENRE_TOPSIS_REF = {}  # 每个流派的 TOPSIS 正负理想解

def _load_weights_and_refs():
    """加载预计算的权重，并从训练数据计算 TOPSIS 参考点"""
    global GENRE_WEIGHTS, GENRE_TOPSIS_REF

    # 1. 加载熵权法权重
    if os.path.exists(WEIGHTS_CSV):
        wdf = pd.read_csv(WEIGHTS_CSV, index_col=0)
        for genre in wdf.index:
            GENRE_WEIGHTS[genre] = {col: float(wdf.loc[genre, col]) for col in FEATURE_COLS}
        print(f"  [OK] 加载了 {len(GENRE_WEIGHTS)} 个流派的熵权法权重")
    else:
        print(f"  [!] 未找到权重文件，将使用等权重")

    # 2. 从训练数据计算每个流派的 TOPSIS 正负理想解
    if os.path.exists(FEATURES_CSV):
        df_all_data = pd.read_csv(FEATURES_CSV)
        for genre in df_all_data['流派'].unique():
            genre_data = df_all_data[df_all_data['流派'] == genre][FEATURE_COLS]
            optimal = GENRE_OPTIMAL_PARAMS.get(genre, DEFAULT_OPTIMAL)
            weights = GENRE_WEIGHTS.get(genre, {f: 1.0/len(FEATURE_COLS) for f in FEATURE_COLS})

            # 对训练数据做高斯正向化
            norm_data = pd.DataFrame(index=genre_data.index)
            for feat in FEATURE_COLS:
                x = genre_data[feat].values
                x_best = optimal[feat]['x_best']
                sigma = optimal[feat]['sigma']
                norm_data[feat] = np.exp(-((x - x_best) ** 2) / (2 * sigma ** 2))

            # 加权
            w = np.array([weights[f] for f in FEATURE_COLS])
            V = norm_data.values * w

            # 正负理想解
            GENRE_TOPSIS_REF[genre] = {
                'V_plus': V.max(axis=0),
                'V_minus': V.min(axis=0),
            }
        print(f"  [OK] 计算了 {len(GENRE_TOPSIS_REF)} 个流派的 TOPSIS 参考点")

_load_weights_and_refs()


def compute_comfort_score(features, genre):
    """对单张画计算舒适度得分 —— 与 step3 完全一致的流程"""
    optimal = GENRE_OPTIMAL_PARAMS.get(genre, DEFAULT_OPTIMAL)

    # 1. 高斯正向化
    normalized = {}
    for feat in FEATURE_COLS:
        x = features[feat]
        x_best = optimal[feat]['x_best']
        sigma = optimal[feat]['sigma']
        F = np.exp(-((x - x_best) ** 2) / (2 * sigma ** 2))
        normalized[feat] = float(F)

    # 2. 使用预计算的熵权法权重
    weights = GENRE_WEIGHTS.get(genre, {f: 1.0/len(FEATURE_COLS) for f in FEATURE_COLS})
    w = np.array([weights[f] for f in FEATURE_COLS])

    # 3. TOPSIS：用训练数据的正负理想解作为参考
    v = np.array([normalized[f] for f in FEATURE_COLS]) * w  # 加权向量

    if genre in GENRE_TOPSIS_REF:
        V_plus = GENRE_TOPSIS_REF[genre]['V_plus']
        V_minus = GENRE_TOPSIS_REF[genre]['V_minus']
    else:
        # 降级为简单加权平均
        score = float(np.sum(v)) * 100
        return round(score, 2), normalized

    d_plus = np.sqrt(np.sum((v - V_plus) ** 2))
    d_minus = np.sqrt(np.sum((v - V_minus) ** 2))
    denom = d_plus + d_minus
    Com = d_minus / denom if denom > 0 else 0

    score = round(float(Com * 100), 2)

    return score, normalized


# ================================================================
# ========== 加载模型和数据 ======================================
# ================================================================

print("正在加载模型...")
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
label_encoder = model_data['label_encoder']
print(f"模型加载完成！支持的流派: {list(label_encoder.classes_)}")

# 加载各流派平均特征值（用于雷达图对比）
df_all = pd.read_csv(FEATURES_CSV)
genre_avg_features = {}
for genre in df_all['流派'].unique():
    genre_data = df_all[df_all['流派'] == genre][FEATURE_COLS]
    genre_avg_features[genre] = {col: round(float(genre_data[col].mean()), 4) for col in FEATURE_COLS}

print(f"参考数据加载完成！共 {len(df_all)} 张画作的平均特征值")


# ================================================================
# ========== Flask 路由 ==========================================
# ================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    try:
        # 1. 读取图片
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes))

        # 2. 提取特征
        features = extract_features_from_image(pil_img)

        # 3. 分类预测
        X = np.array([[features[col] for col in FEATURE_COLS]])
        y_pred = model.predict(X)[0]
        y_proba = model.predict_proba(X)[0]

        predicted_genre = label_encoder.inverse_transform([y_pred])[0]
        max_confidence = float(np.max(y_proba))
        low_confidence = max_confidence < CONFIDENCE_THRESHOLD

        # 所有流派的概率
        all_proba = {label_encoder.inverse_transform([i])[0]: round(float(p), 4)
                     for i, p in enumerate(y_proba)}

        # 4. 计算舒适度
        comfort_score, normalized_values = compute_comfort_score(features, predicted_genre)

        # 5. 获取该流派的平均特征值（雷达图用）
        avg_features = genre_avg_features.get(predicted_genre, {})

        return jsonify({
            'genre': predicted_genre,
            'confidence': round(max_confidence * 100, 1),
            'low_confidence': low_confidence,
            'comfort_score': comfort_score,
            'features': features,
            'normalized_values': normalized_values,
            'genre_avg_features': avg_features,
            'all_proba': all_proba,
            'feature_cols': FEATURE_COLS,
        })

    except Exception as e:
        return jsonify({'error': f'分析失败: {str(e)}'}), 500


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  画作分派别视觉舒适度评价系统")
    print("  访问地址: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
