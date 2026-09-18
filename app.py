# -*- coding: utf-8 -*-
"""
Hệ Thống Web App Dự Báo Gian Lận Báo Cáo Tài Chính (Financial Statement Fraud Detection)
Sử dụng Mô hình 8 chỉ số Beneish M-Score & Hồi quy Logistic (Logistic Regression)
Phát triển trên nền tảng Streamlit
"""

import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    classification_report
)

# ==============================================================================
# 1. CẤU HÌNH TRANG WEB STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="AI Dự Báo Gian Lận Báo Cáo Tài Chính",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện hiện đại, chuyên nghiệp
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .badge-fraud {
        background-color: #FEE2E2;
        color: #DC2626;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-clean {
        background-color: #DCFCE7;
        color: #16A34A;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 8 chỉ số chuẩn Beneish M-Score và biến mục tiêu
FEATURES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI"]
TARGET = "FRAUD_FLAG"

FEATURE_NAMES_VI = {
    "DSRI": "DSRI (Ngày thu tiền khách hàng)",
    "GMI": "GMI (Biên lợi nhuận gộp)",
    "AQI": "AQI (Chất lượng tài sản)",
    "SGI": "SGI (Tăng trưởng doanh thu)",
    "DEPI": "DEPI (Tỷ lệ khấu hao)",
    "SGAI": "SGAI (Chi phí BH & QLDN)",
    "TATA": "TATA (Dồn tích trên tổng tài sản)",
    "LVGI": "LVGI (Đòn bẩy tài chính)"
}

FEATURE_DESCRIPTIONS = {
    "DSRI": "Days Sales in Receivables Index: Đo lường tốc độ tăng của khoản phải thu so với doanh thu. Tỷ lệ > 1 cho thấy doanh thu ghi nhận tăng nhưng chưa thu được tiền.",
    "GMI": "Gross Margin Index: So sánh biên lợi nhuận gộp kỳ trước với kỳ này. Chỉ số > 1 cho thấy biên lợi nhuận đang suy giảm, tạo áp lực gian lận.",
    "AQI": "Asset Quality Index: Đo lường tỷ lệ các tài sản phi hiện hành ngoài TSCĐ. Chỉ số > 1 cảnh báo doanh nghiệp có thể đang vốn hóa chi phí.",
    "SGI": "Sales Growth Index: Tốc độ tăng trưởng doanh thu kỳ này so với kỳ trước. Doanh nghiệp tăng trưởng quá nóng thường có động cơ duy trì con số ảo.",
    "DEPI": "Depreciation Index: So sánh tỷ lệ trích khấu hao kỳ trước với kỳ này. Chỉ số > 1 cho thấy doanh nghiệp giảm tỷ lệ khấu hao để tăng lợi nhuận.",
    "SGAI": "Sales, General and Administrative expenses Index: Tỷ lệ chi phí bán hàng và QLDN trên doanh thu. Chỉ số > 1 cho thấy chi phí vận hành kém hiệu quả.",
    "TATA": "Total Accruals to Total Assets: Đo lường mức độ dồn tích (chênh lệch giữa lợi nhuận kế toán và dòng tiền HĐKD thực tế). Giá trị cao là dấu hiệu chất lượng lợi nhuận thấp.",
    "LVGI": "Leverage Index: Tỷ lệ tổng nợ trên tổng tài sản so với kỳ trước. Chỉ số > 1 cho thấy đòn bẩy tài chính tăng, làm tăng áp lực đáp ứng cam kết vay nợ."
}

# ==============================================================================
# 2. CÁC HÀM XỬ LÝ DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH
# ==============================================================================
@st.cache_data
def load_default_data():
    """Tải dữ liệu mặc định từ file MScore_data.csv đi kèm dự án."""
    possible_paths = [
        "MScore_data.csv",
        os.path.join(os.path.dirname(__file__), "MScore_data.csv")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

def clean_data(df: pd.DataFrame):
    """Làm sạch và kiểm tra dữ liệu đầu vào cho mô hình."""
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dữ liệu thiếu các cột bắt buộc: {missing_cols}")
    
    clean_df = df[FEATURES + [TARGET]].copy()
    for c in FEATURES + [TARGET]:
        clean_df[c] = pd.to_numeric(clean_df[c], errors="coerce")
    
    clean_df = clean_df.dropna().reset_index(drop=True)
    clean_df[TARGET] = clean_df[TARGET].astype(int)
    
    if not set(clean_df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Cột FRAUD_FLAG chỉ được chứa các giá trị 0 hoặc 1.")
    
    return clean_df

@st.cache_resource
def train_model(data: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """Huấn luyện Pipeline gồm StandardScaler và LogisticRegression."""
    X = data[FEATURES]
    y = data[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=5000, random_state=random_state))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline, X_train, X_test, y_train, y_test

def calculate_beneish_mscore(dsri, gmi, aqi, sgi, depi, sgai, tata, lvgi):
    """
    Tính điểm Beneish M-Score chuẩn theo công thức học thuật gốc (8 biến):
    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI - 0.172*SGAI + 4.037*TATA + 0.0327*LVGI
    Ngưỡng đánh giá chuẩn: M > -1.78 cảnh báo khả năng gian lận cao.
    """
    m_score = (
        -4.84
        + 0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.037 * tata
        + 0.0327 * lvgi
    )
    return m_score

# ==============================================================================
# 3. SIDEBAR: ĐIỀU HƯỚNG & CẤU HÌNH DỮ LIỆU
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/balance.png", width=70)
    st.markdown("## **Fraud Detection AI**")
    st.caption("Dự Báo Gian Lận BCTC với 8 Biến Beneish")
    st.markdown("---")
    
    # Menu chọn tính năng
    selected_menu = st.radio(
        "📌 **Chọn chức năng:**",
        [
            "📊 Khám phá Dữ liệu (EDA)",
            "⚙️ Huấn luyện & Đánh giá Mô hình",
            "🔍 Dự báo Đơn lẻ (1 Doanh nghiệp)",
            "📁 Dự báo Hàng loạt (Batch CSV)",
            "📖 Kiến thức 8 Chỉ số Beneish"
        ],
        index=2
    )
    
    st.markdown("---")
    st.markdown("### 📂 **Nguồn dữ liệu**")
    data_source_option = st.radio(
        "Dữ liệu huấn luyện:",
        ["Dữ liệu gốc (MScore_data.csv)", "Tải lên file CSV mới"],
        index=0
    )
    
    uploaded_file = None
    if data_source_option == "Tải lên file CSV mới":
        uploaded_file = st.file_uploader("Chọn file CSV chứa 8 biến và FRAUD_FLAG:", type=["csv"])
    
    st.markdown("---")
    st.markdown("### 🎛️ **Cấu hình mô hình**")
    test_size_val = st.slider("Tỷ lệ tập kiểm tra (Test size):", 0.10, 0.40, 0.20, 0.05)
    threshold_val = st.slider("Ngưỡng quyết định rủi ro (Threshold):", 0.10, 0.90, 0.50, 0.05, 
                             help="Nếu xác suất dự báo >= ngưỡng này, doanh nghiệp sẽ bị xếp vào diện Nguy cơ gian lận (1).")

# ==============================================================================
# 4. TẢI DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH TOÀN CỤC
# ==============================================================================
raw_data = None
if data_source_option == "Tải lên file CSV mới" and uploaded_file is not None:
    try:
        raw_data = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Đã tải: {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"Lỗi đọc file: {e}")
else:
    raw_data = load_default_data()

if raw_data is None:
    st.error("⚠️ Không tìm thấy file dữ liệu `MScore_data.csv`. Vui lòng tải file lên ở thanh bên trái!")
    st.stop()

try:
    data = clean_data(raw_data)
    model, X_train, X_test, y_train, y_test = train_model(data, test_size=test_size_val, random_state=42)
    logistic_step = model.named_steps["logistic"]
    intercept = logistic_step.intercept_[0]
    coefs = logistic_step.coef_[0]
except Exception as e:
    st.error(f"⚠️ Lỗi xử lý dữ liệu hoặc huấn luyện mô hình: {e}")
    st.stop()

# ==============================================================================
# 5. NỘI DUNG TỪNG TAB / CHỨC NĂNG
# ==============================================================================

# ------------------------------------------------------------------------------
# TAB 1: KHÁM PHÁ DỮ LIỆU (EDA)
# ------------------------------------------------------------------------------
if selected_menu == "📊 Khám phá Dữ liệu (EDA)":
    st.markdown('<div class="main-title">📊 Khám Phá & Phân Tích Dữ Liệu (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Khảo sát phân bố 8 chỉ số Beneish M-Score và nhãn gian lận trong tập dữ liệu.</div>', unsafe_allow_html=True)
    
    # Thống kê nhanh
    c1, c2, c3, c4 = st.columns(4)
    total_samples = len(data)
    fraud_count = int(data[TARGET].sum())
    clean_count = total_samples - fraud_count
    fraud_pct = fraud_count / total_samples * 100
    
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Tổng số quan sát</div>
            <div class="metric-value">{total_samples:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Bình thường (Không gian lận - 0)</div>
            <div class="metric-value" style="color: #16A34A;">{clean_count:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Bất thường (Gian lận - 1)</div>
            <div class="metric-value" style="color: #DC2626;">{fraud_count:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Tỷ lệ gian lận</div>
            <div class="metric-value" style="color: #E11D48;">{fraud_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    
    # Biểu đồ phân bố nhãn & Ma trận tương quan
    col_left, col_right = st.columns([1, 1.4])
    with col_left:
        st.markdown("#### 🎯 Phân Bố Nhãn Mục Tiêu (FRAUD_FLAG)")
        fig_pie = px.pie(
            names=["Không gian lận (0)", "Gian lận (1)"],
            values=[clean_count, fraud_count],
            color_discrete_sequence=["#10B981", "#EF4444"],
            hole=0.45
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_right:
        st.markdown("#### 🔗 Ma Trận Tương Quan (Correlation Heatmap)")
        corr_matrix = data.corr().round(2)
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title=""
        )
        fig_corr.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 📈 So Sánh Phân Bố Chỉ Số Giữa 2 Nhóm")
    feature_to_plot = st.selectbox("Chọn chỉ số để so sánh:", FEATURES, format_func=lambda x: f"{x} - {FEATURE_NAMES_VI[x]}")
    
    fig_box = px.box(
        data,
        x=TARGET,
        y=feature_to_plot,
        color=TARGET,
        color_discrete_map={0: "#10B981", 1: "#EF4444"},
        labels={TARGET: "Trạng thái", feature_to_plot: feature_to_plot},
        points="all",
        title=f"Phân bố chỉ số {feature_to_plot} theo nhóm thực tế (0: An toàn vs 1: Gian lận)"
    )
    fig_box.update_layout(height=420)
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Bảng số liệu mô tả
    with st.expander("📋 Xem Bảng Thống Kê Mô Tả Chi Tiết (Descriptive Statistics)"):
        st.dataframe(data.describe().round(4), use_container_width=True)
    
    with st.expander("🔍 Xem 20 Dòng Dữ Liệu Đầu Tiên"):
        st.dataframe(data.head(20), use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 2: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH
# ------------------------------------------------------------------------------
elif selected_menu == "⚙️ Huấn luyện & Đánh giá Mô hình":
    st.markdown('<div class="main-title">⚙️ Huấn Luyện & Đánh Giá Mô Hình Logistic Regression</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Đánh giá hiệu năng phân loại trên tập kiểm tra (Test set) với ngưỡng quyết định đã chọn.</div>', unsafe_allow_html=True)
    
    # Tính dự báo trên tập Test theo Threshold
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold_val).astype(int)
    
    # Các chỉ số đánh giá
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc = roc_auc_score(y_test, y_prob)
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Hiển thị Metrics chính
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{acc:.2%}", help="Tỷ lệ dự báo chính xác toàn bộ")
    m2.metric("Precision", f"{prec:.2%}", help="Độ chuẩn xác khi dự báo là Gian lận")
    m3.metric("Recall (Sensitivity)", f"{rec:.2%}", help="Tỷ lệ bắt trúng các vụ gian lận thực tế")
    m4.metric("F1-Score", f"{f1:.2%}", help="Trung bình điều hòa giữa Precision và Recall")
    m5.metric("Specificity", f"{spec:.2%}", help="Tỷ lệ phân loại đúng các trường hợp sạch")
    m6.metric("ROC AUC", f"{auc:.3f}", help="Diện tích dưới đường cong ROC")
    
    st.markdown("---")
    
    col_cm, col_roc = st.columns([1, 1.2])
    with col_cm:
        st.markdown("#### 🎯 Ma Trận Nhầm Lẫn (Confusion Matrix)")
        cm_display_data = pd.DataFrame(
            cm,
            index=["Thực tế: Sạch (0)", "Thực tế: Gian lận (1)"],
            columns=["Dự báo: Sạch (0)", "Dự báo: Gian lận (1)"]
        )
        fig_cm = px.imshow(
            cm_display_data,
            text_auto=True,
            color_continuous_scale="Reds",
            aspect="auto"
        )
        fig_cm.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_cm, use_container_width=True)
        
        st.caption(f"""
        - **TN (Đúng âm tính)**: {tn} công ty sạch được nhận diện chính xác.
        - **TP (Đúng dương tính)**: {tp} công ty gian lận được phát hiện thành công.
        - **FP (Báo động giả)**: {fp} công ty sạch bị nghi oan. (FPR = {fpr_val:.2%})
        - **FN (Bỏ sót gian lận)**: {fn} công ty gian lận bị lọt lưới. (FNR = {fnr_val:.2%})
        """)
    
    with col_roc:
        st.markdown("#### 📉 Đường Cong ROC (Receiver Operating Characteristic)")
        fpr_curve, tpr_curve, _ = roc_curve(y_test, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_curve, y=tpr_curve, mode='lines', name=f'Logistic Regression (AUC = {auc:.3f})', line=dict(color='#2563EB', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Ngẫu nhiên (AUC = 0.500)', line=dict(color='#9CA3AF', dash='dash')))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Recall)",
            height=350,
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(x=0.45, y=0.1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 📐 Bảng Hệ Số Hồi Quy (Coefficients) & Odds Ratio")
    
    coef_df = pd.DataFrame({
        "Chỉ số": FEATURES,
        "Hệ số (Beta)": coefs,
        "Odds Ratio (e^Beta)": np.exp(coefs),
        "Tác động đến khả năng gian lận": [
            "Làm TĂNG nguy cơ gian lận" if b > 0 else "Làm GIẢM nguy cơ gian lận" for b in coefs
        ]
    }).sort_values(by="Hệ số (Beta)", ascending=False).reset_index(drop=True)
    
    st.dataframe(
        coef_df.style.format({
            "Hệ số (Beta)": "{:.4f}",
            "Odds Ratio (e^Beta)": "{:.4f}"
        }).map(lambda v: 'color: red; font-weight: bold;' if "TĂNG" in str(v) else 'color: green;', subset=["Tác động đến khả năng gian lận"]),
        use_container_width=True
    )
    
    # Diễn giải phương trình Logit
    with st.expander("📝 Xem Phương Trình Hồi Quy Logistic Đầy Đủ (Đã chuẩn hóa StandardScaler)"):
        eq_str = f"**logit(P(FRAUD=1))** = {intercept:.4f}"
        for f, b in zip(FEATURES, coefs):
            sign = "+" if b >= 0 else "-"
            eq_str += f" {sign} {abs(b):.4f} × **{f}_std**"
        st.markdown(eq_str)
        st.info("💡 *Ghi chú*: Biến `_std` là giá trị của chỉ số sau khi được chuẩn hóa theo phương sai và độ lệch chuẩn của tập Train.")
    
    # Nút xuất file Excel kết quả
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        coef_df.to_excel(writer, sheet_name="Coefficients", index=False)
        cm_display_data.to_excel(writer, sheet_name="Confusion_Matrix")
        pd.DataFrame({
            "Chỉ tiêu": ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "FPR", "FNR", "AUC"],
            "Giá trị (%)": [acc*100, prec*100, rec*100, f1*100, spec*100, fpr_val*100, fnr_val*100, auc*100]
        }).to_excel(writer, sheet_name="Metrics", index=False)
    
    st.download_button(
        label="📥 Tải Toàn Bộ Kết Quả Đánh Giá Ra File Excel (.xlsx)",
        data=excel_buffer.getvalue(),
        file_name="Ket_Qua_Danh_Gia_Logistic_Regression.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ------------------------------------------------------------------------------
# TAB 3: DỰ BÁO ĐƠN LẺ (1 DOANH NGHIỆP)
# ------------------------------------------------------------------------------
elif selected_menu == "🔍 Dự báo Đơn lẻ (1 Doanh nghiệp)":
    st.markdown('<div class="main-title">🔍 Dự Báo Gian Lận Cho Từng Doanh Nghiệp</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Nhập 8 chỉ số tài chính để hệ thống tính toán xác suất rủi ro bằng Machine Learning và điểm chuẩn Beneish M-Score.</div>', unsafe_allow_html=True)
    
    st.markdown("##### ⚡ Nạp Nhanh Hồ Sơ Mẫu Để Thử Nghiệm:")
    col_p1, col_p2, col_p3 = st.columns(3)
    
    # Session state để lưu trữ giá trị nhập liệu
    if "input_vals" not in st.session_state:
        st.session_state.input_vals = {
            "DSRI": 1.02, "GMI": 0.95, "AQI": 0.88, "SGI": 1.10,
            "DEPI": 0.98, "SGAI": 0.95, "TATA": 0.02, "LVGI": 1.05
        }
    
    if col_p1.button("🟢 Mẫu 1: Doanh nghiệp lành mạnh (An toàn)", use_container_width=True):
        st.session_state.input_vals = {
            "DSRI": 0.85, "GMI": 0.92, "AQI": 0.80, "SGI": 1.05,
            "DEPI": 1.00, "SGAI": 0.90, "TATA": 0.01, "LVGI": 0.95
        }
        st.rerun()
        
    if col_p2.button("🔴 Mẫu 2: Nguy cơ gian lận rất cao (Bất thường)", use_container_width=True):
        st.session_state.input_vals = {
            "DSRI": 1.75, "GMI": 1.65, "AQI": 1.45, "SGI": 1.85,
            "DEPI": 1.30, "SGAI": 1.25, "TATA": 0.18, "LVGI": 1.40
        }
        st.rerun()
        
    if col_p3.button("🟡 Mẫu 3: Doanh nghiệp vùng ranh giới", use_container_width=True):
        st.session_state.input_vals = {
            "DSRI": 1.20, "GMI": 1.15, "AQI": 1.05, "SGI": 1.30,
            "DEPI": 1.05, "SGAI": 1.05, "TATA": 0.08, "LVGI": 1.10
        }
        st.rerun()

    st.write("")
    
    # Form nhập liệu 8 biến
    with st.form("single_predict_form"):
        st.markdown("#### 📋 Thông Số 8 Chỉ Số Tài Chính Beneish")
        r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
        r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
        
        with r1_c1:
            val_dsri = st.number_input(
                "1. DSRI (Khoản phải thu)",
                value=float(st.session_state.input_vals["DSRI"]),
                step=0.05, format="%.3f",
                help=FEATURE_DESCRIPTIONS["DSRI"]
            )
        with r1_c2:
            val_gmi = st.number_input(
                "2. GMI (Biên lãi gộp)",
                value=float(st.session_state.input_vals["GMI"]),
                step=0.05, format="%.3f",
                help=FEATURE_DESCRIPTIONS["GMI"]
            )
        with r1_c3:
            val_aqi = st.number_input(
                "3. AQI (Chất lượng tài sản)",
                value=float(st.session_state.input_vals["AQI"]),
                step=0.05, format="%.3f",
                help=FEATURE_DESCRIPTIONS["AQI"]
            )
        with r1_c4:
            val_sgi = st.number_input(
                "4. SGI (Tăng trưởng DT)",
                value=float(st.session_state.input_vals["SGI"]),
                step=0.05, format="%.3f",
                help=FEATURE_DESCRIPTIONS["SGI"]
            )
            
        with r2_c1:
            val_depi = st.number_input(
                "5. DEPI (Tỷ lệ khấu hao)",
                value=float(st.session_state.input_vals["DEPI"]),
                step=0.05, format="%.3f",
                help=FEATURE_DESCRIPTIONS["DEPI"]
            )
        with r2_c2:
            val_sgai = st.number_input(
                "6. SGAI (Chi phí BH & QL)",
                value=float(st.session_state.input_vals["SGAI"]),
                step=0.05, format="%.3f",
                help=FEATURE_DESCRIPTIONS["SGAI"]
            )
        with r2_c3:
            val_tata = st.number_input(
                "7. TATA (Dồn tích / Tổng TS)",
                value=float(st.session_state.input_vals["TATA"]),
                step=0.02, format="%.4f",
                help=FEATURE_DESCRIPTIONS["TATA"]
            )
        with r2_c4:
            val_lvgi = st.number_input(
                "8. LVGI (Đòn bẩy tài chính)",
                value=float(st.session_state.input_vals["LVGI"]),
                step=0.05, format="%.3f",
                help=FEATURE_DESCRIPTIONS["LVGI"]
            )
            
        submitted = st.form_submit_button("🚀 PHÂN TÍCH & DỰ BÁO RỦI RO", use_container_width=True)
    
    # Thực hiện dự báo khi bấm nút hoặc hiển thị mặc định
    input_sample = pd.DataFrame([{
        "DSRI": val_dsri, "GMI": val_gmi, "AQI": val_aqi, "SGI": val_sgi,
        "DEPI": val_depi, "SGAI": val_sgai, "TATA": val_tata, "LVGI": val_lvgi
    }])
    
    # Tính xác suất từ mô hình ML
    prob_fraud = model.predict_proba(input_sample)[0, 1]
    is_fraud_ml = prob_fraud >= threshold_val
    
    # Tính điểm Beneish M-Score chuẩn
    m_score_val = calculate_beneish_mscore(
        val_dsri, val_gmi, val_aqi, val_sgi,
        val_depi, val_sgai, val_tata, val_lvgi
    )
    is_fraud_beneish = m_score_val > -1.78
    
    st.markdown("---")
    st.markdown("### 📊 Kết Quả Đánh Giá Rủi Ro")
    
    col_res_l, col_res_r = st.columns([1.1, 1])
    
    with col_res_l:
        # Đồng hồ đo Gauge Chart
        gauge_color = "#10B981" if prob_fraud < 0.4 else ("#F59E0B" if prob_fraud < threshold_val else "#EF4444")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_fraud * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"<b>Xác Suất Gian Lận (Mô hình ML)</b><br><span style='font-size:0.8em;color:gray'>Ngưỡng cảnh báo: {threshold_val*100:.0f}%</span>", 'font': {'size': 18}},
            number={'suffix': "%", 'font': {'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': gauge_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#DCFCE7'},
                    {'range': [40, threshold_val*100], 'color': '#FEF3C7'},
                    {'range': [threshold_val*100, 100], 'color': '#FEE2E2'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold_val * 100
                }
            }
        ))
        fig_gauge.update_layout(height=320, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with col_res_r:
        st.markdown("#### 🎯 Kết Luận Đánh Giá:")
        if is_fraud_ml:
            st.error(f"""
            ### 🚨 CẢNH BÁO: NGUY CƠ GIAN LẬN CAO!
            - **Xác suất mô hình**: **{prob_fraud:.1%}** (Vượt ngưỡng an toàn {threshold_val:.0%})
            - **Đánh giá**: Doanh nghiệp có dấu hiệu thao túng số liệu tài chính rõ rệt.
            """)
        else:
            st.success(f"""
            ### ✅ AN TOÀN: RỦI RO GIAN LẬN THẤP
            - **Xác suất mô hình**: **{prob_fraud:.1%}** (Dưới ngưỡng cảnh báo {threshold_val:.0%})
            - **Đánh giá**: Các chỉ số tài chính nằm trong biên độ hoạt động bình thường.
            """)
        
        # So sánh với mô hình Beneish gốc
        st.markdown("#### ⚖️ Đối Chiếu Với Chuẩn Beneish Gốc:")
        if is_fraud_beneish:
            st.warning(f"**Điểm Beneish M-Score:** `{m_score_val:.3f}` > `-1.78` ➔ **Cảnh báo Thao túng BCTC**")
        else:
            st.info(f"**Điểm Beneish M-Score:** `{m_score_val:.3f}` ≤ `-1.78` ➔ **Doanh nghiệp không thao túng**")
    
    # Khuyến nghị kiểm toán dựa trên các chỉ số
    st.markdown("---")
    st.markdown("#### 🩺 Phân Tích Dấu Hiệu & Thủ Tục Kiểm Toán Cần Chú Ý")
    
    warnings = []
    if val_dsri > 1.25:
        warnings.append(f"⚠️ **DSRI cao ({val_dsri:.3f} > 1.25):** Tốc độ tăng khoản phải thu cao bất thường so với doanh thu. Kiểm toán viên cần gửi thư xác nhận công nợ độc lập và kiểm tra cắt kỳ doanh thu bán hàng cuối niên độ.")
    if val_gmi > 1.15:
        warnings.append(f"⚠️ **GMI cao ({val_gmi:.3f} > 1.15):** Biên lợi nhuận gộp suy giảm mạnh so với kỳ trước. Doanh nghiệp có thể có động cơ hoãn ghi nhận giá vốn hàng bán hoặc tạo doanh thu ảo để giữ số liệu đẹp.")
    if val_aqi > 1.20:
        warnings.append(f"⚠️ **AQI cao ({val_aqi:.3f} > 1.20):** Tài sản phi hiện hành khác tăng mạnh. Cần kiểm tra kỹ các khoản chi phí có đang bị vốn hóa trái quy định (ví dụ ghi nhận chi phí vào chi phí trả trước dài hạn hay TSCĐ vô hình).")
    if val_sgi > 1.35:
        warnings.append(f"⚠️ **SGI cao ({val_sgi:.3f} > 1.35):** Doanh thu tăng trưởng nóng > 35%. Rà soát kỹ các giao dịch với các bên liên quan và các hợp đồng bán hàng lớn vào tháng 12.")
    if val_depi > 1.15:
        warnings.append(f"⚠️ **DEPI cao ({val_depi:.3f} > 1.15):** Tỷ lệ khấu hao giảm. Kiểm tra xem doanh nghiệp có thay đổi thời gian hữu dụng hoặc phương pháp tính khấu hao để giảm chi phí hay không.")
    if val_tata > 0.10:
        warnings.append(f"⚠️ **TATA cao ({val_tata:.4f} > 0.10):** Lợi nhuận kế toán cao hơn nhiều so với dòng tiền thực thu từ HĐKD. Chất lượng lợi nhuận kém, lợi nhuận chủ yếu đến từ các khoản dồn tích.")
    if val_lvgi > 1.20:
        warnings.append(f"⚠️ **LVGI cao ({val_lvgi:.3f} > 1.20):** Đòn bẩy nợ tăng vọt. Áp lực tuân thủ các cam kết vay (covenants) của ngân hàng rất lớn, làm tăng động cơ làm đẹp số liệu.")
        
    if warnings:
        for w in warnings:
            st.markdown(w)
    else:
        st.success("✨ Không có chỉ số đơn lẻ nào vượt ngưỡng cảnh báo nghiêm trọng. Các tỷ số tài chính tương đối đồng đều và ổn định.")


# ------------------------------------------------------------------------------
# TAB 4: DỰ BÁO HÀNG LOẠT (BATCH CSV)
# ------------------------------------------------------------------------------
elif selected_menu == "📁 Dự báo Hàng loạt (Batch CSV)":
    st.markdown('<div class="main-title">📁 Dự Báo Hàng Loạt Từ File CSV</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Tải lên danh sách nhiều doanh nghiệp để hệ thống tự động chấm điểm rủi ro và xếp hạng mức độ gian lận.</div>', unsafe_allow_html=True)
    
    # Nút tải file mẫu
    template_df = pd.DataFrame([
        {"Company": "Cong ty ABC", "DSRI": 1.45, "GMI": 1.20, "AQI": 1.15, "SGI": 1.50, "DEPI": 1.10, "SGAI": 1.05, "TATA": 0.12, "LVGI": 1.25},
        {"Company": "Cong ty XYZ", "DSRI": 0.90, "GMI": 0.95, "AQI": 0.85, "SGI": 1.05, "DEPI": 0.98, "SGAI": 0.92, "TATA": 0.01, "LVGI": 0.98}
    ])
    csv_template = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Tải File Mẫu (CSV Template) Để Điền Số Liệu",
        data=csv_template,
        file_name="Template_Du_Bao_Gian_Lan.csv",
        mime="text/csv"
    )
    
    st.write("")
    batch_file = st.file_uploader("Chọn file CSV chứa 8 cột chỉ số (DSRI, GMI, AQI, SGI, DEPI, SGAI, TATA, LVGI):", type=["csv"])
    
    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            missing_batch_cols = [c for c in FEATURES if c not in batch_df.columns]
            
            if missing_batch_cols:
                st.error(f"❌ File tải lên thiếu các cột bắt buộc: {missing_batch_cols}")
            else:
                st.success(f"✅ Đọc thành công {len(batch_df)} doanh nghiệp từ file!")
                
                # Tiến hành dự báo
                X_batch = batch_df[FEATURES].copy()
                for c in FEATURES:
                    X_batch[c] = pd.to_numeric(X_batch[c], errors="coerce").fillna(0)
                
                # Dự báo ML
                probs = model.predict_proba(X_batch)[:, 1]
                preds = (probs >= threshold_val).astype(int)
                
                # Tính M-Score chuẩn
                m_scores = [
                    calculate_beneish_mscore(
                        row["DSRI"], row["GMI"], row["AQI"], row["SGI"],
                        row["DEPI"], row["SGAI"], row["TATA"], row["LVGI"]
                    )
                    for _, row in X_batch.iterrows()
                ]
                
                result_df = batch_df.copy()
                result_df["Xác_Suất_Gian_Lận_%"] = (probs * 100).round(2)
                result_df["Dự_Báo_ML"] = ["🔴 NGUY CƠ CAO" if p == 1 else "🟢 An toàn" for p in preds]
                result_df["Beneish_MScore"] = np.round(m_scores, 3)
                result_df["Cảnh_Báo_MScore"] = ["⚠️ Gian lận" if m > -1.78 else "✅ Bình thường" for m in m_scores]
                
                # Sắp xếp rủi ro cao nhất lên đầu
                result_df = result_df.sort_values(by="Xác_Suất_Gian_Lận_%", ascending=False).reset_index(drop=True)
                
                # Tóm tắt
                num_flagged = sum(preds)
                c_b1, c_b2, c_b3 = st.columns(3)
                c_b1.metric("Tổng Doanh Nghiệp Đã Quét", len(result_df))
                c_b2.metric("Số Doanh Nghiệp Bị Cảnh Báo", num_flagged, delta=f"{num_flagged/len(result_df):.1%}", delta_color="inverse")
                c_b3.metric("Số Doanh Nghiệp An Toàn", len(result_df) - num_flagged)
                
                st.markdown("#### 📋 Danh Sách Doanh Nghiệp Xếp Theo Mức Độ Rủi Ro")
                st.dataframe(result_df, use_container_width=True)
                
                # Nút tải kết quả
                res_csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Tải Kết Quả Dự Báo (CSV)",
                    data=res_csv,
                    file_name="Ket_Qua_Du_Bao_Gian_Lan_Hang_Loat.csv",
                    mime="text/csv"
                )
        except Exception as err:
            st.error(f"Lỗi khi xử lý file: {err}")


# ------------------------------------------------------------------------------
# TAB 5: KIẾN THỨC & CẨM NANG 8 CHỈ SỐ BENEISH
# ------------------------------------------------------------------------------
elif selected_menu == "📖 Kiến thức 8 Chỉ số Beneish":
    st.markdown('<div class="main-title">📖 Cẩm Nang 8 Chỉ Số Beneish M-Score & Kiểm Toán Gian Lận</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Tài liệu tham khảo chuyên sâu dành cho kiểm toán viên, chuyên viên tài chính và nhà đầu tư.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 1. Tổng Quan Về Mô Hình Beneish M-Score
    Mô hình **Beneish M-Score** được phát triển bởi Giáo sư Messod Beneish (1999) tại Trường Kinh doanh Kelley (Đại học Indiana). Mô hình sử dụng 8 chỉ số tài chính được tính toán từ Báo cáo Tài chính của hai niên độ liên tiếp để phát hiện khả năng một công ty đang **thao túng lợi nhuận (Earnings Manipulation)**.
    
    Công thức hồi quy học thuật gốc:
    $$M = -4.84 + 0.920 \\times DSRI + 0.528 \\times GMI + 0.404 \\times AQI + 0.892 \\times SGI + 0.115 \\times DEPI - 0.172 \\times SGAI + 4.037 \\times TATA + 0.0327 \\times LVGI$$
    
    *Quy tắc quyết định*:
    - **$M > -1.78$**: Doanh nghiệp có xác suất gian lận cao (Manipulator).
    - **$M \\le -1.78$**: Doanh nghiệp ít có khả năng gian lận (Non-manipulator).
    """)
    
    st.markdown("---")
    st.markdown("### 2. Chi Tiết Ý Nghĩa 8 Chỉ Số Thành Phần")
    
    indicators_guide = [
        {
            "Chỉ số": "DSRI",
            "Tên đầy đủ": "Days Sales in Receivables Index",
            "Công thức": "(Khoản phải thu_t / Doanh thu_t) / (Khoản phải thu_{t-1} / Doanh thu_{t-1})",
            "Ý nghĩa kiểm toán": "Đo lường sự thay đổi của khoản phải thu so với doanh thu. Nếu DSRI > 1.0, các khoản phải thu đang tăng nhanh hơn doanh thu, cảnh báo khả năng ghi nhận doanh thu ảo hoặc nới lỏng tín dụng thương mại quá mức."
        },
        {
            "Chỉ số": "GMI",
            "Tên đầy đủ": "Gross Margin Index",
            "Công thức": "[(Doanh thu_{t-1} - Giá vốn_{t-1}) / Doanh thu_{t-1}] / [(Doanh thu_t - Giá vốn_t) / Doanh thu_t]",
            "Ý nghĩa kiểm toán": "So sánh biên lợi nhuận gộp kỳ trước với kỳ này. GMI > 1 cho thấy khả năng sinh lời đang giảm sút, tạo áp lực lớn khiến ban điều hành thực hiện các thủ thuật thổi phồng lợi nhuận."
        },
        {
            "Chỉ số": "AQI",
            "Tên đầy đủ": "Asset Quality Index",
            "Công thức": "[1 - (Tài sản ngắn hạn_t + TSCĐ_t + Chứng khoán_t)/Tổng TS_t] / [1 - (Tài sản ngắn hạn_{t-1} + TSCĐ_{t-1} + Chứng khoán_{t-1})/Tổng TS_{t-1}]",
            "Ý nghĩa kiểm toán": "Đo lường tỷ trọng tài sản phi vật chất, tài sản dài hạn khác. AQI > 1 cho thấy doanh nghiệp có thể đang trì hoãn chi phí bằng cách vốn hóa vào tài sản thay vì đưa vào chi phí trong kỳ."
        },
        {
            "Chỉ số": "SGI",
            "Tên đầy đủ": "Sales Growth Index",
            "Công thức": "Doanh thu_t / Doanh thu_{t-1}",
            "Ý nghĩa kiểm toán": "Tăng trưởng doanh thu không tự nó là gian lận, nhưng các công ty có SGI cao thường chịu áp lực lớn từ thị trường và nhà đầu tư để tiếp tục duy trì kỳ vọng, dẫn tới việc làm đẹp số liệu khi tăng trưởng chững lại."
        },
        {
            "Chỉ số": "DEPI",
            "Tên đầy đủ": "Depreciation Index",
            "Công thức": "(Tỷ lệ khấu hao_{t-1}) / (Tỷ lệ khấu hao_t)",
            "Ý nghĩa kiểm toán": "DEPI > 1 biểu thị tốc độ trích khấu hao tài sản bị giảm sút. Doanh nghiệp có thể đã kéo dài thời gian khấu hao một cách tùy tiện để cắt giảm chi phí kỳ này."
        },
        {
            "Chỉ số": "SGAI",
            "Tên đầy đủ": "Sales, General and Administrative Expenses Index",
            "Công thức": "(Chi phí BH & QLDN_t / Doanh thu_t) / (Chi phí BH & QLDN_{t-1} / Doanh thu_{t-1})",
            "Ý nghĩa kiểm toán": "Đo lường tính hiệu quả của chi phí vận hành. SGAI > 1 cho thấy chi phí bán hàng và quản lý tăng nhanh hơn doanh thu, làm giảm lợi nhuận thuần."
        },
        {
            "Chỉ số": "TATA",
            "Tên đầy đủ": "Total Accruals to Total Assets",
            "Công thức": "(Lợi nhuận thuần từ HĐKD - Dòng tiền thuần từ HĐKD) / Tổng tài sản",
            "Ý nghĩa kiểm toán": "Phản ánh chất lượng dòng tiền. Lợi nhuận kế toán cao nhưng không có dòng tiền thu về thực tế (TATA cao) là dấu hiệu cảnh báo đỏ nguy hiểm nhất trong kiểm toán tài chính."
        },
        {
            "Chỉ số": "LVGI",
            "Tên đầy đủ": "Leverage Index",
            "Công thức": "(Tổng nợ_t / Tổng tài sản_t) / (Tổng nợ_{t-1} / Tổng tài sản_{t-1})",
            "Ý nghĩa kiểm toán": "LVGI > 1 cho thấy doanh nghiệp gia tăng đòn bẩy nợ vay, dẫn tới nguy cơ vi phạm các điều khoản khế ước vay ngân hàng, thúc đẩy động cơ thao túng BCTC."
        }
    ]
    
    st.table(pd.DataFrame(indicators_guide)[["Chỉ số", "Tên đầy đủ", "Ý nghĩa kiểm toán"]])
    
    st.markdown("---")
    st.markdown("""
    ### 3. Ưu Điểm Khi Kết Hợp Với Hồi Quy Logistic (Machine Learning)
    Trong khi công thức Beneish truyền thống dùng trọng số cố định từ mẫu nghiên cứu tại thị trường Mỹ năm 1999, việc ứng dụng **Hồi quy Logistic (Logistic Regression)** kèm **Chuẩn hóa dữ liệu (StandardScaler)** trên tập dữ liệu thực nghiệm giúp:
    1. **Tự động tối ưu hóa trọng số**: Phù hợp với đặc thù phân bố dữ liệu của mẫu hiện tại.
    2. **Xác suất rủi ro linh hoạt**: Cung cấp xác suất $P(\text{FRAUD}=1)$ từ 0% đến 100% thay vì chỉ kết luận nhị phân có/không.
    3. **Điều chỉnh ngưỡng phát hiện (Threshold Tuning)**: Cho phép kiểm toán viên chủ động hạ thấp ngưỡng (ví dụ 0.35) để tối đa hóa độ nhạy (Recall), hạn chế tối đa việc bỏ sót rủi ro.
    """)

# ==============================================================================
# 6. FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    "<center><small style='color: #64748B;'>Dự Báo Gian Lận Báo Cáo Tài Chính © 2026 | Phát triển với Streamlit, Scikit-Learn & Plotly</small></center>",
    unsafe_allow_html=True
)
