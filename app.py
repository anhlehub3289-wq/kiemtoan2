---

### 3. File `app.py`
*Mã nguồn chính của ứng dụng web được tối ưu hóa giao diện đa tab, tích hợp biểu đồ tương tác Plotly và cơ chế Cache tăng tốc.*

```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import io

# Cấu hình trang giao diện rộng rãi (Wide mode)
st.set_page_config(layout="wide", page_title="Hệ thống Phát hiện Gian lận Giao dịch", page_icon="🛡️")

st.title("🛡️ Hệ thống Phát hiện Giao dịch Bất thường (Isolation Forest)")
st.markdown("Ứng dụng hỗ trợ kiểm toán viên phân tích rủi ro dòng tiền lớn và gắn cờ giao dịch nghi vấn tự động.")

# 1. Pipeline xử lý và nạp dữ liệu (Sử dụng Cache để tăng tốc)
@st.cache_data
def load_and_preprocess_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("transactions_Q1_demo.csv")
    
    # Chuẩn hóa ngày tháng và trích xuất giờ giao dịch
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], dayfirst=True, errors='coerce')
    df['gio_giao_dich'] = df['transaction_date'].dt.hour
    df['gio_giao_dich'] = df['gio_giao_dich'].fillna(12).astype(int)
    
    # Chuyển đổi trạng thái nhân viên thành số nguyên (0 hoặc 1)
    df['co_nhan_vien'] = df['is_employee'].astype(int)
    return df

# 2. Pipeline huấn luyện mô hình học máy (Lưu trữ mô hình trong RAM tránh train lại liên tục)
@st.cache_resource
def train_anomaly_model(X_train, n_estimators, contamination):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_scaled)
    return iso, scaler

# --- KHU VỰC THANH ĐIỀU HƯỚNG (SIDEBAR) ---
st.sidebar.header("📁 Cấu hình Dữ liệu & Mô hình")
uploaded_file = st.sidebar.file_uploader("Tải lên tệp CSV giao dịch mới", type=["csv"])

try:
    df_raw = load_and_preprocess_data(uploaded_file)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Tham số Isolation Forest")
    contamination = st.sidebar.slider("Tỷ lệ bất thường giả định (Contamination)", 0.005, 0.05, 0.01, step=0.005, help="Tỷ lệ giao dịch lỗi/gian lận dự kiến trong tập dữ liệu.")
    n_estimators = st.sidebar.number_input("Số cây quyết định (n_estimators)", min_value=50, max_value=500, value=200, step=50)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Bộ lọc nhanh")
    available_locations = ["Tất cả"] + list(df_raw['location'].dropna().unique())
    selected_location = st.sidebar.selectbox("Lọc theo Chi nhánh", available_locations)

    df = df_raw.copy()
    if selected_location != "Tất cả":
        df = df[df['location'] == selected_location]

    # --- KHU VỰC HIỂN THỊ CHÍNH (TABS) ---
    tab1, tab2, tab3 = st.tabs(["📊 Thống kê Tổng quan (EDA)", "🔍 Quét & Phát hiện Bất thường", "🔮 Kiểm tra Giao dịch Đơn lẻ"])

    # --- TAB 1: THỐNG KÊ TỔNG QUAN ---
    with tab1:
        st.subheader("📊 Phân tích Khám phá Dữ liệu Giao dịch")
        q99 = df['amount'].quantile(0.99)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng số giao dịch", f"{len(df):,}")
        col2.metric("Tổng dòng tiền (VND)", f"{df['amount'].sum():,.0f}")
        col3.metric("Giao dịch từ Nhân viên", f"{df['co_nhan_vien'].sum():,}")
        col4.metric("Ngưỡng GD lớn (Phân vị 99%)", f"{int(q99):,} VND")
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### Tần suất Giao dịch theo các khung giờ")
            hour_counts = df['gio_giao_dich'].value_counts().sort_index().reset_index()
            hour_counts.columns = ['Giờ trong ngày', 'Số lượng giao dịch']
            fig_hour = px.bar(hour_counts, x='Giờ trong ngày', y='Số lượng giao dịch', color='Số lượng giao dịch', color_continuous_scale='Blues')
            st.plotly_chart(fig_hour, use_container_width=True)
            
        with c2:
            st.markdown("#### Cơ cấu Giao dịch theo Kênh (Channel)")
            channel_counts = df['channel'].value_counts().reset_index()
            channel_counts.columns = ['Kênh', 'Số lượng']
            fig_channel = px.pie(channel_counts, values='Số lượng', names='Kênh', hole=0.4)
            st.plotly_chart(fig_channel, use_container_width=True)

    # --- TAB 2: QUÉT BẤT THƯỜNG VỚI MÔ HÌNH ---
    with tab2:
        st.subheader("🔍 Kết quả Phát hiện Giao dịch Bất thường")
        X = df[['amount', 'gio_giao_dich', 'co_nhan_vien']]
        
        if st.button("🚀 Kích hoạt huấn luyện và quét rủi ro", type="primary"):
            with st.spinner("Hệ thống đang phân tích cấu trúc dữ liệu..."):
                model, scaler = train_anomaly_model(X, n_estimators, contamination)
                
                X_scaled = scaler.transform(X)
                df["anomaly_score"] = model.decision_function(X_scaled)
                df["is_anomaly"] = model.predict(X_scaled) == -1
                
                df_bat_thuong = df[df['is_anomaly'] == True]
                st.success(f"Quá trình phân tích hoàn tất! Phát hiện thấy {len(df_bat_thuong)} giao dịch bất thường.")
                
                # Trực quan hóa bằng biểu đồ phân tán trực quan sinh động
                df['Phân loại'] = df['is_anomaly'].map({True: 'Bất thường (Anomaly)', False: 'Bình thường (Normal)'})
                fig_scatter = px.scatter(df, x='gio_giao_dich', y='amount', color='Phân loại',
                                         color_discrete_map={'Bình thường (Normal)': '#3B82F6', 'Bất thường (Anomaly)': '#EF4444'},
                                         hover_data=['transaction_id', 'channel', 'location'],
                                         labels={'gio_giao_dich': 'Giờ giao dịch', 'amount': 'Số tiền (VND)'})
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Hiển thị bảng danh sách rủi ro
                st.markdown("#### 📋 Danh sách chi tiết các giao dịch rủi ro cao cần thanh tra")
                if len(df_bat_thuong) > 0:
                    st.dataframe(df_bat_thuong[['transaction_id', 'transaction_date', 'amount', 'transaction_type', 'channel', 'location', 'is_employee', 'anomaly_score']].sort_values(by='anomaly_score'), use_container_width=True)
                    
                    # Nút tải file Excel báo cáo
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_bat_thuong.to_excel(writer, index=False, sheet_name='Báo_cáo_rủi_ro')
                    processed_data = output.getvalue()
                    
                    st.download_button(
                        label="📥 Xuất dữ liệu Báo cáo Giao dịch Bất thường (Excel)",
                        data=processed_data,
                        file_name="bao_cao_giao_dich_bat_thuong.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.info("Không tìm thấy giao dịch bất thường nào phù hợp với bộ lọc hiện tại.")
        else:
            st.info("Nhấn nút phía trên để bắt đầu huấn luyện Isolation Forest và rà soát rủi ro.")

    # --- TAB 3: KIỂM TRA NHANH REAL-TIME ĐƠN LẺ ---
    with tab3:
        st.subheader("🔮 Kiểm tra Giao dịch Đơn lẻ Thời gian thực")
        st.markdown("Nhập thông số của một giao dịch đơn lẻ mới phát sinh để thẩm định nhanh độ an toàn:")
        
        with st.form("inference_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                input_amount = st.number_input("Số tiền giao dịch (VND)", min_value=0, value=1000000, step=50000)
            with c2:
                input_hour = st.slider("Khung giờ giao dịch (0h - 23h)", 0, 23, 12)
            with c3:
                input_employee = st.selectbox("Đối tượng thực hiện giao dịch là nhân viên?", ["KHÔNG", "CÓ"])
            
            submit_btn = st.form_submit_button("🔍 Phân tích Mức độ Rủi ro", type="primary")
            
            if submit_btn:
                # Lấy dữ liệu nền để chuẩn hóa bộ Scaler
                X_base = df_raw[['amount', 'gio_giao_dich', 'co_nhan_vien']]
                model, scaler = train_anomaly_model(X_base, n_estimators, contamination)
                
                # Biến đổi dữ liệu đầu vào đơn lẻ
                is_emp_int = 1 if input_employee == "CÓ" else 0
                single_input = np.array([[input_amount, input_hour, is_emp_int]])
                single_input_scaled = scaler.transform(single_input)
                
                prediction = model.predict(single_input_scaled)[0]
                score = model.decision_function(single_input_scaled)[0]
                
                st.markdown("### Kết quả Phân tích từ Hệ thống:")
                if prediction == -1:
                    st.error(f"🚨 CẢNH BÁO: Giao dịch này có dấu hiệu BẤT THƯỜNG! (Điểm rủi ro: {score:.4f})")
                    st.markdown("- **Khuyến nghị:** Giao dịch này nằm ngoài vùng phân phối an toàn chuẩn. Kiểm toán viên nên kiểm tra chéo chứng từ giao dịch thực tế.")
                else:
                    st.success(f"✅ AN TOÀN: Giao dịch nằm trong ngưỡng hành vi bình thường. (Điểm rủi ro: {score:.4f})")

except Exception as e:
    st.error(f"Đã xảy ra lỗi hệ thống hoặc không tìm thấy tệp 'transactions_Q1_demo.csv'. Vui lòng kiểm tra lại cấu trúc file! Chi tiết: {e}")
