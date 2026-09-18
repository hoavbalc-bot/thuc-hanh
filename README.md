# ⚖️ Hệ Thống Web App Dự Báo Gian Lận Báo Cáo Tài Chính (Fraud Detection AI)

Ứng dụng web thông minh hỗ trợ kiểm toán viên, chuyên viên phân tích tài chính và nhà đầu tư phát hiện sớm các hành vi thao túng Báo cáo Tài chính (BCTC). Ứng dụng kết hợp giữa mô hình học máy **Hồi quy Logistic (Logistic Regression)** và **8 chỉ số Beneish M-Score**, được xây dựng trên nền tảng **Streamlit**.

![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

---

## 📌 1. Giới Thiệu & Cơ Sở Lý Thuyết

Mô hình dự báo dựa trên 8 chỉ số tài chính của **Giáo sư Messod Beneish (1999)** phản ánh các kỹ thuật làm đẹp số liệu kế toán:
1. **DSRI** (*Days Sales in Receivables Index*): Chỉ số kỳ thu tiền khách hàng. Cảnh báo việc ghi nhận doanh thu sớm hoặc doanh thu ảo.
2. **GMI** (*Gross Margin Index*): Chỉ số biên lợi nhuận gộp. Cảnh báo suy giảm năng lực sinh lời, tạo động cơ gian lận.
3. **AQI** (*Asset Quality Index*): Chỉ số chất lượng tài sản. Cảnh báo vốn hóa chi phí trái quy định vào tài sản dài hạn.
4. **SGI** (*Sales Growth Index*): Chỉ số tăng trưởng doanh thu. Doanh nghiệp tăng trưởng quá nóng có rủi ro tạo doanh thu ảo khi đà tăng chậm lại.
5. **DEPI** (*Depreciation Index*): Chỉ số khấu hao. Cảnh báo việc giảm tỷ lệ khấu hao để nâng cao lợi nhuận kỳ kế toán.
6. **SGAI** (*Sales, General and Administrative Expenses Index*): Chỉ số chi phí bán hàng & quản lý doanh nghiệp.
7. **TATA** (*Total Accruals to Total Assets*): Chỉ số dồn tích trên tổng tài sản. Phản ánh chênh lệch giữa lợi nhuận kế toán và dòng tiền thực thu từ HĐKD.
8. **LVGI** (*Leverage Index*): Chỉ số đòn bẩy tài chính. Đo lường mức độ gia tăng nợ vay và áp lực vi phạm cam kết khế ước tín dụng.

Ứng dụng huấn luyện đường ống hồi quy **`StandardScaler -> LogisticRegression`** trên dữ liệu thực nghiệm, tính toán xác suất rủi ro $P(\text{FRAUD}=1)$, đồng thời đối chiếu song song với điểm chuẩn **Beneish M-Score** ($M > -1.78$).

---

## 🌟 2. Các Tính Năng Nổi Bật

- **📊 Khám phá dữ liệu (EDA)**: Thống kê mô tả 8 chỉ số, tỷ lệ gian lận trong mẫu, biểu đồ ma trận tương quan (Correlation Heatmap) và biểu đồ phân phối hộp (Boxplot).
- **⚙️ Huấn luyện & Đánh giá mô hình**:
  - Trực quan hóa Ma trận nhầm lẫn (Confusion Matrix) với các chỉ số TN, FP, FN, TP.
  - Đánh giá toàn diện: Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC.
  - Biểu đồ đường cong ROC tương tác.
  - Bảng hệ số hồi quy (Coefficients), Odds Ratio ($e^\beta$) và giải thích tác động kinh tế/tài chính.
  - Tùy chỉnh tỷ lệ Train/Test và thanh trượt **Ngưỡng quyết định rủi ro (Decision Threshold)** từ 0.1 đến 0.9.
  - Xuất báo cáo đánh giá mô hình ra file Excel (`.xlsx`).
- **🔍 Dự báo đơn lẻ (1 Doanh nghiệp)**:
  - Nhập 8 chỉ số tài chính hoặc nạp nhanh các kịch bản mẫu (Doanh nghiệp an toàn, Gian lận cao, Ranh giới).
  - Đồng hồ đo rủi ro (Gauge Chart) hiển thị xác suất gian lận trực quan.
  - Tự động phân tích các chỉ số bất thường và đưa ra **khuyến nghị thủ tục kiểm toán chuyên sâu**.
- **📁 Dự báo hàng loạt (Batch CSV)**:
  - Tải file CSV danh sách nhiều công ty, hệ thống tự động quét và phân loại rủi ro.
  - Xếp hạng mức độ rủi ro từ cao xuống thấp.
  - Cung cấp file mẫu template và hỗ trợ xuất kết quả ra CSV.
- **📖 Cẩm nang kiến thức**: Tra cứu công thức, ý nghĩa kiểm toán và hướng dẫn đọc kết quả.

---

## 📂 3. Cấu Trúc Thư Mục Dự Án

Để đưa lên GitHub và triển khai trên Streamlit Cloud, thư mục dự án gồm các file chính sau:

```text
├── app.py                     # Mã nguồn chính của Web App Streamlit
├── requirements.txt           # Danh sách các thư viện Python cần cài đặt
├── MScore_data.csv            # Tập dữ liệu mẫu 8 chỉ số Beneish & FRAUD_FLAG
├── README.md                  # Hướng dẫn chi tiết sử dụng và triển khai
└── logistic_regression_mscore_colab.py # Mã nguồn thử nghiệm gốc trên Colab
```

---

## 🚀 4. Hướng Dẫn Chạy Cục Bộ (Local Development)

### Yêu cầu:
- Máy tính đã cài đặt **Python 3.9 trở lên**.

### Các bước thực hiện:

1. **Mở Terminal / PowerShell** tại thư mục dự án:
   ```bash
   cd "duong_dan_den_thu_muc_du_an"
   ```

2. **(Tùy chọn) Tạo môi trường ảo (Virtual Environment)**:
   ```bash
   python -m venv venv
   # Kích hoạt trên Windows:
   .\venv\Scripts\activate
   # Hoặc kích hoạt trên macOS/Linux:
   source venv/bin/activate
   ```

3. **Cài đặt các thư viện phụ thuộc**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Khởi chạy ứng dụng Streamlit**:
   ```bash
   streamlit run app.py
   ```
   Trình duyệt web sẽ tự động mở địa chỉ: `http://localhost:8501`.

---

## 🌐 5. Hướng Dẫn Tải Lên GitHub & Deploy Miễn Phí Trên Streamlit Cloud

### Bước 5.1: Đẩy mã nguồn lên GitHub

1. Tạo một tài khoản trên [GitHub.com](https://github.com/) (nếu chưa có).
2. Tạo một Repository mới (ví dụ đặt tên: `fraud-detection-beneish-streamlit`), chọn chế độ **Public**.
3. Tại thư mục dự án trên máy tính của bạn, mở terminal/command prompt và thực hiện:
   ```bash
   git init
   git add app.py requirements.txt MScore_data.csv README.md
   git commit -m "Khoi tao web app du bao gian lan BCTC tren Streamlit"
   git branch -M main
   git remote add origin https://github.com/TEN_GITHUB_CUA_BAN/fraud-detection-beneish-streamlit.git
   git push -u origin main
   ```
   *(Thay `TEN_GITHUB_CUA_BAN` bằng username GitHub thực tế của bạn).*

---

### Bước 5.2: Triển khai ứng dụng lên Streamlit Community Cloud (Hoàn toàn miễn phí)

1. Truy cập vào trang: [https://share.streamlit.io/](https://share.streamlit.io/)
2. Đăng nhập bằng tài khoản **GitHub** của bạn.
3. Bấm vào nút **"New app"** (hoặc **"Create app"**).
4. Điền các thông tin cấu hình như sau:
   - **Repository**: Chọn repo bạn vừa tải lên (ví dụ: `TEN_GITHUB_CUA_BAN/fraud-detection-beneish-streamlit`).
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Bấm nút **"Deploy!"**.
6. Chờ trong khoảng 1-2 phút để Streamlit Cloud tự động cài đặt các thư viện trong `requirements.txt` và khởi chạy server.
7. Khi hoàn tất, bạn sẽ nhận được một đường link web công khai dạng:
   `https://[ten-app-cua-ban].streamlit.app`
   Bạn có thể gửi liên kết này cho bất kỳ ai (giảng viên, đồng nghiệp, khách hàng) để họ truy cập và sử dụng trực tiếp trên trình duyệt mà không cần cài đặt gì thêm!

---

## 📊 6. Định Dạng Dữ Liệu Đầu Vào (Input Data Format)

Tập dữ liệu CSV dùng để huấn luyện hoặc dự báo cần có đầy đủ 8 cột chỉ số tài chính (viết hoa):

| Tên Cột | Kiểu Dữ Liệu | Diễn Giải |
| :--- | :--- | :--- |
| **DSRI** | Số thực (`float`) | Days Sales in Receivables Index |
| **GMI** | Số thực (`float`) | Gross Margin Index |
| **AQI** | Số thực (`float`) | Asset Quality Index |
| **SGI** | Số thực (`float`) | Sales Growth Index |
| **DEPI** | Số thực (`float`) | Depreciation Index |
| **SGAI** | Số thực (`float`) | Sales, General and Administrative expenses Index |
| **TATA** | Số thực (`float`) | Total Accruals to Total Assets |
| **LVGI** | Số thực (`float`) | Leverage Index |
| **FRAUD_FLAG** *(Bắt buộc nếu huấn luyện lại)* | Số nguyên (`int`: 0 hoặc 1) | 0: Không gian lận, 1: Gian lận |

---

## 📜 7. Bản Quyền & Trích Dẫn

- **Mô hình Beneish**: Beneish, M. D. (1999). *A model to detect earnings manipulation*. Financial Analysts Journal, 55(5), 24-36.
- Ứng dụng được phát triển phục vụ mục đích học thuật, nghiên cứu và hỗ trợ phân tích kiểm toán.
