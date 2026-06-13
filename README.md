# 🇻🇳 AIDEOM-VN Dashboard

> **Hệ thống hỗ trợ ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI**  
> Dữ liệu thực tế 2020–2025 · Mô hình Bài 1–12 · Streamlit Cloud Ready

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## 🚀 Chạy ngay trên Streamlit Cloud (không cài gì)

1. Fork repo này về tài khoản GitHub của bạn
2. Vào [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Chọn repo, branch `main`, file chính: **`Home.py`**
4. Bấm **Deploy** → ứng dụng tự động chạy!

---

## 💻 Chạy trên máy local

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/aideom-vn.git
cd aideom-vn

# 2. Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Cài thư viện
pip install -r requirements.txt

# 4. Chạy dashboard
streamlit run Home.py
```

Mở trình duyệt tại **http://localhost:8501**

---

## 📁 Cấu trúc repo

```
aideom-vn/
├── Home.py                        # Trang chủ Streamlit
├── pages/
│   ├── 1_📊_Tong_quan_vi_mo.py    # Bài 1: Cobb-Douglas & TFP
│   ├── 2_🧮_LP_Phan_bo_ngan_sach.py  # Bài 2 & 4: LP phân bổ
│   ├── 3_📦_Du_an_va_Uu_tien_nganh.py # Bài 3 & 5: MIP + Priority
│   ├── 4_🏆_TOPSIS_Xep_hang.py    # Bài 6: TOPSIS
│   ├── 5_🎯_Bien_Pareto_va_Kich_ban.py # Bài 7 & 12: Pareto + Kịch bản
│   ├── 6_⏱️_Toi_uu_dong.py        # Bài 8: Dynamic optimization
│   ├── 7_👷_Thi_truong_lao_dong.py # Bài 9: Labor market LP
│   ├── 8_🎲_Quy_hoach_ngau_nhien.py # Bài 10: Two-stage SP
│   └── 9_🤖_Q_Learning.py         # Bài 11: Q-Learning
├── utils.py                       # Dữ liệu & hàm dùng chung
├── data/
│   ├── vietnam_macro_2020_2025.csv
│   ├── vietnam_sectors_2024.csv
│   └── vietnam_regions_2024.csv
├── src/                           # Script Python gốc (12 bài)
│   ├── bai01.py … bai12_scenarios.py
├── notebooks/                     # Jupyter notebook cho Google Colab
│   ├── bai01_colab.ipynb … bai12_scenarios_colab.ipynb
├── .streamlit/config.toml         # Cấu hình giao diện
├── requirements.txt               # Thư viện cho Streamlit Cloud
└── BAO_CAO_KET_QUA.md             # Báo cáo đầy đủ 12 bài
```

---

## 📋 Tính năng Dashboard (9 trang)

| Trang | Bài | Tính năng |
|---|---|---|
| 📊 Tổng quan vĩ mô | Bài 1 | TFP, Growth Accounting, slider dự báo GDP 2030 |
| 🧮 LP Phân bổ | Bài 2 & 4 | Giải LP trực tiếp, thanh trượt ngân sách/λ, cảnh báo vô nghiệm |
| 📦 Dự án & Ưu tiên | Bài 3 & 5 | MIP 15 dự án, chỉ số ưu tiên 10 ngành có slider trọng số |
| 🏆 TOPSIS | Bài 6 | Xếp hạng 6 vùng, Entropy vs chuyên gia, độ nhạy w_AI |
| 🎯 Biên Pareto | Bài 7 & 12 | Tập Pareto ngẫu nhiên, TOPSIS compromise, so sánh 5 kịch bản |
| ⏱️ Tối ưu động | Bài 8 | Mô phỏng cơ cấu đầu tư, so sánh chiến lược, phân tích cú sốc |
| 👷 Lao động | Bài 9 | LP NetJob, biến thể xAI ngoại sinh, ngưỡng đào tạo |
| 🎲 SP ngẫu nhiên | Bài 10 | VSS/EVPI tương tác, slider xác suất kịch bản |
| 🤖 Q-Learning | Bài 11 | Huấn luyện trực tiếp trên browser, learning curve, so sánh chính sách |

---

## 📓 Google Colab Notebooks

Mỗi bài có notebook riêng trong thư mục `notebooks/` — upload lên Colab và chạy ngay:
- Tự cài thư viện (PuLP, CVXPY, pymoo, gymnasium...)
- Dữ liệu nhúng sẵn, không cần upload CSV
- Hình vẽ hiển thị inline

---

## ⚠️ Lưu ý kỹ thuật

- **Bài 4 (λ=0,70)**: VÔ NGHIỆM khi giữ trần vùng 12.000 tỷ (λ_max ≈ 0,683) — dashboard cảnh báo và giải thích nguyên nhân tự động
- **Bài 8 (φ nguyên văn)**: TFP tăng +35%/năm (phi thực tế) — app dùng φ/10 và hiển thị cả hai
- **Bài 10 (mô hình gốc)**: VSS = EVPI = 0 (suy biến) — app chỉ rõ và dùng mô hình mở rộng có ý nghĩa
- **Bài 11 (Q-Learning)**: chạy 10.000 episodes ngay trên browser, mất ~20–30 giây

---

## 📚 Tài liệu tham khảo

- Nghị quyết 57-NQ/TW (2024) · QĐ 749, 127, 411/QĐ-TTg
- NSO/GSO Việt Nam 2025 · World Bank WDI · WIPO GII 2025
- Solow (1956) · Deb et al. (2002, NSGA-II) · Birge & Louveaux (2011) · Sutton & Barto (2018)

---

## 🤖 Khai báo AI

Mã nguồn và phân tích được phát triển với sự hỗ trợ của **Claude (Anthropic)** theo đúng quy định liêm chính học thuật của đề thi (Phụ lục F2).
