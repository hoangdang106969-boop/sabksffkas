# -*- coding: utf-8 -*-
"""
AIDEOM-VN — Trang chủ
Chạy: streamlit run Home.py
"""
import streamlit as st

st.set_page_config(
    page_title="AIDEOM-VN",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🇻🇳 AIDEOM-VN Dashboard")
st.subheader("Hệ thống hỗ trợ ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI")
st.caption("Dữ liệu thực tế 2020–2025 · NSO/GSO, MoST, MIC, World Bank · Mô hình: Bài 1–12")

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("GDP 2025", "12.848 nghìn tỷ VND", "+8,02%")
    st.metric("Kinh tế số/GDP", "19,5%", "+1,2 pp")
with col2:
    st.metric("FDI giải ngân 2025", "27,6 tỷ USD", "+8,9%")
    st.metric("Xuất khẩu 2025", "475 tỷ USD", "+17,1%")
with col3:
    st.metric("GDP per capita", "5.026 USD", "+6,9%")
    st.metric("Năng suất LĐ", "245 triệu VND/người", "+10,4%")

st.markdown("---")
st.markdown("### 📋 Điều hướng")

pages = [
    ("📊", "Tổng quan vĩ mô", "Phân tích GDP, cơ cấu, TFP, lao động 2020–2025 (Bài 1)"),
    ("🧮", "LP Phân bổ ngân sách", "Tối ưu phân bổ 4 hạng mục đầu tư số · 6 vùng × 4 hạng mục (Bài 2, 4)"),
    ("📦", "Lựa chọn dự án (MIP)", "Chọn danh mục tối ưu từ 15 dự án CĐS quốc gia (Bài 5)"),
    ("🏆", "Xếp hạng TOPSIS", "Xếp hạng ưu tiên 10 ngành và 6 vùng cho đầu tư AI (Bài 3, 6)"),
    ("🎯", "Biên Pareto", "Tối ưu đa mục tiêu: tăng trưởng vs bao trùm vs môi trường (Bài 7)"),
    ("⏱️", "Tối ưu động", "Chiến lược phân bổ vốn 2026–2035 · 5 kịch bản so sánh (Bài 8, 12)"),
    ("👷", "Thị trường lao động", "Tác động AI/tự động hóa · NetJob ròng · ngưỡng đào tạo (Bài 9)"),
    ("🎲", "Quy hoạch ngẫu nhiên", "Two-stage SP · VSS · EVPI · Robust minimax regret (Bài 10)"),
    ("🤖", "Q-Learning", "Chính sách thích nghi qua học tăng cường tabular (Bài 11)"),
]

for icon, name, desc in pages:
    st.markdown(f"**{icon} {name}** — {desc}")

st.markdown("---")
st.info(
    "**Khai báo AI:** Toàn bộ mô hình và mã nguồn được phát triển với sự hỗ trợ của "
    "trợ lý AI (Claude — Anthropic) theo đúng quy định liêm chính học thuật "
    "(Phụ lục F2 của đề thi).",
    icon="ℹ️"
)
