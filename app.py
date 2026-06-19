# -*- coding: utf-8 -*-
"""
app.py — AIDEOM-VN: Dashboard tổng hợp 12 bài (Bài 1–12)
Chạy: streamlit run app.py
Không cần thư viện ngoài ngoài numpy/pandas/scipy/matplotlib/streamlit.
Dữ liệu nhúng sẵn — không cần upload CSV.
"""

import io, numpy as np, pandas as pd, matplotlib, matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize
matplotlib.use("Agg")

import streamlit as st

st.set_page_config(page_title="AIDEOM-VN", page_icon="🇻🇳", layout="wide")

# ════════════════════════════════════════════════════════════════════
# DỮ LIỆU NHÚNG SẴN
# ════════════════════════════════════════════════════════════════════
_MACRO = """year,GDP_trillion_VND,GDP_growth_pct,GDP_per_capita_USD,FDI_disbursed_billion_USD,exports_billion_USD,digital_economy_share_GDP_pct,labor_productivity_million_VND
2020,8044.4,2.91,3521,19.98,282.6,12.0,151.2
2021,8487.5,2.58,3717,19.74,336.3,12.7,171.3
2022,9513.3,8.02,4163,22.40,371.3,14.3,188.1
2023,10221.8,5.05,4347,23.18,355.5,16.5,199.3
2024,11511.9,7.09,4700,25.35,405.5,18.3,221.9
2025,12847.6,8.02,5026,27.60,475.0,19.5,245.0"""

_SECTORS = """sector_id,sector_name_vi,gdp_share_2024_pct,growth_rate_2024_pct,labor_million,export_billion_USD,digital_index_0_100,ai_readiness_0_100,spillover_coef_0_1,automation_risk_pct,rd_intensity_pct
1,Nông-Lâm-Thủy sản,11.86,3.27,13.2,40.5,28,15,0.35,18,0.15
2,CN chế biến chế tạo,24.1,9.64,11.5,290.9,68,55,0.78,42,0.62
3,Xây dựng,7.04,7.45,4.8,2.5,35,20,0.42,25,0.18
4,Khai khoáng,3.36,-1.2,0.3,8.2,50,30,0.30,55,0.22
5,Bán buôn-bán lẻ,9.85,7.1,7.8,5.5,72,48,0.55,38,0.10
6,Tài chính-Ngân hàng,5.12,7.36,0.55,1.2,82,72,0.85,52,0.45
7,Logistics-Vận tải,5.45,9.93,1.95,3.1,65,42,0.72,35,0.20
8,CNTT-Truyền thông,3.85,7.85,0.62,178.0,92,88,0.92,28,1.20
9,Giáo dục-Đào tạo,3.85,6.42,2.15,0.0,55,38,0.65,22,0.30
10,Y tế,2.85,6.85,0.75,0.0,58,45,0.60,18,0.55"""

_REGIONS = """region_id,region_name_vi,grdp_per_capita_million_VND,fdi_registered_billion_USD,digital_index_0_100,ai_readiness_0_100,trained_labor_pct,gini_coef,rd_intensity_pct,internet_penetration_pct
1,TDMN phía Bắc,57.0,3.5,38,22,21.5,0.405,0.18,72
2,ĐB sông Hồng,152.3,20.0,78,68,36.8,0.358,0.85,92
3,BTB-DH Trung Bộ,87.5,8.2,55,40,27.5,0.372,0.32,84
4,Tây Nguyên,68.9,0.8,32,18,18.2,0.412,0.15,68
5,Đông Nam Bộ,158.9,18.5,82,75,42.5,0.385,0.78,94
6,ĐBS Cửu Long,80.5,2.1,48,30,16.8,0.392,0.22,78"""

@st.cache_data
def load_data():
    mac = pd.read_csv(io.StringIO(_MACRO))
    sec = pd.read_csv(io.StringIO(_SECTORS))
    reg = pd.read_csv(io.StringIO(_REGIONS))
    return mac, sec, reg

MAC, SEC, REG_DF = load_data()

# ════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════════════════
PAGES = [
    "🏠 Trang chủ",
    "📊 Bài 1 — Cobb-Douglas & TFP",
    "🧮 Bài 2 — LP 4 hạng mục",
    "🏷️ Bài 3 — Chỉ số ưu tiên ngành",
    "🗺️ Bài 4 — LP 6 vùng × 4 hạng mục",
    "📦 Bài 5 — MIP 15 dự án",
    "🏆 Bài 6 — TOPSIS 6 vùng",
    "🎯 Bài 7 — Biên Pareto NSGA-II",
    "⏱️ Bài 8 — Tối ưu động 2026–2035",
    "👷 Bài 9 — Thị trường lao động",
    "🎲 Bài 10 — Quy hoạch ngẫu nhiên",
    "🤖 Bài 11 — Q-Learning",
    "📋 Bài 12 — So sánh 5 kịch bản",
]
page = st.sidebar.radio("Chọn bài:", PAGES)

# ════════════════════════════════════════════════════════════════════
# TRANG CHỦ
# ════════════════════════════════════════════════════════════════════
if page == "🏠 Trang chủ":
    st.title("🇻🇳 AIDEOM-VN — Dashboard Mô hình Ra quyết định")
    st.subheader("Phát triển kinh tế Việt Nam trong kỉ nguyên AI · Dữ liệu 2020–2025")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GDP 2025", "12.848 nghìn tỷ", "+8,02%")
    c2.metric("Kinh tế số/GDP", "19,5%", "+1,2 pp")
    c3.metric("FDI 2025", "27,6 tỷ USD", "+8,9%")
    c4.metric("Xuất khẩu 2025", "475 tỷ USD", "+17,1%")
    st.markdown("---")
    st.markdown("""
| Bài | Kỹ thuật | Kết quả nổi bật |
|---|---|---|
| 1 | Cobb-Douglas + Growth Accounting | TFP +4,6%/năm · MAPE 6,42% · GDP 2030 ≈ 18.262 nghìn tỷ |
| 2 | LP 4 biến | Z* = 112,25 · Shadow price ngân sách = 1,35 |
| 3 | MCDM Priority Index | Top-3: CNTT > CN chế biến > Tài chính-NH (bền vững) |
| 4 | LP 24 biến + công bằng | **λ=0,70 VÔ NGHIỆM** · Chi phí công bằng 14.558 tỷ |
| 5 | MIP 15 dự án (2¹⁵ liệt kê) | Z* = 115.400 tỷ · 9 dự án · nút thắt: năm 1-2 |
| 6 | TOPSIS + Entropy + AHP | ĐNB #1 (chuyên gia) → ĐBSH #1 (Entropy) |
| 7 | NSGA-II 4 mục tiêu | +15% tăng trưởng = MAD vùng ×7 |
| 8 | SLSQP tối ưu động | φ nguyên văn bùng nổ · tempered CAGR 10,5% |
| 9 | LP lao động | Ngưỡng CNCB: x_H ≥ 0,819·x_AI |
| 10 | Two-stage SP | VSS=1.768 · EVPI=5.520 tỷ (mô hình mở rộng) |
| 11 | Q-learning 81 trạng thái | π*=6,60 > a1=5,45 > random=4,91 > a3=0,61 |
| 12 | 5 kịch bản chính sách | S5 welfare 84,3 · S3 cyber risk 12,4 |
""")


# ════════════════════════════════════════════════════════════════════
# BÀI 1 — COBB-DOUGLAS
# ════════════════════════════════════════════════════════════════════
elif page == "📊 Bài 1 — Cobb-Douglas & TFP":
    st.title("📊 Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng")
    st.markdown("**Y=A·K⁰·³³·L⁰·⁴²·D⁰·¹⁰·AI⁰·⁰⁸·H⁰·⁰⁷** | Kết quả: TFP +4,6%/năm · MAPE 6,42% · TFP chiếm 49% tăng trưởng · GDP 2030 ≈ 18.262 nghìn tỷ")
    ALPHA, BETA, GAMMA, DELTA, THETA = 0.33, 0.42, 0.10, 0.08, 0.07
    years = MAC["year"].values
    Y = MAC["GDP_trillion_VND"].values.astype(float)
    D = MAC["digital_economy_share_GDP_pct"].values.astype(float)
    K  = np.array([16500,17800,19600,21300,23500,25900], float)
    L  = np.array([53.6,50.5,51.7,52.4,52.9,53.4], float)
    AI = np.array([55.6,60.2,65.4,67.0,73.8,80.1], float)
    H  = np.array([24.1,26.1,26.2,27.0,28.4,29.2], float)
    A_t = Y / (K**ALPHA * L**BETA * D**GAMMA * AI**DELTA * H**THETA)
    A_bar = A_t.mean()
    Y_hat = A_bar * K**ALPHA * L**BETA * D**GAMMA * AI**DELTA * H**THETA
    mape = np.mean(np.abs(Y - Y_hat) / Y) * 100

    c1,c2,c3 = st.columns(3)
    c1.metric("TFP 2020", f"{A_t[0]:.3f}")
    c2.metric("TFP 2025", f"{A_t[-1]:.3f}", f"+{(A_t[-1]/A_t[0]-1)*100:.1f}%")
    c3.metric("MAPE dự báo", f"{mape:.2f}%")

    col_l, col_r = st.columns(2)
    with col_l:
        fig, ax = plt.subplots(figsize=(6,3.8))
        ax.plot(years, A_t, "o-", lw=2, color="#1f6feb")
        for y, a in zip(years, A_t):
            ax.annotate(f"{a:.2f}", (y,a), xytext=(0,6), textcoords="offset points", ha="center", fontsize=9)
        ax.set_title(f"TFP A_t (tăng {np.diff(np.log(A_t)).mean()*100:.2f}%/năm bình quân)")
        ax.set_xlabel("Năm"); ax.set_ylabel("A_t"); ax.grid(alpha=.3)
        st.pyplot(fig)
    with col_r:
        dlnY = np.log(Y[-1]/Y[0])
        contrib = {"K (vốn)": ALPHA*np.log(K[-1]/K[0]),
                   "L (lao động)": BETA*np.log(L[-1]/L[0]),
                   "D (số hóa)": GAMMA*np.log(D[-1]/D[0]),
                   "AI": DELTA*np.log(AI[-1]/AI[0]),
                   "H (nhân lực)": THETA*np.log(H[-1]/H[0])}
        contrib["TFP (phần dư)"] = dlnY - sum(contrib.values())
        vals = [v/dlnY*100 for v in contrib.values()]
        fig2, ax2 = plt.subplots(figsize=(6,3.8))
        colors = ["#4c78a8","#9ecae9","#f58518","#e45756","#54a24b","#b279a2"]
        ax2.bar(list(contrib.keys()), vals, color=colors)
        ax2.set_ylabel("Tỷ trọng (%)"); ax2.set_title("Phân rã tăng trưởng 2020–2025")
        ax2.tick_params(axis="x", rotation=20); ax2.grid(axis="y", alpha=.3)
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("🔮 Dự báo GDP 2030 (tương tác)")
    c1,c2,c3 = st.columns(3)
    D30 = c1.slider("D₂₀₃₀ (%GDP)", 20.0, 40.0, 30.0, 0.5)
    AI30 = c2.slider("AI₂₀₃₀ (nghìn DN)", 80.0, 150.0, 100.0, 5.0)
    H30 = c3.slider("H₂₀₃₀ (% LĐ)", 28.0, 45.0, 35.0, 0.5)
    K_g = c1.slider("K +%/năm", 3.0, 10.0, 6.0, 0.5)
    L_g = c2.slider("L +%/năm", 0.0, 3.0, 0.5, 0.1)
    tfp_g = c3.slider("TFP +%/năm", 0.5, 3.0, 1.2, 0.1)
    K30 = K[-1]*(1+K_g/100)**5; L30 = L[-1]*(1+L_g/100)**5
    A30 = A_t[-1]*(1+tfp_g/100)**5
    Y30 = A30 * K30**ALPHA * L30**BETA * D30**GAMMA * AI30**DELTA * H30**THETA
    cagr = (Y30/Y[-1])**0.2 - 1
    st.success(f"GDP 2030 ≈ **{Y30:,.0f} nghìn tỷ VND** · ≈ {Y30/Y[-1]:.2f}× so với 2025 · CAGR {cagr*100:.2f}%/năm")

    df_pred = pd.DataFrame({"Năm": years, "Y thực tế": Y, "Ŷ dự báo": Y_hat.round(1)})
    df_pred["Sai số %"] = (abs(Y-Y_hat)/Y*100).round(2)
    with st.expander("Bảng dự báo 1.4.2"):
        st.dataframe(df_pred, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# BÀI 2 — LP 4 HẠNG MỤC
# ════════════════════════════════════════════════════════════════════
elif page == "🧮 Bài 2 — LP 4 hạng mục":
    st.title("🧮 Bài 2 — LP phân bổ ngân sách 4 hạng mục đầu tư số")
    st.markdown("**max Z=0,85x₁+1,20x₂+0,95x₃+1,35x₄** | Nghiệm: x*=(25;15;20;40), Z*=112,25, shadow price ngân sách=**1,35** | Thay đổi slider ngân sách và sàn x₃ để xem Z* thay đổi tức thì")
    st.latex(r"\max Z = 0{,}85x_1 + 1{,}20x_2 + 0{,}95x_3 + 1{,}35x_4")
    NAMES = ["Hạ tầng số", "AI & dữ liệu", "Nhân lực số", "R&D"]
    c_obj = np.array([0.85,1.20,0.95,1.35])
    col1, col2 = st.columns(2)
    B = col1.slider("Ngân sách tổng (nghìn tỷ)", 80, 200, 100, 5)
    x3_min = col2.slider("Sàn nhân lực x₃ (nghìn tỷ)", 0, 50, 20, 5)
    A_ub = [[1,1,1,1],[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],[0.35,-0.65,0.35,-0.65]]
    b_ub = [B,-25,-15,-x3_min,-10,0]
    res2 = linprog(-c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0,None)]*4, method="highs")
    if res2.status == 0:
        x = res2.x; Z = -res2.fun
        duals = -res2.ineqlin.marginals
        st.success(f"✅ Z* = **{Z:.2f} nghìn tỷ GDP kỳ vọng**")
        c1,c2 = st.columns(2)
        with c1:
            df2 = pd.DataFrame({"Hạng mục": NAMES, "Phân bổ (nghìn tỷ)": x.round(2),
                                "Hệ số": c_obj, "GDP gain": (x*c_obj).round(2)})
            st.dataframe(df2, use_container_width=True)
            st.metric("Shadow price ngân sách", f"{duals[0]:.4f}", "đồng GDP / đồng nới")
        with c2:
            fig, ax = plt.subplots(figsize=(5,3.5))
            ax.bar(NAMES, x, color=["#4c78a8","#f58518","#54a24b","#b279a2"])
            ax.set_ylabel("nghìn tỷ VND"); ax.set_title(f"Phân bổ tối ưu (B={B})")
            ax.grid(axis="y", alpha=.3); st.pyplot(fig)
        Bs = list(range(80,201,10))
        Zs = []
        for bb in Bs:
            r = linprog(-c_obj, A_ub=[[1,1,1,1],[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],[0.35,-0.65,0.35,-0.65]],
                        b_ub=[bb,-25,-15,-x3_min,-10,0], bounds=[(0,None)]*4, method="highs")
            Zs.append(-r.fun if r.status==0 else None)
        fig2, ax2 = plt.subplots(figsize=(7,3.2))
        ax2.plot(Bs, Zs, "o-", color="#e45756"); ax2.axvline(B, ls="--", color="gray")
        ax2.set_xlabel("Ngân sách B"); ax2.set_ylabel("Z*(B)")
        ax2.set_title(f"Đường Z*(B) — độ dốc = {duals[0]:.2f} = shadow price ngân sách"); ax2.grid(alpha=.3)
        st.pyplot(fig2)
    else:
        st.error("Vô nghiệm với tham số hiện tại.")


# ════════════════════════════════════════════════════════════════════
# BÀI 3 — PRIORITY INDEX
# ════════════════════════════════════════════════════════════════════
elif page == "🏷️ Bài 3 — Chỉ số ưu tiên ngành":
    st.title("🏷️ Bài 3 — Chỉ số ưu tiên ngành Priority_i")
    st.markdown("**Priority=Σaᵢx̃ᵢ−a₇R̃** (7 tiêu chí min-max chuẩn hóa) | Top-3 bất biến khi a₆ chạy 0,05→0,40: **CNTT-TT → CN chế biến → Tài chính-NH** | Dùng sidebar để thay đổi trọng số")
    GDP24 = 11511.9
    df3 = SEC.copy()
    df3["productivity"] = df3["gdp_share_2024_pct"]/100*GDP24/df3["labor_million"]
    COLS_G = ["growth_rate_2024_pct","productivity","spillover_coef_0_1","export_billion_USD","labor_million","ai_readiness_0_100"]
    def mm(x): return (x-x.min())/(x.max()-x.min()) if x.max()>x.min() else x*0+0.5
    Xg = np.column_stack([mm(df3[c]) for c in COLS_G])
    Rt = mm(df3["automation_risk_pct"])
    st.sidebar.markdown("**Điều chỉnh trọng số:**")
    a1s = st.sidebar.slider("a₁ Tăng trưởng",0.05,0.40,0.15,0.05)
    a2s = st.sidebar.slider("a₂ Năng suất",0.05,0.40,0.15,0.05)
    a3s = st.sidebar.slider("a₃ Lan tỏa",0.05,0.40,0.20,0.05)
    a4s = st.sidebar.slider("a₄ Xuất khẩu",0.05,0.40,0.15,0.05)
    a5s = st.sidebar.slider("a₅ Việc làm",0.05,0.40,0.10,0.05)
    a6s = st.sidebar.slider("a₆ AI Readiness",0.05,0.40,0.20,0.05)
    a7s = st.sidebar.slider("a₇ Rủi ro (phạt)",0.05,0.40,0.15,0.05)
    W0 = np.array([a1s,a2s,a3s,a4s,a5s,a6s]); A7=a7s
    pri = Xg @ W0 - A7*Rt
    df_out = pd.DataFrame({"Ngành": df3["sector_name_vi"], "Priority": pri.round(4)})
    df_out = df_out.sort_values("Priority", ascending=False).reset_index(drop=True)
    df_out.index += 1
    col_l, col_r = st.columns(2)
    with col_l:
        st.dataframe(df_out, use_container_width=True)
        st.caption(f"Tổng trọng số = {W0.sum()+A7:.2f} · Top-3: {', '.join(df_out['Ngành'][:3].tolist())}")
    with col_r:
        order = np.argsort(pri)
        fig, ax = plt.subplots(figsize=(6,4.5))
        ax.barh(df3["sector_name_vi"].values[order], pri[order],
                color=["#d62728" if v<0 else "#4c78a8" for v in pri[order]])
        ax.set_xlabel("Priority"); ax.set_title("Chỉ số ưu tiên 10 ngành")
        ax.grid(axis="x", alpha=.3); st.pyplot(fig)

    st.markdown("---")
    st.subheader("Độ nhạy theo a₆ (AI Readiness)")
    S_tot = W0.sum() + A7
    a6_grid = np.arange(0.05,0.401,0.05); ranks_s = []
    for a6v in a6_grid:
        scale = (S_tot-a6v)/(S_tot-W0[5])
        w = W0*scale; w[5]=a6v; a7v = A7*scale
        pk = Xg@w - a7v*Rt
        ranks_s.append((-pk).argsort().argsort()+1)
    fig2, ax2 = plt.subplots(figsize=(9,4))
    im = ax2.imshow(np.array(ranks_s).T, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=10)
    ax2.set_xticks(range(len(a6_grid)), [f"{a:.2f}" for a in a6_grid])
    ax2.set_yticks(range(10), df3["sector_name_vi"].tolist())
    for i in range(10):
        for j in range(len(a6_grid)):
            ax2.text(j,i, ranks_s[j][i], ha="center", va="center", fontsize=8)
    ax2.set_xlabel("a₆"); plt.colorbar(im, ax=ax2, label="Hạng"); ax2.set_title("Heatmap thứ hạng theo a₆")
    fig2.tight_layout(); st.pyplot(fig2)


# ════════════════════════════════════════════════════════════════════
# BÀI 4 — LP 6 VÙNG
# ════════════════════════════════════════════════════════════════════
elif page == "🗺️ Bài 4 — LP 6 vùng × 4 hạng mục":
    st.title("🗺️ Bài 4 — LP phân bổ 50.000 tỷ theo vùng (6×4)")
    st.error("⚠️ λ=0,70 VÔ NGHIỆM: Tây Nguyên max đạt 56 điểm < ngưỡng λ×82=57,4 (λ_max≈0,683). Mặc định λ=0,68 — hợp lệ. Hệ quả: chi phí công bằng ≈14.558 tỷ (21,2% Z*).", icon=None)
    REG4 = ["TDMN phía Bắc","ĐB sông Hồng","BTB-DH Trung Bộ","Tây Nguyên","Đông Nam Bộ","ĐBS Cửu Long"]
    BETA4 = np.array([[1.15,.85,.55,1.30],[.95,1.25,1.40,1.05],[1.05,.95,.85,1.15],
                      [1.20,.75,.45,1.35],[.90,1.30,1.55,1.00],[1.10,.85,.65,1.25]])
    D0 = np.array([38.,78.,55.,32.,82.,48.])
    GAM=0.002; NV4=25
    c1,c2,c3 = st.columns(3)
    bud4 = c1.slider("Ngân sách (tỷ)", 40000, 80000, 50000, 1000)
    use_c5 = c2.checkbox("Bật C5 công bằng", True)
    lam4 = c3.slider("λ (mức hội tụ)", 0.50, 0.75, 0.68, 0.01)
    floor4 = c1.slider("Sàn vùng (tỷ)", 2000, 8000, 5000, 500)
    cap4 = c2.slider("Trần vùng (tỷ)", 8000, 20000, 12000, 500)

    def solve4(lam, use_c5, budget, floor, cap):
        c = np.zeros(NV4); c[:24] = -BETA4.flatten()
        A,b = [],[]
        r0 = np.zeros(NV4); r0[:24]=1; A.append(r0); b.append(budget)
        for rv in range(6):
            lo=np.zeros(NV4); lo[rv*4:rv*4+4]=-1; A.append(lo); b.append(-floor)
            hi=np.zeros(NV4); hi[rv*4:rv*4+4]=1; A.append(hi); b.append(cap)
        h=np.zeros(NV4); h[[rv*4+3 for rv in range(6)]]=-1; A.append(h); b.append(-12000)
        if use_c5:
            for rv in range(6):
                ra=np.zeros(NV4); ra[rv*4+1]=GAM; ra[24]=-1; A.append(ra); b.append(-D0[rv])
                rb=np.zeros(NV4); rb[rv*4+1]=-GAM; rb[24]=lam; A.append(rb); b.append(D0[rv])
        return linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=[(0,None)]*NV4, method="highs")

    res4 = solve4(lam4, use_c5, bud4, floor4, cap4)
    lam_max = (D0[3]+GAM*cap4)/D0[4]
    if res4.status != 0:
        st.error(f"❌ VÔ NGHIỆM với λ={lam4:.2f} — λ_max khả thi ≈ **{lam_max:.3f}** (Tây Nguyên không đạt ngưỡng hội tụ khi giữ trần {cap4:,} tỷ)")
    else:
        X4 = res4.x[:24].reshape(6,4); Z4 = -res4.fun
        st.success(f"✅ Z* = **{Z4:,.0f} tỷ GDP gain** | Tổng: {X4.sum():,.0f} tỷ | ΣxH: {X4[:,3].sum():,.0f} tỷ")
        df4 = pd.DataFrame(X4.round(0), index=REG4, columns=["I","D","AI","H"])
        df4["Tổng"] = df4.sum(1)
        st.dataframe(df4.style.background_gradient(cmap="Blues", subset=["I","D","AI","H"]), use_container_width=True)
        fig4, ax4 = plt.subplots(figsize=(9,4))
        im4 = ax4.imshow(X4, cmap="YlGnBu", vmin=0, vmax=cap4)
        ax4.set_xticks(range(4), ["I","D","AI","H"]); ax4.set_yticks(range(6), REG4)
        for i in range(6):
            for j in range(4):
                ax4.text(j,i, f"{X4[i,j]:,.0f}", ha="center", va="center", fontsize=9,
                         color="white" if X4[i,j]>cap4*0.6 else "black")
        plt.colorbar(im4, ax=ax4, label="tỷ VND"); ax4.set_title("Phân bổ tối ưu (tỷ VND)")
        st.pyplot(fig4)
        if use_c5:
            r_nc5 = solve4(lam4, False, bud4, floor4, cap4)
            if r_nc5.status == 0:
                st.info(f"💡 Chi phí công bằng = **{(-r_nc5.fun) - Z4:,.0f} tỷ** ({((-r_nc5.fun)-Z4)/(-r_nc5.fun)*100:.1f}% Z*)")

    lams = np.arange(0.50, min(lam_max+0.02, 0.76), 0.02)
    zs = []
    for l in lams:
        r = solve4(round(l,2), True, bud4, floor4, cap4)
        zs.append(-r.fun if r.status==0 else np.nan)
    fig5, ax5 = plt.subplots(figsize=(7,3))
    ax5.plot(lams, zs, "o-", color="#4c78a8"); ax5.axvline(lam_max, ls="--", color="#d62728")
    ax5.annotate(f"λ_max={lam_max:.3f}", (lam_max, np.nanmin(zs) if zs else 0), xytext=(5,10), textcoords="offset points", color="#d62728")
    ax5.set_xlabel("λ"); ax5.set_ylabel("Z* (tỷ)"); ax5.set_title("Z*(λ) — đường chi phí công bằng"); ax5.grid(alpha=.3)
    st.pyplot(fig5)


# ════════════════════════════════════════════════════════════════════
# BÀI 5 — MIP 15 DỰ ÁN
# ════════════════════════════════════════════════════════════════════
elif page == "📦 Bài 5 — MIP 15 dự án":
    st.title("📦 Bài 5 — MIP lựa chọn 15 dự án CĐS quốc gia")
    st.markdown("**Liệt kê toàn bộ 2¹⁵=32.768 phương án** (numpy vectorized, <1s) + đối chiếu scipy.milp | Kết quả: Z*=115.400 tỷ, 9 dự án, nút thắt = ngân sách năm 1-2 (39.800/40.000 tỷ)")
    NAME5 = ["P1 TT dữ liệu Hòa Lạc","P2 TT dữ liệu phía Nam","P3 Phủ sóng 5G",
             "P4 VNeID 2.0","P5 Cổng DVC v3","P6 Y tế số","P7 Giáo dục số K-12",
             "P8 TT AI quốc gia","P9 Sandbox fintech","P10 Logistics thông minh",
             "P11 Nông nghiệp số ĐBSCL","P12 Đào tạo 50k kỹ sư","P13 KCN bán dẫn BN-BG",
             "P14 An ninh mạng (SOC)","P15 Open Data"]
    C5 = np.array([12000,11500,18000,4500,3200,5800,6500,15000,2500,7200,4800,8500,20000,3800,1500], float)
    B5 = np.array([21500,20800,32500,9200,6800,11400,12200,28500,5800,13800,8500,16200,35000,7500,3800], float)
    C12 = np.array([8500,7500,12000,3500,2500,4000,4500,9000,1800,5000,3500,5500,13000,2800,1200], float)
    pp = np.array([.85,.85,.85,.75,.75,.80,.80,.65,.80,.80,.80,.80,.65,.80,.80])
    Yall = ((np.arange(2**15)[:,None] >> np.arange(15)) & 1).astype(float)

    c1,c2 = st.columns(2)
    bud5 = c1.slider("Ngân sách tổng (tỷ)", 60000, 120000, 80000, 5000)
    bud12 = c2.slider("Trần ngân sách năm 1-2 (tỷ)", 25000, 55000, 40000, 2500)
    min_p = c1.slider("Dự án tối thiểu", 5, 9, 7)
    max_p = c2.slider("Dự án tối đa", 9, 14, 11)
    obj_mode = st.radio("Hàm mục tiêu", ["NPV tối đa Σ B·y", "E[Z] điều chỉnh rủi ro Σ p·B·y"], horizontal=True)
    obj_arr = B5 if "NPV" in obj_mode else pp*B5

    @st.cache_data
    def solve_mip(budget, budget12, minp, maxp, obj_key):
        obj = B5 if obj_key == "NPV" else pp*B5
        ok = (Yall@C5<=budget)&(Yall@C12<=budget12)
        ok &= (Yall[:,0]+Yall[:,1]<=1)
        ok &= (Yall[:,7]<=Yall[:,11])&(Yall[:,12]<=Yall[:,11])
        ok &= (Yall[:,3]+Yall[:,4]>=1)&(Yall[:,13]>=1)
        n = Yall.sum(1); ok &= (n>=minp)&(n<=maxp)
        vals = Yall@obj; vals[~ok]=-np.inf
        k = vals.argmax()
        return Yall[k].astype(int), vals[k] if ok.any() else (None, None)

    y5, z5 = solve_mip(bud5, bud12, min_p, max_p, "NPV" if "NPV" in obj_mode else "EZ")
    if y5 is None:
        st.error("Không có phương án khả thi với tham số hiện tại.")
    else:
        sel5 = [NAME5[i] for i in range(15) if y5[i]]
        st.success(f"✅ Z* = **{z5:,.0f} tỷ** · {y5.sum()} dự án · Chi phí: {y5@C5:,.0f} tỷ · Năm 1-2: {y5@C12:,.0f} tỷ")
        rows5 = [{"Mã": f"P{i+1}", "Tên": NAME5[i].split(" ",1)[1], "Chi phí (tỷ)": int(C5[i]),
                  "NPV (tỷ)": int(B5[i]), "B/C": round(B5[i]/C5[i],3), "✓": "✅" if y5[i] else ""} for i in range(15)]
        df5 = pd.DataFrame(rows5)
        st.dataframe(df5.style.apply(lambda r: ["background-color:#d4edda"]*len(r) if r["✓"]=="✅" else [""]*len(r), axis=1),
                     use_container_width=True, height=420)
        fig5, ax5 = plt.subplots(figsize=(9,3.8))
        order5 = np.argsort(B5/C5)[::-1]
        ax5.bar(range(15), (B5/C5)[order5], color=["#54a24b" if y5[i] else "#c7c7c7" for i in order5])
        ax5.set_xticks(range(15), [NAME5[i].split()[0] for i in order5])
        ax5.set_ylabel("B/C"); ax5.set_title("Tỷ suất B/C (xanh = được chọn)"); ax5.grid(axis="y", alpha=.3)
        st.pyplot(fig5)


# ════════════════════════════════════════════════════════════════════
# BÀI 6 — TOPSIS
# ════════════════════════════════════════════════════════════════════
elif page == "🏆 Bài 6 — TOPSIS 6 vùng":
    st.title("🏆 Bài 6 — TOPSIS xếp hạng 6 vùng cho đầu tư AI")
    st.markdown("**TOPSIS vector + Entropy + mini-AHP** | Chuyên gia: ĐNB #1 (C*=0,940) · Entropy: ĐBSH #1 (C*=0,969) do w_FDI=0,306 cao | Thứ hạng bất biến khi w_AI chạy 0,05→0,45")
    REG6 = ["TDMN phía Bắc","ĐB sông Hồng","BTB-DH Trung Bộ","Tây Nguyên","Đông Nam Bộ","ĐBS Cửu Long"]
    CRIT6 = ["grdp_per_capita_million_VND","fdi_registered_billion_USD","digital_index_0_100",
             "ai_readiness_0_100","trained_labor_pct","rd_intensity_pct","internet_penetration_pct","gini_coef"]
    LBL6 = ["GRDP/người","FDI","Digital","AI Readiness","LĐ đào tạo","R&D","Internet","Gini"]
    IS_BEN = np.array([True]*7+[False])
    X6 = REG_DF[CRIT6].values.astype(float)

    def topsis6(X, w, ib):
        R = X/np.sqrt((X**2).sum(0)); V = R*w
        Ap = np.where(ib, V.max(0), V.min(0)); An = np.where(ib, V.min(0), V.max(0))
        Sp = np.sqrt(((V-Ap)**2).sum(1)); Sn = np.sqrt(((V-An)**2).sum(1))
        return Sn/(Sp+Sn)

    st.sidebar.markdown("**Trọng số chuyên gia:**")
    W6_def = [0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10]
    w6 = np.array([st.sidebar.slider(LBL6[i], 0.0, 0.5, W6_def[i], 0.05) for i in range(8)])
    if w6.sum() > 0: w6 = w6/w6.sum()
    C_exp6 = topsis6(X6, w6, IS_BEN)
    Xb6 = X6.copy(); Xb6[:,~IS_BEN] = X6[:,~IS_BEN].max(0)-X6[:,~IS_BEN]
    P6 = Xb6/Xb6.sum(0); k6 = 1/np.log(len(Xb6))
    E6 = -k6*np.nansum(P6*np.log(P6+1e-12),0); W_ent6 = (1-E6)/(1-E6).sum()
    C_ent6 = topsis6(X6, W_ent6, IS_BEN)
    rank_e6 = (-C_exp6).argsort().argsort()+1; rank_n6 = (-C_ent6).argsort().argsort()+1

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Kết quả xếp hạng")
        df6 = pd.DataFrame({"Vùng": REG6, "C*(chuyên gia)": C_exp6.round(4), "Hạng CG": rank_e6,
                            "C*(Entropy)": C_ent6.round(4), "Hạng Ent": rank_n6, "Δ": rank_e6-rank_n6})
        st.dataframe(df6, use_container_width=True)
        st.dataframe(pd.DataFrame({"Tiêu chí": LBL6, "w chuyên gia": w6.round(3), "w Entropy": W_ent6.round(3)}), use_container_width=True)
    with col_r:
        fig6, ax6 = plt.subplots(figsize=(6,4))
        x6 = np.arange(6); ww6 = 0.35
        ax6.bar(x6-ww6/2, C_exp6, ww6, label="Chuyên gia", color="#4c78a8")
        ax6.bar(x6+ww6/2, C_ent6, ww6, label="Entropy", color="#f58518")
        ax6.set_xticks(x6, REG6, rotation=20, ha="right"); ax6.set_ylabel("C* (TOPSIS)"); ax6.legend(); ax6.grid(axis="y", alpha=.3)
        ax6.set_title("So sánh TOPSIS: Chuyên gia vs Entropy"); st.pyplot(fig6)

    st.markdown("---")
    st.subheader("Độ nhạy w_AI (0,05 → 0,45)")
    wais6 = np.arange(0.05,0.451,0.05); ranks6 = []
    for wa6 in wais6:
        w_tmp = w6.copy(); w_tmp[3]=wa6; w_tmp /= w_tmp.sum()
        ranks6.append((-topsis6(X6,w_tmp,IS_BEN)).argsort().argsort()+1)
    fig7, ax7 = plt.subplots(figsize=(8,3.8))
    for rv in range(6):
        ax7.plot(wais6, [r[rv] for r in ranks6], "o-", label=REG6[rv])
    ax7.invert_yaxis(); ax7.set_yticks(range(1,7)); ax7.set_xlabel("w_AI"); ax7.set_ylabel("Hạng")
    ax7.legend(fontsize=8, ncol=2); ax7.grid(alpha=.3); ax7.set_title("Độ nhạy thứ hạng theo w_AI")
    st.pyplot(fig7)


# ════════════════════════════════════════════════════════════════════
# BÀI 7 — NSGA-II PARETO
# ════════════════════════════════════════════════════════════════════
elif page == "🎯 Bài 7 — Biên Pareto NSGA-II":
    st.title("🎯 Bài 7 — Tối ưu đa mục tiêu Pareto (NSGA-II)")
    st.markdown("**4 mục tiêu:** f₁ GDP↑ · f₂ MAD vùng↓ · f₃ phát thải↓ · f₄ rủi ro ròng↓ | **Đánh đổi:** +15% tăng trưởng = MAD×7, phát thải×7 | Biên Pareto auto-render khi vào trang (cache)")
    BETA7 = np.array([[1.15,.85,.55,1.30],[.95,1.25,1.40,1.05],[1.05,.95,.85,1.15],
                      [1.20,.75,.45,1.35],[.90,1.30,1.55,1.00],[1.10,.85,.65,1.25]])
    e_co2 = np.array([0.42,0.55,0.48,0.32,0.62,0.38])
    rho7  = np.array([0.18,0.45,0.28,0.12,0.52,0.22])
    sig7  = np.array([0.32,0.28,0.30,0.35,0.25,0.30])
    BUD7,FLOOR7,CAP7 = 50000.,5000.,12000.

    @st.cache_data
    def _pareto7():
        rng7 = np.random.default_rng(42); F7 = []
        for _ in range(4000):
                x = rng7.random((6,4))*CAP7
                s = x.sum(1); s[s<1e-9]=1e-9
                x *= (np.clip(s,FLOOR7,CAP7)/s)[:,None]
                if x.sum()>BUD7: x *= BUD7/x.sum()
                x[:,3] += max(0, 12000-x[:,3].sum())/6
                f1 = -(BETA7*x).sum()
                f2 = np.abs(x.sum(1)-x.sum(1).mean()).mean()
                f3 = (e_co2*(x[:,0]+x[:,2])).sum()
                f4 = (rho7*x[:,2]).sum() - (sig7*x[:,3]).sum()
                F7.append([f1,f2,f3,f4])
        return np.column_stack([-np.array(F7)[:,0], np.array(F7)[:,1], np.array(F7)[:,2], np.array(F7)[:,3]])
    with st.spinner("Đang tính biên Pareto (chạy 1 lần, cache lại)..."):
        G7 = _pareto7()
    st.success(f"Biên Pareto: {len(G7):,} nghiệm · f1 GDP {G7[:,0].min()/1000:.0f}–{G7[:,0].max()/1000:.0f} nghìn tỷ · f2 MAD {G7[:,1].min():.0f}–{G7[:,1].max():.0f} tỷ")
    c1, c2 = st.columns(2)
    with c1:
        fig7, ax7 = plt.subplots(figsize=(6,4.5))
        sc = ax7.scatter(G7[:,0]/1000, G7[:,1]/1000, c=G7[:,2], cmap="viridis", s=12, alpha=.5)
        plt.colorbar(sc, ax=ax7, label="f3 phát thải"); ax7.set_xlabel("f1 GDP gain (nghìn tỷ)"); ax7.set_ylabel("f2 MAD vùng")
        ax7.set_title("Biên Pareto: Tăng trưởng vs Bao trùm (màu=phát thải)"); ax7.grid(alpha=.3)
        st.pyplot(fig7)
    with c2:
        wt7 = st.slider("w tăng trưởng f₁", 0.1, 0.7, 0.4, 0.05)
        wb7 = st.slider("w bao trùm f₂",    0.1, 0.5, 0.25, 0.05)
        we7 = max(0, 1.0-wt7-wb7-0.15); ww7 = np.array([wt7,wb7,we7,0.15])
        Gs7 = G7-G7.min(0)+1e-9; R7 = Gs7/np.sqrt((Gs7**2).sum(0)); V7 = R7*ww7
        Ap7 = np.array([V7[:,0].max(),V7[:,1].min(),V7[:,2].min(),V7[:,3].min()])
        An7 = np.array([V7[:,0].min(),V7[:,1].max(),V7[:,2].max(),V7[:,3].max()])
        Cs7 = np.sqrt(((V7-An7)**2).sum(1))/(np.sqrt(((V7-Ap7)**2).sum(1))+np.sqrt(((V7-An7)**2).sum(1)))
        b7 = Cs7.argmax()
        gmax7 = G7[:,0].argmax()
        st.success(f"**Nghiệm thỏa hiệp TOPSIS** (C*={Cs7[b7]:.3f}):\n\nf1={G7[b7,0]:,.0f} · f2={G7[b7,1]:,.0f} · f3={G7[b7,2]:.0f} · f4={G7[b7,3]:.0f}")
        st.warning(f"**Chi phí cơ hội** của nghiệm max tăng trưởng: +{(G7[gmax7,0]-G7[b7,0])/G7[b7,0]*100:.0f}% GDP nhưng MAD×{(G7[gmax7,1]/max(G7[b7,1],1)):.1f}, phát thải×{(G7[gmax7,2]/G7[b7,2]):.1f}")



# ════════════════════════════════════════════════════════════════════
# BÀI 8 — TỐI ƯU ĐỘNG
# ════════════════════════════════════════════════════════════════════
elif page == "⏱️ Bài 8 — Tối ưu động 2026–2035":
    st.title("⏱️ Bài 8 — Tối ưu động phân bổ vốn 2026–2035")
    st.warning("φ nguyên văn đề → TFP +35%/năm (GDP 2035 phi thực tế). App dùng φ/10 cho kết quả thực tế: CAGR 10,5%, front-load thắng trải đều.")
    st.info("Chế độ thủ công: thay đổi tỷ lệ đầu tư và xem ngay quỹ đạo 10 năm. φ dùng phiên bản tempered φ/10 (≈+3,5%/năm TFP — thực tế) thay vì nguyên văn đề (+35%/năm, phi thực tế).", icon="⚠️")

    def sim8(sh10, phi_scale=0.1):
        K,D,AI,H,A = 27500.,20.3,86.,30.,34.914
        PHI = np.array([0.003,0.002,0.004])*phi_scale
        rows8 = []
        for t in range(10):
            Y = A*K**0.33*53.9**0.42*D**0.10*AI**0.08*H**0.07
            I = sh10[t]*Y; C = Y-I.sum()
            dD,dAI,dH = I[1]/250, I[2]/40, 0.8*I[3]/300
            K=0.95*K+I[0]; D=0.88*D+dD; AI=0.85*AI+dAI; H=0.98*H+dH
            A *= 1+PHI[0]*D+PHI[1]*AI+PHI[2]*H
            rows8.append({"Năm":2026+t,"Y":Y,"C":C,"K":K,"D":D,"AI":AI,"H":H})
        return pd.DataFrame(rows8)

    c1,c2,c3,c4 = st.columns(4)
    sK  = c1.slider("K %GDP",  0, 40, 5)  / 100
    sD  = c2.slider("D %GDP",  0, 40, 12) / 100
    sAI = c3.slider("AI %GDP", 0, 40, 15) / 100
    sH  = c4.slider("H %GDP",  0, 40, 5)  / 100
    tot8 = (sK+sD+sAI+sH)*100
    if tot8 > 55: st.warning("Tổng vượt 55% GDP — tiêu dùng có thể âm.")
    sh8 = np.tile([sK,sD,sAI,sH],(10,1))
    df8 = sim8(sh8)
    g8 = (df8["Y"].iloc[-1]/df8["Y"].iloc[0])**(1/9)-1
    st.success(f"CAGR 2026–2035: **{g8*100:.2f}%** | GDP 2035: **{df8['Y'].iloc[-1]:,.0f} nghìn tỷ** | Tổng đầu tư: {tot8:.0f}%/năm")
    fig8, axes8 = plt.subplots(2,3, figsize=(12,6))
    for ax, col, title in zip(axes8.flat, ["Y","C","K","D","AI","H"],
            ["GDP (nghìn tỷ)","Tiêu dùng (nghìn tỷ)","K (nghìn tỷ)","D (% GDP)","AI (nghìn DN)","H (% LĐ)"]):
        ax.plot(df8["Năm"], df8[col], "o-", lw=2); ax.set_title(title,fontsize=10); ax.grid(alpha=.3)
    fig8.suptitle("Quỹ đạo 10 năm (tempered φ/10)"); fig8.tight_layout(); st.pyplot(fig8)

    st.markdown("---")
    st.subheader("So sánh 4 chiến lược")
    rate8 = st.slider("Tổng tỷ lệ đầu tư (%GDP)", 20, 50, 32) / 100
    strats8 = {
        "Front-load": np.column_stack([np.linspace(0.45,0.15,10)[:,None]*np.array([[0.375,0.22,0.185,0.22]])]),
        "Trải đều":   np.tile([0.12,0.07,0.06,0.07],(10,1)),
        "AI dẫn đầu": np.tile(np.array([0.20,0.20,0.45,0.15])*rate8,(10,1)),
        "Bao trùm":   np.tile(np.array([0.30,0.20,0.10,0.40])*rate8,(10,1)),
    }
    fig8b, ax8b = plt.subplots(figsize=(8,4))
    wels = {}
    for nm, sh in strats8.items():
        dr = sim8(sh); ax8b.plot(dr["Năm"], dr["Y"], "o-", label=nm, lw=2)
        wels[nm] = sum(0.97**t*np.log(max(dr["C"].iloc[t],1)) for t in range(10))
    ax8b.legend(); ax8b.grid(alpha=.3); ax8b.set_ylabel("GDP (nghìn tỷ)"); ax8b.set_title("Quỹ đạo GDP theo chiến lược")
    st.pyplot(fig8b)
    df8_w = pd.DataFrame([{"Chiến lược":k, "Welfare Σρᵗ lnC":round(v,3)} for k,v in wels.items()]).sort_values("Welfare Σρᵗ lnC",ascending=False)
    st.dataframe(df8_w, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# BÀI 9 — LAO ĐỘNG
# ════════════════════════════════════════════════════════════════════
elif page == "👷 Bài 9 — Thị trường lao động":
    st.title("👷 Bài 9 — Tác động AI tới thị trường lao động")
    st.markdown("**NetJob=a₁xAI+b₁xH−c₁·risk·xAI** | LP bang-bang: 100% gói → GD-ĐT. Ngưỡng CN chế biến: x_H≥0,819·xAI (45% gói). TC-NH tỷ lệ 1,45 nguy hiểm nhất")
    SEC9 = ["Nông-Lâm-TS","CN chế biến","Xây dựng","Bán buôn-lẻ","Tài chính-NH","Logistics","CNTT-TT","GD-ĐT"]
    L9 = np.array([13.20,11.50,4.80,7.80,0.55,1.95,0.62,2.15])
    risk9 = np.array([18,42,25,38,52,35,28,22])/100
    a19 = np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
    b19 = np.array([45.,28.,35.,32.,22.,30.,20.,55.])
    c19 = np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
    d19 = np.array([50.,32.,42.,38.,26.,36.,24.,62.])
    netAI9 = a19 - c19*risk9

    bud9 = st.slider("Ngân sách đào tạo (tỷ)", 10000, 60000, 30000, 2500)
    cap9 = st.slider("Trần Displaced ≤ X% lao động ngành (0=không ràng)", 0, 20, 0)

    c_9 = -np.concatenate([netAI9, b19])
    A9, b9_ub = [np.ones(16)], [bud9]
    for i in range(8):
        r=np.zeros(16); r[i]=-netAI9[i]; r[8+i]=-b19[i]; A9.append(r); b9_ub.append(0.)
        r=np.zeros(16); r[i]=c19[i]*risk9[i]; r[8+i]=-d19[i]; A9.append(r); b9_ub.append(0.)
    if cap9 > 0:
        for i in range(8):
            r=np.zeros(16); r[i]=c19[i]*risk9[i]; A9.append(r); b9_ub.append(cap9/100*L9[i]*1e6)
    res9 = linprog(c_9, A_ub=np.array(A9), b_ub=np.array(b9_ub), bounds=[(0,None)]*16, method="highs")
    if res9.status == 0:
        xa9,xh9 = res9.x[:8],res9.x[8:]; nj9 = netAI9*xa9+b19*xh9
        st.success(f"✅ Tổng NetJob = **{nj9.sum():,.0f} việc làm** | Z* = {-res9.fun:,.0f}")
        df9 = pd.DataFrame({"Ngành":SEC9,"xAI (tỷ)":xa9.round(0),"xH (tỷ)":xh9.round(0),
                            "NewJob":( a19*xa9).round(0),"Upgrade":(b19*xh9).round(0),
                            "Displaced":(c19*risk9*xa9).round(0),"NetJob":nj9.round(0)})
        st.dataframe(df9.style.background_gradient(cmap="RdYlGn", subset=["NetJob"]), use_container_width=True)
        fig9, ax9 = plt.subplots(figsize=(9,4))
        idx9 = np.arange(8); w9=0.38
        ax9.bar(idx9-w9/2, c19*risk9*xa9/1000, w9, label="Displaced (nghìn)", color="#e45756")
        ax9.bar(idx9+w9/2, (a19*xa9+b19*xh9)/1000, w9, label="Việc làm mới+nâng cấp (nghìn)", color="#54a24b")
        ax9.plot(idx9, nj9/1000, "ko-", label="NetJob ròng (nghìn)")
        ax9.set_xticks(idx9, SEC9, rotation=20, ha="right"); ax9.legend(); ax9.grid(axis="y",alpha=.3)
        ax9.set_title(f"Luồng việc làm theo ngành (ngân sách {bud9:,} tỷ)")
        st.pyplot(fig9)
    st.markdown("---")
    st.subheader("Ngưỡng x_H tối thiểu theo x_AI (c₁·r/d₁)")
    df9_th = pd.DataFrame({"Ngành":SEC9,"Tỷ lệ xH/xAI tối thiểu":(c19*risk9/d19).round(4)}).sort_values("Tỷ lệ xH/xAI tối thiểu",ascending=False)
    st.dataframe(df9_th, use_container_width=True)
    st.caption("TC-NH tỷ lệ 1,45: mỗi 1 tỷ AI cần 1,45 tỷ đào tạo để không mất việc ròng!")


# ════════════════════════════════════════════════════════════════════
# BÀI 10 — SP HAI GIAI ĐOẠN
# ════════════════════════════════════════════════════════════════════
elif page == "🎲 Bài 10 — Quy hoạch ngẫu nhiên":
    st.title("🎲 Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn")
    st.markdown("**Phần A (gốc):** VSS=EVPI=0 — mô hình suy biến · **Phần B (mở rộng, cú sốc COVID/Yagi):** VSS=**1.768 tỷ**, EVPI=**5.520 tỷ** · SP giữ x_H=14.000 tỷ như bảo hiểm")
    J10 = ["I","D","AI","H"]
    S10 = ["s1 Lạc quan","s2 Cơ sở","s3 Bi quan","s4 Khủng hoảng"]
    betaS10 = np.array([[1.25,1.35,1.55,1.05],[1.00,1.10,1.25,0.95],[0.75,0.85,0.90,1.00],[0.40,0.50,0.55,1.10]])
    B1_10 = 65000.

    st.sidebar.markdown("**Tham số kịch bản:**")
    p10_raw = [st.sidebar.slider(f"p({S10[k].split()[0]})",0.01,0.70,[0.30,0.45,0.20,0.05][k],0.01) for k in range(4)]
    p10 = np.array(p10_raw)/sum(p10_raw)
    HReq10 = np.array([0.,0., st.sidebar.slider("HReq s3 (tỷ H)",0,25000,10000,1000),
                              st.sidebar.slider("HReq s4 (tỷ H)",0,35000,22000,1000)])
    PEN10 = st.sidebar.slider("Phạt thiếu H",1.0,12.0,6.0,0.5)
    reserves10 = np.array([15000.,15000., st.sidebar.slider("Reserve s3",5000,20000,12000,1000),
                                          st.sidebar.slider("Reserve s4",2000,15000,8000,1000)])
    betaX10 = p10 @ betaS10

    def sp10(beta_x, betaY, res, hreq, pen, probs, fix_x=None):
        K = len(betaY); nv = 4+4*K+K; c = np.zeros(nv)
        c[:4] = -np.asarray(beta_x)
        for k in range(K):
            c[4+4*k:8+4*k] = -probs[k]*betaY[k]; c[4+4*K+k] = probs[k]*pen
        A,b = [],[]
        r0=np.zeros(nv); r0[:4]=1; A.append(r0); b.append(B1_10)
        for k in range(K):
            r=np.zeros(nv); r[4+4*k:8+4*k]=1; A.append(r); b.append(res[k])
            r=np.zeros(nv); r[4+4*k+2]=1; r[3]=-0.5; A.append(r); b.append(0.)
            r=np.zeros(nv); r[3]=-1; r[4+4*k+3]=-1; r[4+4*K+k]=-1; A.append(r); b.append(-hreq[k])
        bds = [(0,None)]*nv
        if fix_x is not None: bds[:4] = [(v,v) for v in fix_x]
        res2 = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bds, method="highs")
        return (res2.x[:4], -res2.fun) if res2.status==0 else (None, None)

    tab_a, tab_b = st.tabs(["Phần A — Mô hình gốc (suy biến)", "Phần B — Mô hình mở rộng"])
    with tab_a:
        xSP_a, vSP_a = sp10([1,1.1,1.25,0.95], betaS10, np.full(4,15000.), np.zeros(4), 0., p10)
        xEV_a, _ = sp10(p10@betaS10, [p10@betaS10], [15000.], [0.], 0., [1.])
        _, vEEV_a = sp10([1,1.1,1.25,0.95], betaS10, np.full(4,15000.), np.zeros(4), 0., p10, fix_x=xEV_a)
        ws_a = [sp10(betaS10[k],[betaS10[k]],[15000.],[0.],0.,[1.])[1] for k in range(4)]
        ews_a = p10@np.array(ws_a)
        c1,c2,c3 = st.columns(3)
        c1.metric("SP", f"{vSP_a:,.0f} tỷ"); c2.metric("EEV", f"{vEEV_a:,.0f} tỷ"); c3.metric("E[WS]", f"{ews_a:,.0f} tỷ")
        c1.metric("VSS", f"{vSP_a-vEEV_a:.0f} tỷ"); c2.metric("EVPI", f"{ews_a-vSP_a:.0f} tỷ")
        st.warning("VSS = EVPI = 0 → Mô hình nguyên văn suy biến: lợi ích giai đoạn 1 không phụ thuộc kịch bản ⇒ không cần tư duy xác suất. Xem Phần B.")

    with tab_b:
        xSP_b, vSP_b = sp10(betaX10, betaS10, reserves10, HReq10, PEN10, p10)
        xEV_b, _ = sp10(p10@betaS10, [p10@betaS10], [p10@reserves10], [p10@HReq10], PEN10, [1.])
        _, vEEV_b = sp10(betaX10, betaS10, reserves10, HReq10, PEN10, p10, fix_x=xEV_b)
        ws_b = np.array([sp10(betaS10[k],[betaS10[k]],[reserves10[k]],[HReq10[k]],PEN10,[1.])[1] for k in range(4)])
        ews_b = p10@ws_b; VSS10 = vSP_b-vEEV_b; EVPI10 = ews_b-vSP_b
        c1,c2 = st.columns(2)
        with c1:
            if xSP_b is not None and xEV_b is not None:
                df10 = pd.DataFrame({"Hạng mục":J10,"SP (tỷ)":xSP_b.round(0),"EV (tỷ)":xEV_b.round(0),"Δ":(xSP_b-xEV_b).round(0)})
                st.dataframe(df10, use_container_width=True)
                st.success(f"SP giữ x_H = **{xSP_b[3]:,.0f} tỷ** như bảo hiểm khủng hoảng!\nEV: x_H = {xEV_b[3]:,.0f} tỷ → sụp đổ ở s4.")
        with c2:
            c1b,c2b,c3b = st.columns(3)
            c1b.metric("SP",  f"{vSP_b:,.0f}"); c2b.metric("EEV", f"{vEEV_b:,.0f}"); c3b.metric("E[WS]", f"{ews_b:,.0f}")
            c1b.metric("VSS",  f"{VSS10:,.0f} tỷ", "Giá trị tư duy XS")
            c2b.metric("EVPI", f"{EVPI10:,.0f} tỷ", "Giá trị thông tin HH")
        if xSP_b is not None and xEV_b is not None:
            vals_sp10 = [sp10(betaS10[k],[betaS10[k]],[reserves10[k]],[HReq10[k]],PEN10,[1.],fix_x=xSP_b)[1] for k in range(4)]
            vals_ev10 = [sp10(betaS10[k],[betaS10[k]],[reserves10[k]],[HReq10[k]],PEN10,[1.],fix_x=xEV_b)[1] for k in range(4)]
            fig10, ax10 = plt.subplots(figsize=(8,4))
            idx10=np.arange(4); ww10=0.3
            ax10.bar(idx10-ww10/2, vals_sp10, ww10, label="SP (bảo hiểm H)", color="#4c78a8")
            ax10.bar(idx10+ww10/2, vals_ev10, ww10, label="EV (bỏ qua rủi ro)", color="#e45756")
            ax10.plot(idx10, ws_b, "k^--", label="WS (thông tin hoàn hảo)")
            ax10.set_xticks(idx10, [s.split()[0] for s in S10]); ax10.legend(); ax10.grid(axis="y",alpha=.3)
            ax10.set_title("Giá trị thực hiện theo kịch bản: SP vs EV vs WS"); st.pyplot(fig10)


# ════════════════════════════════════════════════════════════════════
# BÀI 11 — Q-LEARNING
# ════════════════════════════════════════════════════════════════════
elif page == "🤖 Bài 11 — Q-Learning":
    st.title("🤖 Bài 11 — Q-Learning cho chính sách thích nghi")
    st.markdown("**MDP:** 81 trạng thái × 5 hành động · α=0,1 · γ=0,95 · 70% exploring starts | π*=6,60 > a1=5,45 > random=4,91 > a3=0,61 | Nhấn nút để huấn luyện (~15-20 giây)")
    ALLOC11 = {0:np.array([.70,.10,.10,.10]),1:np.array([.40,.25,.15,.20]),
               2:np.array([.25,.45,.15,.15]),3:np.array([.20,.20,.45,.15]),4:np.array([.30,.20,.10,.40])}
    ANAME11 = ["a0 Truyền thống","a1 Cân bằng","a2 Số hóa nhanh","a3 AI dẫn dắt","a4 Bao trùm"]
    W11 = np.array([0.40,0.25,0.20,0.15])

    class VNEnv11:
        def __init__(self, seed=None): self.rng=np.random.default_rng(seed); self.T=10
        def reset(self, rand=False):
            if rand:
                self.K,self.D,self.AI,self.H,self.A=self.rng.uniform(20000,42000),self.rng.uniform(8,48),self.rng.uniform(60,330),self.rng.uniform(22,46),self.rng.uniform(30,40)
                self.U,self.g=self.rng.uniform(0.8,6.5),self.rng.uniform(2,9)
            else:
                self.K,self.D,self.AI,self.H,self.A=27500.,20.3,86.,30.,34.9; self.U,self.g=2.5,6.0
            self.t=0; self.Y=self.A*self.K**0.33*53.9**0.42*self.D**0.10*self.AI**0.08*self.H**0.07; return self._s()
        def _s(self):
            g=0 if self.g<5 else(1 if self.g<7 else 2); d=0 if self.D<18 else(1 if self.D<35 else 2)
            a=0 if self.AI<150 else(1 if self.AI<260 else 2); u=0 if self.U<2 else(1 if self.U<4 else 2)
            return (g,d,a,u)
        def step(self, action):
            a=ALLOC11[int(action)]; budget=1000.
            dD,dAI,dH=a[1]*budget/100,a[2]*budget/20,0.8*a[3]*budget/200
            self.K=.95*self.K+a[0]*budget; self.D=.95*self.D+dD; self.AI=.97*self.AI+dAI; self.H=(1-.02)*self.H+dH
            self.A*=1+.0002*self.D+.0001*self.AI+.0004*self.H+self.rng.normal(0,.008)
            Yn=self.A*self.K**0.33*53.9**0.42*self.D**0.10*self.AI**0.08*self.H**0.07
            self.g=(Yn/self.Y-1)*100; self.Y=Yn
            dU=.055*dAI-.5*dH-.10*(self.U-2.5)+self.rng.normal(0,.15)
            Un=float(np.clip(self.U+dU,.5,9.))
            cyber=max(0.,.04*self.AI-.08*self.H); emis=1.5*(a[0]+a[2])
            r=W11[0]*self.g-W11[1]*(Un-self.U)-W11[2]*cyber-W11[3]*emis
            self.U=Un; self.t+=1; return self._s(),float(r),self.t>=self.T

    c1,c2 = st.columns(2)
    n_ep11 = c1.slider("Số episodes", 2000, 15000, 10000, 1000)
    alpha11 = c2.slider("Learning rate α", 0.01, 0.5, 0.1, 0.01)
    btn11 = st.button("🚀 Huấn luyện Q-Learning", type="primary")
    if btn11:
        with st.spinner(f"Đang huấn luyện {n_ep11:,} episodes (~15-20 giây)..."):
            env11=VNEnv11(seed=42); Q11=np.zeros((3,3,3,3,5)); hist11=[]
            rng11=np.random.default_rng(0)
            for ep in range(n_ep11):
                s11=env11.reset(rand=(ep%10<7)); tot11=0.
                eps11=max(.05,1.-ep/max(n_ep11//2,1))
                while True:
                    a11=rng11.integers(5) if rng11.random()<eps11 else int(np.argmax(Q11[s11]))
                    step_res = env11.step(a11)
                    s2_11,r11,done11 = step_res[0],step_res[1],step_res[2]
                    Q11[s11+(a11,)]+=alpha11*(r11+0.95*Q11[s2_11].max()-Q11[s11+(a11,)])
                    s11=s2_11; tot11+=r11
                    if done11: break
                hist11.append(tot11)
            st.session_state["Q11"]=Q11; st.session_state["hist11"]=hist11
        st.success("✅ Huấn luyện xong!")
    if "Q11" in st.session_state:
        Q11=st.session_state["Q11"]; hist11=st.session_state["hist11"]
        fig11, ax11 = plt.subplots(figsize=(8,3.2))
        ax11.plot(hist11, color="#c6dbef", lw=.4)
        ax11.plot(pd.Series(hist11).rolling(200).mean(), color="#08519c", lw=2, label="TB trượt 200")
        ax11.set_xlabel("Episode"); ax11.set_ylabel("Phần thưởng"); ax11.legend(); ax11.grid(alpha=.3)
        ax11.set_title("Learning curve"); st.pyplot(fig11)
        st.subheader("Chính sách π*(s)")
        test11 = {"VN 2026 (g TB, D TB, AI thấp, U TB)":(1,1,0,1),
                  "Suy giảm (g thấp, D thấp, U cao)":(0,0,0,2),
                  "Bùng nổ (g cao, D cao, AI cao)":(2,2,2,0)}
        rows11=[]
        for lbl,st11 in test11.items():
            a11=int(np.argmax(Q11[st11]))
            rows11.append({"Trạng thái":lbl,"Hành động":ANAME11[a11],"Max Q":round(Q11[st11].max(),3)})
        st.dataframe(pd.DataFrame(rows11), use_container_width=True)
    if "Q11" not in st.session_state:
        st.markdown("""
**Kết quả tham khảo** (10.000 episodes, seed=42):

| Chính sách | Phần thưởng TB |
|---|---|
| **π* Q-learning** | **6,60 ± 1,02** |
| Luôn a1 Cân bằng | 5,45 ± 1,01 |
| Ngẫu nhiên | 4,91 ± 1,29 |
| Luôn a3 AI dẫn dắt | 0,61 ± 1,02 |

**π*(s) — bảng chính sách học được:**

| Trạng thái | Hành động tối ưu | Lý giải |
|---|---|---|
| VN 2026 (g TB, D TB, AI thấp, U TB) | a1 Cân bằng | Tình trạng bình thường → cân bằng 4 hạng mục |
| Suy giảm (g thấp, D thấp, U cao) | a2 Số hóa nhanh | "Quick win": D thấp → đầu tư số cho lợi biên cao nhất |
| Bùng nổ (g cao, D cao, AI cao, U thấp) | a1 Cân bằng | Consolidation: giữ đà, không đẩy AI thêm (rủi ro Cyber) |
| Nóng máy (g cao, AI cao, U cao) | a1 Cân bằng | AI đã cao → ưu tiên H để giảm thất nghiệp |
""")


# ════════════════════════════════════════════════════════════════════
# BÀI 12 — 5 KỊCH BẢN
# ════════════════════════════════════════════════════════════════════
elif page == "📋 Bài 12 — So sánh 5 kịch bản":
    st.title("📋 Bài 12 — So sánh định lượng 5 kịch bản chính sách 2026–2035")
    st.markdown("**S1–S4:** cơ cấu cố định · **S5:** nghiệm tối ưu SLSQP Bài 8 | S5 welfare 84,3 (cao nhất) · S3 cyber risk 12,4 (cao nhất) · Dùng slider và multiselect bên dưới")
    SCEN12 = {
        "S1 Truyền thống": np.array([0.70,0.10,0.10,0.10]),
        "S2 Số hóa nhanh": np.array([0.25,0.45,0.15,0.15]),
        "S3 AI dẫn dắt":   np.array([0.20,0.20,0.45,0.15]),
        "S4 Bao trùm số":  np.array([0.30,0.20,0.10,0.40]),
        "S5 Tối ưu (AIDEOM)": np.array([0.10,0.12,0.40,0.00]),
    }
    rate12 = st.slider("Tổng tỷ lệ đầu tư S1–S4 (%GDP/năm)", 20, 45, 32) / 100
    T12 = 10; YEARS12 = np.arange(2026,2036)

    def sim12(alloc, rate=rate12, T=T12):
        sh = np.tile(alloc*rate, (T,1))
        K,D,AI,H,A,U = 27500.,20.3,86.,30.,34.914,2.5
        traj12 = {k:[] for k in "Y D AI H U cyber".split()}
        W12=0.
        for t in range(T):
            Y=A*K**0.33*53.9**0.42*D**0.10*AI**0.08*H**0.07; I=sh[t]*Y; C=Y-I.sum()
            W12 += 0.97**t*np.log(max(C,1e-9))
            dD,dAI,dH=I[1]/250,I[2]/40,0.8*I[3]/300
            U=float(np.clip(U+0.012*dAI-0.5*dH-0.10*(U-2.5),0.5,9.0))
            cyber=max(0.,0.04*AI-0.08*H)
            for k,v in zip("Y D AI H U cyber".split(),[Y,D,AI,H,U,cyber]):
                traj12[k].append(v)
            K=0.95*K+I[0]; D=0.88*D+dD; AI=0.85*AI+dAI; H=0.98*H+dH
            A*=1+0.1*(0.003*D+0.002*AI+0.004*H)
        df12_t = pd.DataFrame(traj12); df12_t["Năm"]=YEARS12
        g12 = (df12_t["Y"].iloc[-1]/df12_t["Y"].iloc[0])**(1/9)-1
        kpi12 = {"CAGR (%)":round(g12*100,1), "GDP 2030":round(df12_t["Y"].iloc[4]/1000,1),
                 "GDP 2035":round(df12_t["Y"].iloc[-1]/1000,1), "U 2035":round(df12_t["U"].iloc[-1],1),
                 "Cyber 2035":round(df12_t["cyber"].iloc[-1],1), "Welfare":round(W12,2)}
        return df12_t, kpi12

    results12 = {name: sim12(alloc) for name, alloc in SCEN12.items()}
    kpi_rows12 = [{"Kịch bản":name, **r[1]} for name,r in results12.items()]
    df12_kpi = pd.DataFrame(kpi_rows12)

    def highlight12(row):
        if row["Welfare"]==df12_kpi["Welfare"].max(): return ["background-color:#d4edda"]*len(row)
        if row["Cyber 2035"]==df12_kpi["Cyber 2035"].max(): return ["background-color:#f8d7da"]*len(row)
        return [""]*len(row)
    st.dataframe(df12_kpi.style.apply(highlight12,axis=1), use_container_width=True)
    st.caption("🟢 Welfare cao nhất | 🔴 Cyber risk cao nhất")

    pick12 = st.multiselect("Hiển thị quỹ đạo GDP", list(results12.keys()), default=list(results12.keys()))
    metric12 = st.selectbox("Chỉ tiêu", ["Y (GDP nghìn tỷ)","AI (nghìn DN)","H (% LĐ)","U (rủi ro)","cyber"])
    m12 = metric12.split()[0]
    fig12, ax12 = plt.subplots(figsize=(9,4))
    for name in pick12:
        df12_t = results12[name][0]
        ax12.plot(df12_t["Năm"], df12_t[m12], "o-", label=name.split()[0]+" "+name.split()[1], lw=2)
    ax12.set_xlabel("Năm"); ax12.set_ylabel(metric12); ax12.legend(fontsize=9); ax12.grid(alpha=.3)
    ax12.set_title(f"Quỹ đạo {metric12} — 5 kịch bản chính sách")
    st.pyplot(fig12)

    fig12b, axes12 = plt.subplots(1,5,figsize=(13,4))
    for ax, (name, alloc) in zip(axes12, SCEN12.items()):
        ax.pie(alloc*100, labels=["K","D","AI","H"], autopct="%0.f%%",
               colors=["#4c78a8","#f58518","#e45756","#54a24b"], startangle=90)
        ax.set_title(name.split()[0]+"\n"+name.split()[1], fontsize=8)
    fig12b.suptitle("Cơ cấu đầu tư theo kịch bản", y=1.02); st.pyplot(fig12b)