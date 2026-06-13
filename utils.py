# -*- coding: utf-8 -*-
"""utils.py — Dữ liệu dùng chung + các hàm mô hình cho toàn bộ ứng dụng."""
import io
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import streamlit as st

# ─────────────────────────────────────────────────────────────────
# 1. DỮ LIỆU NHÚNG SẴN (fallback nếu file CSV không có)
# ─────────────────────────────────────────────────────────────────
_MACRO_CSV = """year,GDP_trillion_VND,GDP_growth_pct,GDP_per_capita_USD,population_million,FDI_disbursed_billion_USD,exports_billion_USD,digital_economy_share_GDP_pct,labor_productivity_million_VND
2020,8044.4,2.91,3521,97.58,19.98,282.6,12.0,151.2
2021,8487.5,2.58,3717,98.51,19.74,336.3,12.7,171.3
2022,9513.3,8.02,4163,99.46,22.40,371.3,14.3,188.1
2023,10221.8,5.05,4347,100.30,23.18,355.5,16.5,199.3
2024,11511.9,7.09,4700,101.30,25.35,405.5,18.3,221.9
2025,12847.6,8.02,5026,102.30,27.60,475.0,19.5,245.0"""

_SECTORS_CSV = """sector_id,sector_name_vi,gdp_share_2024_pct,growth_rate_2024_pct,labor_million,export_billion_USD,digital_index_0_100,ai_readiness_0_100,spillover_coef_0_1,automation_risk_pct,rd_intensity_pct
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

_REGIONS_CSV = """region_id,region_name_vi,grdp_per_capita_million_VND,fdi_registered_billion_USD,digital_index_0_100,ai_readiness_0_100,trained_labor_pct,gini_coef,rd_intensity_pct,internet_penetration_pct
1,TDMN phía Bắc,57.0,3.5,38,22,21.5,0.405,0.18,72
2,ĐB sông Hồng,152.3,20.0,78,68,36.8,0.358,0.85,92
3,BTB-DH Trung Bộ,87.5,8.2,55,40,27.5,0.372,0.32,84
4,Tây Nguyên,68.9,0.8,32,18,18.2,0.412,0.15,68
5,Đông Nam Bộ,158.9,18.5,82,75,42.5,0.385,0.78,94
6,ĐBS Cửu Long,80.5,2.1,48,30,16.8,0.392,0.22,78"""


@st.cache_data
def load_macro():
    try:
        from pathlib import Path
        return pd.read_csv(Path(__file__).parent / "data" / "vietnam_macro_2020_2025.csv").sort_values("year").reset_index(drop=True)
    except Exception:
        return pd.read_csv(io.StringIO(_MACRO_CSV)).sort_values("year").reset_index(drop=True)


@st.cache_data
def load_sectors():
    try:
        from pathlib import Path
        df = pd.read_csv(Path(__file__).parent / "data" / "vietnam_sectors_2024.csv")
        df = df.rename(columns={"sector_name_en": "sector_name_vi"})
        return df
    except Exception:
        return pd.read_csv(io.StringIO(_SECTORS_CSV))


@st.cache_data
def load_regions():
    try:
        from pathlib import Path
        df = pd.read_csv(Path(__file__).parent / "data" / "vietnam_regions_2024.csv")
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return pd.read_csv(io.StringIO(_REGIONS_CSV))


# ─────────────────────────────────────────────────────────────────
# 2. HẰNG SỐ & THAM SỐ MÔ HÌNH
# ─────────────────────────────────────────────────────────────────
ALPHA, BETA, GAMMA, DELTA, THETA = 0.33, 0.42, 0.10, 0.08, 0.07
L0 = 53.9  # triệu lao động

REGIONS_VI = ["TDMN phía Bắc", "ĐB sông Hồng", "BTB-DH Trung Bộ",
              "Tây Nguyên", "Đông Nam Bộ", "ĐBS Cửu Long"]
ITEMS = ["I (Hạ tầng số)", "D (CĐS doanh nghiệp)", "AI", "H (Nhân lực số)"]
ITEMS_SHORT = ["I", "D", "AI", "H"]

BETA_MATRIX = np.array([
    [1.15, 0.85, 0.55, 1.30],
    [0.95, 1.25, 1.40, 1.05],
    [1.05, 0.95, 0.85, 1.15],
    [1.20, 0.75, 0.45, 1.35],
    [0.90, 1.30, 1.55, 1.00],
    [1.10, 0.85, 0.65, 1.25],
])
D0_REGIONS = np.array([38., 78., 55., 32., 82., 48.])

PROJECTS = {
    "P1": ("TT dữ liệu Hòa Lạc",  12000, 21500, 8500),
    "P2": ("TT dữ liệu phía Nam",  11500, 20800, 7500),
    "P3": ("5G phủ sóng toàn quốc",18000, 32500,12000),
    "P4": ("VNeID 2.0",             4500,  9200,  3500),
    "P5": ("Cổng DVC v3",           3200,  6800,  2500),
    "P6": ("Y tế số",               5800, 11400,  4000),
    "P7": ("Giáo dục số K-12",      6500, 12200,  4500),
    "P8": ("TT AI quốc gia",       15000, 28500,  9000),
    "P9": ("Sandbox fintech",       2500,  5800,  1800),
    "P10":("Logistics thông minh",  7200, 13800,  5000),
    "P11":("Nông nghiệp số ĐBSCL",  4800,  8500,  3500),
    "P12":("Đào tạo 50k kỹ sư AI",  8500, 16200,  5500),
    "P13":("KCN bán dẫn BN-BG",    20000, 35000, 13000),
    "P14":("An ninh mạng SOC",      3800,  7500,  2800),
    "P15":("Open Data quốc gia",    1500,  3800,  1200),
}

# ─────────────────────────────────────────────────────────────────
# 3. HÀM MÔ HÌNH DÙNG CHUNG
# ─────────────────────────────────────────────────────────────────

def cobb_douglas(K, L, D, AI, H, A=1.0):
    return A * K**ALPHA * L**BETA * D**GAMMA * AI**DELTA * H**THETA


def solve_lp_region(budget=50000., lam=0.68, use_c5=True, floor=5000., cap=12000., h_floor=12000.):
    """LP Bài 4: 6 vùng × 4 hạng mục + biến M (trả về dict kết quả)."""
    GAM = 0.002
    nv = 25  # 24 biến x + 1 biến M
    c = np.zeros(nv); c[:24] = -BETA_MATRIX.flatten()
    A_ub, b_ub = [], []
    # C1 tổng ngân sách
    r = np.zeros(nv); r[:24] = 1; A_ub.append(r); b_ub.append(budget)
    # C2 sàn vùng; C3 trần vùng
    for rv in range(6):
        lo = np.zeros(nv); lo[rv*4:rv*4+4] = -1; A_ub.append(lo); b_ub.append(-floor)
        hi = np.zeros(nv); hi[rv*4:rv*4+4] = 1; A_ub.append(hi); b_ub.append(cap)
    # C4 sàn H
    h = np.zeros(nv); h[[rv*4+3 for rv in range(6)]] = -1; A_ub.append(h); b_ub.append(-h_floor)
    # C5 công bằng
    if use_c5:
        for rv in range(6):
            ra = np.zeros(nv); ra[rv*4+1] = GAM; ra[24] = -1
            A_ub.append(ra); b_ub.append(-D0_REGIONS[rv])
            rb = np.zeros(nv); rb[rv*4+1] = -GAM; rb[24] = lam
            A_ub.append(rb); b_ub.append(D0_REGIONS[rv])
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=[(0, None)]*nv, method="highs")
    return res


def simulate_economy(shares_T4: np.ndarray, T=10, phi_scale=0.1):
    """Mô phỏng kinh tế Bài 8/12. shares_T4: (T,4) tỷ lệ đầu tư."""
    K, D, AI, H, A = 27500., 20.3, 86., 30., 34.914
    PHI = np.array([0.003, 0.002, 0.004]) * phi_scale
    rows = []
    for t in range(T):
        Y = cobb_douglas(K, L0, D, AI, H, A)
        I = shares_T4[t] * Y
        C = Y - I.sum()
        dD, dAI, dH = I[1]/250, I[2]/40, 0.8*I[3]/300
        U = max(0.5, min(9, 2.5 + 0.012*dAI - 0.5*dH))
        cyber = max(0.0, 0.04*AI - 0.08*H)
        emis = 1.5*(shares_T4[t, 0] + shares_T4[t, 2])
        rows.append(dict(year=2026+t, Y=Y, C=C, K=K, D=D, AI=AI, H=H,
                         U=U, cyber=cyber, emission=emis))
        K = 0.95*K + I[0]
        D = 0.88*D + dD
        AI = 0.85*AI + dAI
        H = 0.98*H + dH
        A *= 1 + PHI[0]*D + PHI[1]*AI + PHI[2]*H
    return pd.DataFrame(rows)


SCENARIO_ALLOCS = {
    "S1 Truyền thống": np.array([0.70, 0.10, 0.10, 0.10]),
    "S2 Số hóa nhanh":  np.array([0.25, 0.45, 0.15, 0.15]),
    "S3 AI dẫn dắt":    np.array([0.20, 0.20, 0.45, 0.15]),
    "S4 Bao trùm số":   np.array([0.30, 0.20, 0.10, 0.40]),
    "S5 Tối ưu (AIDEOM)": np.array([0.10, 0.12, 0.40, 0.00]),
}

def run_scenarios(inv_rate=0.32, T=10):
    results = {}
    for name, alloc in SCENARIO_ALLOCS.items():
        sh = np.tile(alloc * inv_rate, (T, 1))
        df = simulate_economy(sh, T)
        g = (df["Y"].iloc[-1]/df["Y"].iloc[0])**(1/(T-1)) - 1
        results[name] = {
            "df": df, "CAGR": g*100,
            "Y2030": df[df.year==2030]["Y"].values[0] if 2030 in df.year.values else float("nan"),
            "Y2035": df["Y"].iloc[-1],
            "U2035": df["U"].iloc[-1],
            "Cyber2035": df["cyber"].iloc[-1],
            "Welfare": sum(0.97**t * np.log(max(df["C"].iloc[t], 1)) for t in range(T)),
        }
    return results
