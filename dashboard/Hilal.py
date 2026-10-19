"""
Dashboard Fiskal & Pembangunan 97 Kota (2014-2024)
Jalankan:  streamlit run app.py
"""
import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN & TAMPILAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Fiskal 97 Kota",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_fiskal_97_hipoFull.csv")

PALETTE = ["#4F46E5", "#06B6D4", "#F59E0B", "#EF4444", "#10B981", "#8B5CF6", "#EC4899", "#64748B"]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PALETTE

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    .hero {
        background: linear-gradient(120deg, #4F46E5 0%, #06B6D4 100%);
        padding: 1.6rem 2rem; border-radius: 18px; color: white; margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(79,70,229,.25);
    }
    .hero h1 {margin: 0; font-size: 2rem; color: white;}
    .hero p {margin: .3rem 0 0 0; opacity: .92; font-size: 1.02rem;}
    div[data-testid="stMetric"] {
        background: rgba(120,120,160,.08); border: 1px solid rgba(120,120,160,.2);
        padding: .9rem 1rem; border-radius: 14px;
    }
    .insight {
        background: rgba(79,70,229,.08); border-left: 5px solid #4F46E5;
        padding: .8rem 1rem; border-radius: 8px; margin: .4rem 0;
    }
    button[data-baseweb="tab"] {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# DEFINISI VARIABEL
# ----------------------------------------------------------------------------
TRANSFER = ["DBH_PAJAK", "DBH_SDA", "DBH_Lainnya", "DAU_Grant", "DAU_Earmark", "DAK_Fisik", "DAK_Nonfisik"]
BELANJA = [
    "bm_pendidikan", "bm_kesehatan", "bm_perumahan", "bm_lingkungan",
    "bm_ekonomi", "bm_infrastruktur", "bm_pariwisata", "bm_perlindungan_sosial",
]
FISKAL_LAIN = ["PAD", "ICT"]
OUTCOME = ["KEMISKINAN", "PENGANGGURAN", "KETIMPANGAN", "IPM", "PE"]
PREDIKTOR = TRANSFER + FISKAL_LAIN + BELANJA

LABEL = {
    "DBH_PAJAK": "DBH Pajak", "DBH_SDA": "DBH SDA", "DBH_Lainnya": "DBH Lainnya",
    "DAU_Grant": "DAU Grant", "DAU_Earmark": "DAU Earmark",
    "DAK_Fisik": "DAK Fisik", "DAK_Nonfisik": "DAK Nonfisik",
    "PAD": "PAD", "ICT": "ICT",
    "bm_pendidikan": "Belanja Modal Pendidikan", "bm_kesehatan": "Belanja Modal Kesehatan",
    "bm_perumahan": "Belanja Modal Perumahan", "bm_lingkungan": "Belanja Modal Lingkungan",
    "bm_ekonomi": "Belanja Modal Ekonomi", "bm_infrastruktur": "Belanja Modal Infrastruktur",
    "bm_pariwisata": "Belanja Modal Pariwisata", "bm_perlindungan_sosial": "Belanja Modal Perlindungan Sosial",
    "KEMISKINAN": "Kemiskinan", "PENGANGGURAN": "Pengangguran",
    "KETIMPANGAN": "Ketimpangan (Gini)", "IPM": "IPM", "PE": "Pertumbuhan Ekonomi",
}
HIGHER_BETTER = {"KEMISKINAN": False, "PENGANGGURAN": False, "KETIMPANGAN": False, "IPM": True, "PE": True}


def lab(c):
    return LABEL.get(c, c)


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Memuat data…")
def load_data(file_bytes: bytes | None) -> pd.DataFrame:
    src = io.BytesIO(file_bytes) if file_bytes is not None else DEFAULT_FILE
    df = pd.read_csv(src)
    df["Tahun"] = df["Tahun"].astype(int)
    df["Kota_Pendek"] = df["Kota"].str.replace(r"^(Kota|Kab\.?|Kabupaten)\s+", "", regex=True)
    return df


with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    up = st.file_uploader("Upload dataset lain (opsional, CSV)", type="csv")

try:
    raw = load_data(up.getvalue() if up else None)
except FileNotFoundError:
    st.error(
        "File `dataset_fiskal_97_hipoFull.csv` tidak ditemukan. "
        "Letakkan di folder yang sama dengan `app.py` atau upload lewat sidebar."
    )
    st.stop()

missing = [c for c in ["Kota", "Tahun"] + PREDIKTOR + OUTCOME if c not in raw.columns]
if missing:
    st.error(f"Kolom berikut tidak ada di dataset: {missing}")
    st.stop()

# ----------------------------------------------------------------------------
# FILTER SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔎 Filter Data")
    y_min, y_max = int(raw.Tahun.min()), int(raw.Tahun.max())
    rentang = st.slider("Rentang tahun", y_min, y_max, (y_min, y_max))
    semua_kota = sorted(raw.Kota.unique())
    kota_pilih = st.multiselect(
        "Kota (kosongkan = semua)", semua_kota, placeholder="Semua 97 kota"
    )
    st.caption("Filter berlaku untuk seluruh tab, termasuk model regresi.")

df = raw[(raw.Tahun.between(*rentang))]
if kota_pilih:
    df = df[df.Kota.isin(kota_pilih)]

if df.empty:
    st.warning("Tidak ada data untuk filter yang dipilih.")
    st.stop()

with st.sidebar:
    st.success(f"**{len(df):,}** observasi · **{df.Kota.nunique()}** kota · **{df.Tahun.nunique()}** tahun")

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>📊 Dashboard Fiskal & Pembangunan Kota</h1>
        <p>Dana transfer, PAD, belanja modal, dan capaian pembangunan di {raw.Kota.nunique()} kota
        periode {y_min}–{y_max}. Lengkap dengan analisis regresi berganda interaktif.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_ring, tab_tren, tab_kota, tab_korel, tab_reg, tab_data = st.tabs(
    ["🏠 Ringkasan", "📈 Tren", "🏙️ Perbandingan Kota", "🔗 Korelasi", "🧮 Regresi Berganda", "🗂️ Data"]
)

# ============================================================================
# TAB 1 — RINGKASAN
# ============================================================================
with tab_ring:
    th_awal, th_akhir = df.Tahun.min(), df.Tahun.max()
    st.subheader("Indikator Utama")
    cols = st.columns(5)
    for col, v in zip(cols, OUTCOME):
        akhir = df[df.Tahun == th_akhir][v].mean()
        awal = df[df.Tahun == th_awal][v].mean()
        delta = akhir - awal
        bagus = HIGHER_BETTER[v]
        col.metric(
            lab(v) + f" ({th_akhir})",
            f"{akhir:,.2f}",
            f"{delta:+.2f} vs {th_awal}" if th_akhir != th_awal else None,
            delta_color="normal" if (delta > 0) == bagus else "inverse",
        )
    st.caption("Rata-rata seluruh kota terpilih pada tahun terakhir; panah hijau = membaik, merah = memburuk.")

    c1, c2 = st.columns([1.35, 1])
    with c1:
        st.subheader("Komposisi Dana Transfer per Tahun")
        comp = df.groupby("Tahun")[TRANSFER].mean().reset_index().melt("Tahun", var_name="Komponen", value_name="Rata-rata")
        comp["Komponen"] = comp["Komponen"].map(lab)
        fig = px.area(comp, x="Tahun", y="Rata-rata", color="Komponen")
        fig.update_layout(height=400, legend=dict(orientation="h", y=-0.25), margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Bauran Dana Rata-rata")
        tot = df[TRANSFER + ["PAD"]].mean().rename(index=lab).reset_index()
        tot.columns = ["Komponen", "Nilai"]
        fig = px.pie(tot, names="Komponen", values="Nilai", hole=0.55)
        fig.update_traces(textinfo="percent", pull=[0.03] * len(tot))
        fig.update_layout(height=400, margin=dict(t=10), legend=dict(font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Prioritas Belanja Modal (rata-rata)")
        bm = df[BELANJA].mean().rename(index=lambda x: lab(x).replace("Belanja Modal ", "")).sort_values()
        fig = px.bar(bm, orientation="h", color=bm.values, color_continuous_scale="Viridis")
        fig.update_layout(height=380, showlegend=False, coloraxis_showscale=False,
                          xaxis_title="Rata-rata", yaxis_title="", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.subheader("Heatmap Capaian per Tahun")
        v_hm = st.selectbox("Indikator", OUTCOME, index=3, format_func=lab, key="hm_var")
        piv = df.groupby("Tahun")[v_hm].agg(["mean", "min", "max"]).round(2)
        piv.columns = ["Rata-rata", "Minimum", "Maksimum"]
        fig = px.imshow(piv.T, text_auto=True, aspect="auto", color_continuous_scale="Tealgrn")
        fig.update_layout(height=300, margin=dict(t=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("💡 Insight Cepat")
    last = df[df.Tahun == th_akhir]
    best = last.loc[last.IPM.idxmax()]
    worst_k = last.loc[last.KEMISKINAN.idxmax()]
    st.markdown(
        f"""
        <div class="insight">🏆 IPM tertinggi pada {th_akhir}: <b>{best.Kota}</b> ({best.IPM:.2f}).</div>
        <div class="insight">⚠️ Kemiskinan tertinggi pada {th_akhir}: <b>{worst_k.Kota}</b> ({worst_k.KEMISKINAN:.2f}).</div>
        <div class="insight">💰 Sumber dana terbesar rata-rata: <b>{tot.loc[tot.Nilai.idxmax(), 'Komponen']}</b>
        ({tot.Nilai.max() / tot.Nilai.sum() * 100:.1f}% dari total dana).</div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================================
# TAB 2 — TREN
# ============================================================================
with tab_tren:
    st.subheader("Tren Antar Waktu")
    t1, t2 = st.columns([2, 1])
    var_tren = t1.multiselect(
        "Variabel", PREDIKTOR + OUTCOME, default=["IPM", "KEMISKINAN"], format_func=lab, key="tren_vars"
    )
    mode = t2.radio("Mode", ["Rata-rata semua kota", "Per kota"], horizontal=True)
    norm = st.checkbox("Normalisasi (indeks tahun pertama = 100) agar bisa dibandingkan", value=len(var_tren) > 1)

    if not var_tren:
        st.info("Pilih minimal satu variabel.")
    elif mode == "Rata-rata semua kota":
        g = df.groupby("Tahun")[var_tren].mean()
        if norm:
            g = g / g.iloc[0] * 100
        fig = go.Figure()
        for i, v in enumerate(var_tren):
            fig.add_trace(go.Scatter(x=g.index, y=g[v], name=lab(v), mode="lines+markers",
                                     line=dict(width=3, color=PALETTE[i % len(PALETTE)])))
        fig.update_layout(height=450, hovermode="x unified", yaxis_title="Indeks" if norm else "Nilai",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Sebaran per tahun (boxplot)")
        vb = st.selectbox("Variabel boxplot", var_tren, format_func=lab, key="box_var")
        fig = px.box(df, x="Tahun", y=vb, color="Tahun", points="outliers",
                     color_discrete_sequence=px.colors.sequential.Viridis)
        fig.update_layout(height=400, showlegend=False, yaxis_title=lab(vb))
        st.plotly_chart(fig, use_container_width=True)
    else:
        vsel = st.selectbox("Variabel", var_tren, format_func=lab, key="per_kota_var")
        pilih = st.multiselect(
            "Kota yang ditampilkan", sorted(df.Kota.unique()),
            default=sorted(df.Kota.unique())[:5], key="per_kota_sel",
        )
        if pilih:
            fig = px.line(df[df.Kota.isin(pilih)], x="Tahun", y=vsel, color="Kota", markers=True)
            fig.update_layout(height=450, hovermode="x unified", yaxis_title=lab(vsel))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Distribusi variabel")
    dv1, dv2 = st.columns([1, 3])
    vd = dv1.selectbox("Variabel", PREDIKTOR + OUTCOME, index=len(PREDIKTOR) + 3, format_func=lab, key="dist_var")
    fig = px.histogram(df, x=vd, nbins=40, marginal="violin", color_discrete_sequence=[PALETTE[0]])
    fig.update_layout(height=380, bargap=0.05)
    dv2.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 3 — PERBANDINGAN KOTA
# ============================================================================
with tab_kota:
    st.subheader("Peringkat Kota")
    k1, k2, k3 = st.columns(3)
    var_rank = k1.selectbox("Indikator", OUTCOME + PREDIKTOR, index=3, format_func=lab, key="rank_var")
    thn_rank = k2.selectbox("Tahun", sorted(df.Tahun.unique(), reverse=True), key="rank_year")
    n_top = k3.slider("Jumlah kota", 5, 25, 10)

    d_r = df[df.Tahun == thn_rank].sort_values(var_rank, ascending=False)
    ra, rb = st.columns(2)
    top = d_r.head(n_top).iloc[::-1]
    bot = d_r.tail(n_top)
    f1 = px.bar(top, x=var_rank, y="Kota", orientation="h", color=var_rank, color_continuous_scale="Teal")
    f1.update_layout(title=f"Tertinggi — {lab(var_rank)}", height=120 + n_top * 28, coloraxis_showscale=False, yaxis_title="")
    f2 = px.bar(bot, x=var_rank, y="Kota", orientation="h", color=var_rank, color_continuous_scale="Sunset")
    f2.update_layout(title=f"Terendah — {lab(var_rank)}", height=120 + n_top * 28, coloraxis_showscale=False, yaxis_title="")
    ra.plotly_chart(f1, use_container_width=True)
    rb.plotly_chart(f2, use_container_width=True)

    st.divider()
    st.subheader("🎞️ Animasi Bubble Chart: Hubungan Antar Variabel dari Tahun ke Tahun")
    b1, b2, b3, b4 = st.columns(4)
    bx = b1.selectbox("Sumbu X", PREDIKTOR, index=PREDIKTOR.index("PAD"), format_func=lab, key="bx")
    by = b2.selectbox("Sumbu Y", OUTCOME, index=3, format_func=lab, key="by")
    bs = b3.selectbox("Ukuran bubble", TRANSFER, index=3, format_func=lab, key="bs")
    bc = b4.selectbox("Warna", OUTCOME, index=0, format_func=lab, key="bc")
    fig = px.scatter(
        df.sort_values("Tahun"), x=bx, y=by, size=bs, color=bc, animation_frame="Tahun",
        hover_name="Kota", size_max=32, color_continuous_scale="Turbo",
        range_x=[df[bx].min() * 0.95, df[bx].max() * 1.05],
        range_y=[df[by].min() * 0.95, df[by].max() * 1.05],
    )
    fig.update_layout(height=520, xaxis_title=lab(bx), yaxis_title=lab(by))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🕸️ Radar: Bandingkan Profil Kota")
    kota_all = sorted(df.Kota.unique())
    rad = st.multiselect("Pilih 2–5 kota", kota_all, default=kota_all[:3], max_selections=5, key="radar_sel")
    if len(rad) >= 2:
        g = df.groupby("Kota")[OUTCOME].mean()
        nrm = (g - g.min()) / (g.max() - g.min())
        for v in OUTCOME:
            if not HIGHER_BETTER[v]:
                nrm[v] = 1 - nrm[v]
        cats = [lab(v) + (" (↓ dibalik)" if not HIGHER_BETTER[v] else "") for v in OUTCOME]
        fig = go.Figure()
        for i, k in enumerate(rad):
            vals = nrm.loc[k].tolist()
            fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], fill="toself", name=k,
                                          line_color=PALETTE[i % len(PALETTE)], opacity=0.65))
        fig.update_layout(height=480, polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Skor 0–1 relatif terhadap seluruh kota. Indikator yang semakin rendah semakin baik "
                   "(kemiskinan, pengangguran, ketimpangan) dibalik, sehingga area lebih luas = kinerja lebih baik.")
    else:
        st.info("Pilih minimal 2 kota untuk menampilkan radar.")

# ============================================================================
# TAB 4 — KORELASI
# ============================================================================
with tab_korel:
    st.subheader("Matriks Korelasi")
    grup = st.multiselect(
        "Kelompok variabel", ["Dana Transfer", "PAD & ICT", "Belanja Modal", "Outcome"],
        default=["Dana Transfer", "PAD & ICT", "Outcome"],
    )
    mapg = {"Dana Transfer": TRANSFER, "PAD & ICT": FISKAL_LAIN, "Belanja Modal": BELANJA, "Outcome": OUTCOME}
    vars_c = [v for g in grup for v in mapg[g]]
    metode = st.radio("Metode", ["pearson", "spearman"], horizontal=True)
    if len(vars_c) >= 2:
        corr = df[vars_c].corr(method=metode)
        corr.index = [lab(c) for c in corr.index]
        corr.columns = [lab(c) for c in corr.columns]
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=max(450, 32 * len(vars_c)))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Korelasi terkuat")
        cc = df[vars_c].corr(method=metode)
        pairs = cc.where(np.triu(np.ones(cc.shape), k=1).astype(bool)).stack().reset_index()
        pairs.columns = ["Variabel A", "Variabel B", "r"]
        pairs["|r|"] = pairs["r"].abs()
        pairs = pairs.sort_values("|r|", ascending=False).head(10)
        pairs["Variabel A"] = pairs["Variabel A"].map(lab)
        pairs["Variabel B"] = pairs["Variabel B"].map(lab)
        st.dataframe(pairs[["Variabel A", "Variabel B", "r"]].round(3), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Korelasi terhadap Satu Variabel Target")
    tg = st.selectbox("Target", OUTCOME, index=3, format_func=lab, key="corr_target")
    ct = df[PREDIKTOR + [tg]].corr()[tg].drop(tg).sort_values()
    ct.index = [lab(i) for i in ct.index]
    fig = px.bar(ct, orientation="h", color=ct.values, color_continuous_scale="RdBu_r", range_color=[-1, 1])
    fig.update_layout(height=520, coloraxis_showscale=False, xaxis_title=f"Korelasi dengan {lab(tg)}", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Scatter dengan Garis Regresi Sederhana")
    s1, s2, s3 = st.columns(3)
    sx = s1.selectbox("X", PREDIKTOR, format_func=lab, key="sx")
    sy = s2.selectbox("Y", OUTCOME, index=3, format_func=lab, key="sy")
    sc = s3.selectbox("Warna titik", ["(tidak ada)", "Tahun"] + OUTCOME, format_func=lambda x: lab(x) if x in LABEL else x, key="sc")
    fig = px.scatter(df, x=sx, y=sy, color=None if sc == "(tidak ada)" else sc, trendline="ols",
                     hover_name="Kota", opacity=0.65, trendline_color_override="#EF4444")
    fig.update_layout(height=450, xaxis_title=lab(sx), yaxis_title=lab(sy))
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 5 — REGRESI BERGANDA
# ============================================================================
def fit_ols(data, y, xs, std=False, fe_city=False, fe_year=False, se="Biasa"):
    d = data[[y] + xs + ["Kota", "Tahun"]].dropna().copy()
    if std:
        for c in [y] + xs:
            d[c] = (d[c] - d[c].mean()) / d[c].std(ddof=0)
    X = d[xs].copy()
    if fe_city:
        X = X.join(pd.get_dummies(d["Kota"], prefix="kota", drop_first=True, dtype=float))
    if fe_year:
        X = X.join(pd.get_dummies(d["Tahun"].astype(str), prefix="thn", drop_first=True, dtype=float))
    X = sm.add_constant(X, has_constant="add")
    base = sm.OLS(d[y], X)
    if se == "Robust (HC3)":
        res = base.fit(cov_type="HC3")
    elif se == "Cluster per Kota":
        res = base.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["Kota"])[0]})
    else:
        res = base.fit()
    return res, d, X


with tab_reg:
    st.subheader("🧮 Regresi Linear Berganda (OLS)")
    st.caption("Pilih variabel dependen (Y) dan beberapa variabel independen (X), lalu atur spesifikasi model.")

    r1, r2 = st.columns([1, 2])
    y_var = r1.selectbox("Variabel dependen (Y)", OUTCOME, index=3, format_func=lab, key="reg_y")
    default_x = ["PAD", "DAU_Grant", "DAK_Fisik", "DBH_PAJAK", "ICT"]
    x_vars = r2.multiselect(
        "Variabel independen (X)", PREDIKTOR, default=default_x, format_func=lab, key="reg_x"
    )

    o1, o2, o3, o4 = st.columns(4)
    std = o1.toggle("Standardisasi (beta)", help="Semua variabel diubah ke z-score sehingga koefisien bisa dibandingkan antar variabel.")
    fe_c = o2.toggle("Fixed effect Kota", help="Menambah dummy kota untuk mengontrol karakteristik kota yang tetap.")
    fe_t = o3.toggle("Fixed effect Tahun", help="Menambah dummy tahun untuk mengontrol guncangan umum tiap tahun.")
    se_type = o4.selectbox("Standard error", ["Biasa", "Robust (HC3)", "Cluster per Kota"], index=0)

    if len(x_vars) < 2:
        st.info("Pilih **minimal 2 variabel independen** untuk regresi berganda.")
    elif len(df) <= len(x_vars) + 2:
        st.warning("Data terlalu sedikit untuk model ini. Perluas filter.")
    else:
        model, d_fit, X_fit = fit_ols(df, y_var, x_vars, std, fe_c, fe_t, se_type)
        yhat = model.fittedvalues
        resid = model.resid

        m = st.columns(6)
        m[0].metric("R²", f"{model.rsquared:.4f}")
        m[1].metric("Adj. R²", f"{model.rsquared_adj:.4f}")
        try:
            m[2].metric("F-statistic", f"{model.fvalue:,.2f}")
            m[3].metric("Prob (F)", fmt_p(float(model.f_pvalue)))
        except Exception:
            m[2].metric("F-statistic", "n/a")
            m[3].metric("Prob (F)", "n/a")
        m[4].metric("RMSE", f"{np.sqrt(np.mean(resid ** 2)):.4f}")
        m[5].metric("N", f"{int(model.nobs):,}")

        ci = model.conf_int()
        coef = pd.DataFrame({
            "Variabel": model.params.index, "Koefisien": model.params.values, "Std. Error": model.bse.values,
            "t": model.tvalues.values, "p-value": model.pvalues.values,
            "CI 2.5%": ci[0].values, "CI 97.5%": ci[1].values,
        })
        coef = coef[~coef.Variabel.str.startswith(("kota_", "thn_"))].copy()
        coef["Sig."] = coef["p-value"].map(stars)
        coef["Label"] = coef.Variabel.map(lambda v: "Konstanta" if v == "const" else lab(v))
        n_fe = len(model.params) - len(coef)
        if n_fe:
            st.caption(f"ℹ️ {n_fe} dummy fixed effect diikutkan dalam model namun tidak ditampilkan pada tabel.")

        sub_ringkas, sub_koef, sub_diag, sub_part, sub_3d, sub_sim = st.tabs(
            ["📋 Ringkasan & Interpretasi", "📊 Koefisien", "🩺 Diagnostik", "🔬 Partial Regression", "🧊 Bidang Regresi 3D", "🎛️ Simulator"]
        )

        with sub_ringkas:
            st.markdown("#### Persamaan model")
            terms = " ".join(
                f"{'+' if r.Koefisien >= 0 else '−'} {abs(r.Koefisien):.4f}·{r.Label}"
                for r in coef[coef.Variabel != "const"].itertuples()
            )
            const_val = coef.loc[coef.Variabel == "const", "Koefisien"].iloc[0]
            st.code(f"{lab(y_var)}  =  {const_val:.4f} {terms}", language="text")

            st.markdown("#### Interpretasi otomatis")
            sigs = coef[(coef.Variabel != "const") & (coef["p-value"] < 0.05)]
            satuan = "simpangan baku" if std else "satuan"
            if sigs.empty:
                st.markdown(
                    '<div class="insight">Tidak ada variabel independen yang signifikan pada α = 5% '
                    "dalam spesifikasi ini.</div>", unsafe_allow_html=True)
            for _, r in sigs.iterrows():
                arah = "naik" if r["Koefisien"] > 0 else "turun"
                st.markdown(
                    f'<div class="insight">Setiap kenaikan 1 {satuan} <b>{r["Label"]}</b>, '
                    f"<b>{lab(y_var)}</b> diperkirakan <b>{arah} {abs(r['Koefisien']):.4f} {satuan}</b> "
                    f"(ceteris paribus, p = {fmt_p(r['p-value'])}).</div>", unsafe_allow_html=True)
            tak = coef[(coef.Variabel != "const") & (coef["p-value"] >= 0.05)].Label.tolist()
            if tak:
                st.markdown(f"Tidak signifikan pada 5%: **{', '.join(tak)}**.")
            st.markdown(
                f"Model menjelaskan **{model.rsquared * 100:.2f}%** variasi {lab(y_var)} "
                f"(Adj. R² = {model.rsquared_adj * 100:.2f}%)."
            )
            if model.rsquared < 0.1:
                st.warning("R² sangat rendah: variabel yang dipilih belum menjelaskan variasi Y secara berarti. "
                           "Coba kombinasi variabel lain atau aktifkan fixed effect.")

            with st.expander("Lihat output lengkap statsmodels"):
                st.text(model.summary().as_text())
            st.download_button("⬇️ Unduh output regresi (.txt)", model.summary().as_text(),
                               file_name="hasil_regresi.txt")

        with sub_koef:
            tampil = coef[["Label", "Koefisien", "Std. Error", "t", "p-value", "CI 2.5%", "CI 97.5%", "Sig."]]
            tampil = tampil.rename(columns={"Label": "Variabel"})
            st.dataframe(
                tampil.style.format({"Koefisien": "{:.4f}", "Std. Error": "{:.4f}", "t": "{:.3f}",
                                     "p-value": "{:.4f}", "CI 2.5%": "{:.4f}", "CI 97.5%": "{:.4f}"})
                .background_gradient(subset=["Koefisien"], cmap="RdBu_r", vmin=-tampil["Koefisien"].abs().max(),
                                     vmax=tampil["Koefisien"].abs().max()),
                hide_index=True, use_container_width=True,
            )
            st.caption("*** p<0.01 · ** p<0.05 · * p<0.10")
            st.download_button("⬇️ Unduh koefisien (.csv)", tampil.to_csv(index=False), file_name="koefisien_regresi.csv")

            cf = coef[coef.Variabel != "const"].sort_values("Koefisien")
            cf = cf.assign(Signifikan=np.where(cf["p-value"] < 0.05, "Signifikan (p<0.05)", "Tidak signifikan"))
            fig = go.Figure()
            for s, colr in [("Signifikan (p<0.05)", PALETTE[0]), ("Tidak signifikan", "#9CA3AF")]:
                sub = cf[cf.Signifikan == s]
                fig.add_trace(go.Scatter(
                    x=sub.Koefisien, y=sub.Label, mode="markers", name=s,
                    marker=dict(size=12, color=colr),
                    error_x=dict(type="data", symmetric=False,
                                 array=sub["CI 97.5%"] - sub.Koefisien, arrayminus=sub.Koefisien - sub["CI 2.5%"],
                                 color=colr),
                ))
            fig.add_vline(x=0, line_dash="dash", line_color="#EF4444")
            fig.update_layout(title="Forest plot koefisien dengan interval kepercayaan 95%", height=120 + 55 * len(cf),
                              xaxis_title="Koefisien" + (" (standar)" if std else ""), legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)

        with sub_diag:
            a, b = st.columns(2)
            with a:
                lo, hi = float(min(d_fit[y_var].min(), yhat.min())), float(max(d_fit[y_var].max(), yhat.max()))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=yhat, y=d_fit[y_var], mode="markers", name="Observasi",
                                         marker=dict(color=PALETTE[0], opacity=0.5, size=6),
                                         text=raw.loc[d_fit.index, "Kota"] + " " + raw.loc[d_fit.index, "Tahun"].astype(str),
                                         hovertemplate="%{text}<br>Prediksi %{x:.3f}<br>Aktual %{y:.3f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Garis ideal (y = x)",
                                         line=dict(color="#EF4444", dash="dash")))
                fig.update_layout(title="Aktual vs Prediksi", xaxis_title="Prediksi", yaxis_title="Aktual", height=420,
                                  legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig, use_container_width=True)
            with b:
                lw = lowess(resid, yhat, frac=0.5)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=yhat, y=resid, mode="markers", name="Residual",
                                         marker=dict(color=PALETTE[1], opacity=0.5, size=6)))
                fig.add_trace(go.Scatter(x=lw[:, 0], y=lw[:, 1], mode="lines", name="LOWESS",
                                         line=dict(color="#EF4444", width=3)))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(title="Residual vs Fitted (cek linearitas & homoskedastisitas)",
                                  xaxis_title="Fitted", yaxis_title="Residual", height=420, legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig, use_container_width=True)

            c, d = st.columns(2)
            with c:
                (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", marker=dict(color=PALETTE[4], opacity=0.6), name="Residual"))
                fig.add_trace(go.Scatter(x=[osm.min(), osm.max()], y=[slope * osm.min() + intercept, slope * osm.max() + intercept],
                                         mode="lines", line=dict(color="#EF4444", dash="dash"), name="Normal"))
                fig.update_layout(title="Q-Q Plot residual", xaxis_title="Kuantil teoretis", yaxis_title="Kuantil sampel",
                                  height=380, legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig, use_container_width=True)
            with d:
                fig = px.histogram(resid, nbins=40, marginal="box", color_discrete_sequence=[PALETTE[5]])
                fig.update_layout(title="Distribusi residual", xaxis_title="Residual", showlegend=False, height=380)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Uji asumsi klasik")
            jb_stat, jb_p = stats.jarque_bera(resid)
            try:
                bp_stat, bp_p, _, _ = het_breuschpagan(resid, X_fit)
            except Exception:
                bp_stat, bp_p = np.nan, np.nan
            dw = durbin_watson(resid)
            uji = pd.DataFrame([
                ["Normalitas residual (Jarque-Bera)", jb_stat, jb_p, "Normal" if jb_p > 0.05 else "Tidak normal"],
                ["Heteroskedastisitas (Breusch-Pagan)", bp_stat, bp_p, "Homoskedastis" if bp_p > 0.05 else "Heteroskedastis"],
                ["Autokorelasi (Durbin-Watson)", dw, np.nan, "Mendekati 2 = tidak ada autokorelasi"],
            ], columns=["Uji", "Statistik", "p-value", "Kesimpulan (α=5%)"])
            st.dataframe(uji.style.format({"Statistik": "{:.4f}", "p-value": "{:.4f}"}, na_rep="—"),
                         hide_index=True, use_container_width=True)

            st.markdown("#### Multikolinearitas (VIF)")
            Xv = sm.add_constant(d_fit[x_vars])
            vif = pd.DataFrame({
                "Variabel": [lab(v) for v in x_vars],
                "VIF": [variance_inflation_factor(Xv.values, i + 1) for i in range(len(x_vars))],
            })
            vif["Status"] = np.where(vif.VIF < 5, "✅ Aman", np.where(vif.VIF < 10, "⚠️ Moderat", "❌ Tinggi"))
            va, vb = st.columns([1, 1.3])
            va.dataframe(vif.style.format({"VIF": "{:.3f}"}), hide_index=True, use_container_width=True)
            fig = px.bar(vif, x="VIF", y="Variabel", orientation="h", color="VIF",
                         color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"], range_color=[1, 10])
            fig.add_vline(x=5, line_dash="dot", line_color="orange")
            fig.add_vline(x=10, line_dash="dot", line_color="red")
            fig.update_layout(height=60 + 40 * len(vif), coloraxis_showscale=False, yaxis_title="")
            vb.plotly_chart(fig, use_container_width=True)
            st.caption("Aturan umum: VIF < 5 aman, 5–10 moderat, > 10 mengindikasikan multikolinearitas serius.")

        with sub_part:
            st.markdown(
                "Setiap panel menunjukkan hubungan **X terhadap Y setelah mengontrol semua variabel lain** "
                "dalam model. Kemiringan garis = koefisien regresi berganda variabel tersebut."
            )
            n = len(x_vars)
            ncol = 2 if n <= 4 else 3
            nrow = int(np.ceil(n / ncol))
            yv = d_fit[y_var].values
            Xa = X_fit.values
            names = list(X_fit.columns)
            av = []
            for xv in x_vars:
                j = names.index(xv)
                others = np.delete(Xa, j, axis=1)
                ey = yv - others @ np.linalg.lstsq(others, yv, rcond=None)[0]
                ex = Xa[:, j] - others @ np.linalg.lstsq(others, Xa[:, j], rcond=None)[0]
                av.append((ex, ey, (ex @ ey) / (ex @ ex)))
            fig = make_subplots(
                rows=nrow, cols=ncol, vertical_spacing=0.16 if nrow > 1 else 0.1,
                subplot_titles=[f"{lab(xv)} (β = {a_[2]:.4f})" for xv, a_ in zip(x_vars, av)],
            )
            for i, (xv, (ex, ey, slope)) in enumerate(zip(x_vars, av)):
                r_, c_ = divmod(i, ncol)
                fig.add_trace(go.Scatter(x=ex, y=ey, mode="markers",
                                         marker=dict(color=PALETTE[i % len(PALETTE)], size=5, opacity=0.45),
                                         showlegend=False, hoverinfo="skip"), row=r_ + 1, col=c_ + 1)
                xl = np.array([ex.min(), ex.max()])
                fig.add_trace(go.Scatter(x=xl, y=slope * xl, mode="lines", line=dict(color="#EF4444", width=3),
                                         showlegend=False), row=r_ + 1, col=c_ + 1)
                fig.update_xaxes(title_text=f"{lab(xv)} | lainnya", row=r_ + 1, col=c_ + 1, title_font_size=11)
                fig.update_yaxes(title_text=f"{lab(y_var)} | lainnya", row=r_ + 1, col=c_ + 1, title_font_size=11)
            fig.update_layout(height=340 * nrow + 40)
            st.plotly_chart(fig, use_container_width=True)

        with sub_3d:
            st.markdown("Visualisasi **bidang regresi** untuk dua variabel X terpilih (Y diregresikan pada keduanya). "
                        "Putar grafik dengan drag mouse.")
            p1, p2 = st.columns(2)
            x1 = p1.selectbox("Sumbu X₁", x_vars, index=0, format_func=lab, key="x3d1")
            x2 = p2.selectbox("Sumbu X₂", [v for v in x_vars if v != x1], index=0, format_func=lab, key="x3d2")
            d3 = df[[y_var, x1, x2, "Kota", "Tahun"]].dropna()
            m3 = sm.OLS(d3[y_var], sm.add_constant(d3[[x1, x2]])).fit()
            g1 = np.linspace(d3[x1].quantile(0.01), d3[x1].quantile(0.99), 30)
            g2 = np.linspace(d3[x2].quantile(0.01), d3[x2].quantile(0.99), 30)
            G1, G2 = np.meshgrid(g1, g2)
            Z = m3.params["const"] + m3.params[x1] * G1 + m3.params[x2] * G2
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=d3[x1], y=d3[x2], z=d3[y_var], mode="markers", name="Observasi",
                marker=dict(size=3.5, color=m3.resid, colorscale="RdBu", cmid=0, opacity=0.85,
                            colorbar=dict(title="Residual", x=1.02)),
                text=d3.Kota + " " + d3.Tahun.astype(str),
                hovertemplate="%{text}<br>X₁ %{x:.2f}<br>X₂ %{y:.2f}<br>Y %{z:.2f}<extra></extra>"))
            fig.add_trace(go.Surface(x=G1, y=G2, z=Z, opacity=0.55, colorscale="Viridis", showscale=False, name="Bidang regresi"))
            fig.update_layout(height=650, margin=dict(l=0, r=0, t=30, b=0),
                              scene=dict(xaxis_title=lab(x1), yaxis_title=lab(x2), zaxis_title=lab(y_var)))
            st.plotly_chart(fig, use_container_width=True)
            e1, e2, e3 = st.columns(3)
            e1.metric(f"β {lab(x1)}", f"{m3.params[x1]:.4f}", f"p = {fmt_p(m3.pvalues[x1])}", delta_color="off")
            e2.metric(f"β {lab(x2)}", f"{m3.params[x2]:.4f}", f"p = {fmt_p(m3.pvalues[x2])}", delta_color="off")
            e3.metric("R² (2 variabel)", f"{m3.rsquared:.4f}")
            st.caption("Warna titik = residual (biru: aktual di atas bidang, merah: di bawah bidang).")

        with sub_sim:
            st.markdown("Ubah nilai variabel independen untuk melihat **prediksi** variabel dependen "
                        "(memakai model OLS biasa tanpa standardisasi & fixed effect pada X terpilih).")
            ms, ds, _ = fit_ols(df, y_var, x_vars, False, False, False, "Biasa")
            base_vals = ds[x_vars].mean()
            sl = st.columns(min(len(x_vars), 3))
            nilai = {}
            for i, xv in enumerate(x_vars):
                lo_, hi_ = float(ds[xv].min()), float(ds[xv].max())
                nilai[xv] = sl[i % len(sl)].slider(lab(xv), lo_, hi_, float(base_vals[xv]), key=f"sim_{xv}")
            exog = pd.DataFrame([{"const": 1.0, **nilai}])[ms.params.index]
            pred = ms.get_prediction(exog).summary_frame(alpha=0.05)
            p_val = float(pred["mean"].iloc[0])
            p_base = float(ms.predict(pd.DataFrame([{"const": 1.0, **base_vals.to_dict()}])[ms.params.index]).iloc[0])

            g1_, g2_ = st.columns([1, 1.4])
            with g1_:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=p_val,
                    delta=dict(reference=p_base, valueformat=".3f"),
                    title=dict(text=f"Prediksi {lab(y_var)}"),
                    number=dict(valueformat=".3f"),
                    gauge=dict(axis=dict(range=[float(ds[y_var].min()), float(ds[y_var].max())]),
                               bar=dict(color=PALETTE[0]),
                               threshold=dict(line=dict(color="#EF4444", width=4), value=p_base)),
                ))
                fig.update_layout(height=320, margin=dict(t=60, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Interval prediksi rata-rata 95%: {pred['mean_ci_lower'].iloc[0]:.3f} – {pred['mean_ci_upper'].iloc[0]:.3f}. "
                           "Garis merah = prediksi pada nilai rata-rata semua X.")
            with g2_:
                kontribusi = {lab(xv): ms.params[xv] * (nilai[xv] - base_vals[xv]) for xv in x_vars}
                fig = go.Figure(go.Waterfall(
                    orientation="v", measure=["absolute"] + ["relative"] * len(kontribusi) + ["total"],
                    x=["Prediksi pada rata-rata X"] + list(kontribusi.keys()) + ["Prediksi simulasi"],
                    y=[p_base] + list(kontribusi.values()) + [0],
                    increasing=dict(marker_color="#10B981"), decreasing=dict(marker_color="#EF4444"),
                    totals=dict(marker_color=PALETTE[0]),
                ))
                fig.update_layout(title="Kontribusi tiap variabel terhadap perubahan prediksi", height=360,
                                  yaxis_title=lab(y_var), xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 6 — DATA
# ============================================================================
with tab_data:
    st.subheader("Eksplorasi Data")
    cols_show = st.multiselect("Kolom ditampilkan", list(raw.columns.drop("Kota_Pendek")),
                               default=["Kota", "Tahun"] + OUTCOME + ["PAD", "DAU_Grant"])
    cari = st.text_input("Cari nama kota")
    dv = df if not cari else df[df.Kota.str.contains(cari, case=False)]
    st.dataframe(dv[cols_show], hide_index=True, use_container_width=True, height=420)
    st.download_button("⬇️ Unduh data terfilter (.csv)", dv[cols_show].to_csv(index=False), file_name="data_terfilter.csv")
    st.markdown("##### Statistik deskriptif")
    desc = df[PREDIKTOR + OUTCOME].describe().T.rename(index=lab).round(3)
    st.dataframe(desc, use_container_width=True)