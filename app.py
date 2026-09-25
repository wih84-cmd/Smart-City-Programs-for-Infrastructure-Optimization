import uuid

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ===========================================================
# 0. 상수 & 기본 설정
# ===========================================================
GRID_SIZE = 10
ZONE_TYPES = ["주거 구역", "상업 구역", "공업 구역", "녹지 구역"]
FACILITY_TYPES = ["경찰서", "병원", "회사"]

FACILITY_STYLE = {
    "경찰서": {"color": "#2563eb", "icon": "🚓"},
    "병원":   {"color": "#dc2626", "icon": "🏥"},
    "회사":   {"color": "#f59e0b", "icon": "🏢"},
}

DEFAULT_FACILITIES = [
    {"id": "f1", "name": "중앙 경찰서",   "type": "경찰서", "x": 2, "y": 3, "capacity": 150, "radius": 3.0},
    {"id": "f2", "name": "동부 경찰서",   "type": "경찰서", "x": 8, "y": 7, "capacity": 100, "radius": 2.5},
    {"id": "f3", "name": "IT 테크노밸리", "type": "회사",   "x": 5, "y": 5, "capacity": 200, "radius": 3.5},
    {"id": "f4", "name": "시립 종합병원", "type": "병원",   "x": 3, "y": 8, "capacity": 180, "radius": 4.0},
]

st.set_page_config(
    page_title="스마트 도시 인프라 시뮬레이터",
    page_icon="🏙️",
    layout="wide",
)

# ---- 커스텀 스타일 -----------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.6rem; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px 16px 10px 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; color: #6b7280; }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: white;
    }
    .facility-row {
        display: flex; align-items: center; gap: 8px;
        padding: 6px 8px; border-radius: 8px;
        border: 1px solid #eee; margin-bottom: 6px; font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏙️ 스마트 도시 인프라 과부화 분석 · AI 추천 배치")
st.caption("도시 구역별 인프라 과부화 현황을 시각화하고, 신규 시설의 최적 배치 위치를 추천합니다.")


def risk_level(value: float) -> tuple[str, str]:
    """과부화 지수를 등급/색상으로 변환"""
    if value >= 150:
        return "심각", "#dc2626"
    if value >= 100:
        return "위험", "#f97316"
    if value >= 70:
        return "주의", "#eab308"
    return "양호", "#16a34a"


def badge_html(value: float) -> str:
    label, color = risk_level(value)
    return f'<span class="badge" style="background:{color}">{label} · {value:.0f}%</span>'


# ===========================================================
# 1. 기본 도시 격자 생성 (고정 시드 - 세션 내내 동일)
# ===========================================================
@st.cache_data
def generate_city_grid() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    weights = [0.4, 0.3, 0.2, 0.1]
    demand_ranges = {
        "주거 구역": (60, 95),
        "상업 구역": (70, 100),
        "공업 구역": (40, 80),
        "녹지 구역": (10, 30),
    }

    rows = []
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            zone = rng.choice(ZONE_TYPES, p=weights)
            lo, hi = demand_ranges[zone]
            rows.append({"x": x, "y": y, "zone": zone, "demand": rng.integers(lo, hi + 1)})
    return pd.DataFrame(rows)


city_df = generate_city_grid()

# ===========================================================
# 2. 세션 상태 초기화
# ===========================================================
if "facilities" not in st.session_state:
    st.session_state.facilities = [dict(f) for f in DEFAULT_FACILITIES]


def add_facility(name, ftype, x, y, cap, rad):
    st.session_state.facilities.append(
        {"id": uuid.uuid4().hex[:8], "name": name, "type": ftype, "x": x, "y": y, "capacity": cap, "radius": rad}
    )


def remove_facilities(ids_to_remove: set):
    st.session_state.facilities = [f for f in st.session_state.facilities if f["id"] not in ids_to_remove]


def reset_facilities():
    st.session_state.facilities = [dict(f) for f in DEFAULT_FACILITIES]


# ===========================================================
# 3. 사이드바 - 인프라 시설 관리
# ===========================================================
st.sidebar.header("⚙️ 인프라 시설 관리")

facility_type_filter = st.sidebar.selectbox("분석할 인프라 유형", ["전체"] + FACILITY_TYPES)

st.sidebar.markdown("---")
st.sidebar.subheader("📋 등록된 시설")

if not st.session_state.facilities:
    st.sidebar.info("등록된 시설이 없습니다.")
else:
    for f in st.session_state.facilities:
        style = FACILITY_STYLE[f["type"]]
        st.sidebar.markdown(
            f'<div class="facility-row">{style["icon"]} <b>{f["name"]}</b>'
            f'&nbsp;<span style="color:#6b7280">({f["type"]} · 좌표 {f["x"]},{f["y"]} · 용량 {f["capacity"]})</span></div>',
            unsafe_allow_html=True,
        )

    del_options = {f'{FACILITY_STYLE[f["type"]]["icon"]} {f["name"]}': f["id"] for f in st.session_state.facilities}
    to_delete_labels = st.sidebar.multiselect("삭제할 시설 선택", list(del_options.keys()))
    col_del, col_reset = st.sidebar.columns(2)
    if col_del.button("🗑️ 삭제", use_container_width=True, disabled=not to_delete_labels):
        remove_facilities({del_options[label] for label in to_delete_labels})
        st.rerun()
    if col_reset.button("↺ 초기화", use_container_width=True):
        reset_facilities()
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("➕ 신규 시설 추가")
with st.sidebar.form("add_facility_form", clear_on_submit=True):
    f_name = st.text_input("시설 이름", "")
    f_type = st.selectbox("시설 유형", FACILITY_TYPES)
    c1, c2 = st.columns(2)
    f_x = c1.slider("X 좌표", 0, GRID_SIZE - 1, 5)
    f_y = c2.slider("Y 좌표", 0, GRID_SIZE - 1, 5)
    f_cap = st.number_input("수용 용량", min_value=50, max_value=500, value=120, step=10)
    f_rad = st.slider("커버리지 반경", 1.0, 5.0, 3.0, step=0.5)
    submitted = st.form_submit_button("시설 추가", use_container_width=True)

    if submitted:
        clean_name = f_name.strip()
        if not clean_name:
            st.sidebar.error("시설 이름을 입력해 주세요.")
        else:
            add_facility(clean_name, f_type, f_x, f_y, f_cap, f_rad)
            st.sidebar.success(f"'{clean_name}' 시설이 추가되었습니다!")
            st.rerun()


# ===========================================================
# 4. 과부화 계산 엔진 (거리 감쇄 모델)
# ===========================================================
def calculate_overload(grid_df: pd.DataFrame, facilities: list, selected_type: str = "전체"):
    """
    공급량 = Σ Capacity / (1 + (거리 / 반경)^2)
    과부화 지수(%) = 수요량 / 공급량 * 100  (공급 0 → 임계값 200 부여)
    """
    filtered = [f for f in facilities if selected_type == "전체" or f["type"] == selected_type]

    xs = np.arange(GRID_SIZE)
    ys = np.arange(GRID_SIZE)
    gx, gy = np.meshgrid(xs, ys)  # gx, gy shape: (GRID_SIZE, GRID_SIZE), indexed [y, x]

    supply_matrix = np.zeros((GRID_SIZE, GRID_SIZE))
    for fac in filtered:
        dist = np.sqrt((gx - fac["x"]) ** 2 + (gy - fac["y"]) ** 2)
        supply_matrix += fac["capacity"] / (1.0 + (dist / fac["radius"]) ** 2)

    result = grid_df.copy()
    supply_vals = supply_matrix[result["y"].to_numpy(), result["x"].to_numpy()]
    result["supply"] = np.round(supply_vals, 1)
    result["overload_index"] = np.where(
        supply_vals == 0, 200.0, np.round(result["demand"] / np.maximum(supply_vals, 1e-9) * 100.0, 1)
    )
    return result, filtered


def recommend_best_location(grid_df: pd.DataFrame, facilities: list, selected_type: str) -> dict:
    """선택된 유형(또는 전체일 경우 모든 유형)에 대해 과부화가 가장 심한 지점 추천"""
    candidate_types = FACILITY_TYPES if selected_type == "전체" else [selected_type]

    best = None
    for t in candidate_types:
        analyzed, _ = calculate_overload(grid_df, facilities, t)
        top = analyzed.loc[analyzed["overload_index"].idxmax()]
        candidate = {
            "x": int(top["x"]),
            "y": int(top["y"]),
            "zone": top["zone"],
            "overload_index": float(top["overload_index"]),
            "suggested_type": t,
        }
        if best is None or candidate["overload_index"] > best["overload_index"]:
            best = candidate
    return best


analyzed_df, active_facs = calculate_overload(city_df, st.session_state.facilities, facility_type_filter)
rec_location = recommend_best_location(city_df, st.session_state.facilities, facility_type_filter)

# ===========================================================
# 5. 상단 핵심 지표
# ===========================================================
avg_overload = analyzed_df["overload_index"].mean()
max_overload = analyzed_df["overload_index"].max()
high_risk_count = int((analyzed_df["overload_index"] >= 100).sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("도시 평균 부하율", f"{avg_overload:.1f}%")
m2.metric("최대 부하 구역", f"{max_overload:.1f}%")
m3.metric("위험 이상 구역 수", f"{high_risk_count} / {GRID_SIZE * GRID_SIZE} 개")
m4.metric(
    "AI 추천 신설 위치",
    f"({rec_location['x']}, {rec_location['y']})",
    f"{FACILITY_STYLE[rec_location['suggested_type']]['icon']} {rec_location['suggested_type']}",
)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 과부화 지도", "💡 AI 추천 배치 비교", "📋 상세 데이터"])

# ---- 공통: 히트맵 그리드 -------------------------------------------------
def to_grid(df: pd.DataFrame, value_col: str) -> np.ndarray:
    z = np.zeros((GRID_SIZE, GRID_SIZE))
    z[df["y"].to_numpy(), df["x"].to_numpy()] = df[value_col].to_numpy()
    return z


# ===========================================================
# TAB 1 — 과부화 지도
# ===========================================================
with tab1:
    col_map, col_info = st.columns([7, 3])

    with col_map:
        st.subheader("도시 격자별 인프라 과부화 히트맵")
        z_data = to_grid(analyzed_df, "overload_index")

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=z_data,
                x=list(range(GRID_SIZE)),
                y=list(range(GRID_SIZE)),
                colorscale="Reds",
                zmin=0,
                colorbar=dict(title="부하율(%)"),
                hovertemplate="X: %{x} · Y: %{y}<br>부하율: %{z:.1f}%<extra></extra>",
            )
        )

        if active_facs:
            for ftype in FACILITY_TYPES:
                subset = [f for f in active_facs if f["type"] == ftype]
                if not subset:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=[f["x"] for f in subset],
                        y=[f["y"] for f in subset],
                        mode="markers+text",
                        marker=dict(
                            symbol="square",
                            size=16,
                            color=FACILITY_STYLE[ftype]["color"],
                            line=dict(width=2, color="white"),
                        ),
                        text=[f["name"] for f in subset],
                        textposition="top center",
                        name=f'{FACILITY_STYLE[ftype]["icon"]} {ftype}',
                        hovertext=[f"{f['name']} · 용량 {f['capacity']} · 반경 {f['radius']}" for f in subset],
                        hoverinfo="text",
                    )
                )

        fig.update_layout(
            xaxis=dict(title="X 좌표 (동-서)", dtick=1),
            yaxis=dict(title="Y 좌표 (남-북)", dtick=1),
            height=560,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.subheader("구역 유형별 평균 부하율")
        zone_summary = analyzed_df.groupby("zone", as_index=False)["overload_index"].mean()
        fig_bar = px.bar(
            zone_summary.sort_values("overload_index"),
            x="overload_index",
            y="zone",
            orientation="h",
            color="overload_index",
            color_continuous_scale="Reds",
            labels={"overload_index": "평균 부하율(%)", "zone": ""},
        )
        fig_bar.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("시설 유형별 현황")
        if st.session_state.facilities:
            type_counts = pd.Series([f["type"] for f in st.session_state.facilities]).value_counts()
            for t in FACILITY_TYPES:
                cnt = int(type_counts.get(t, 0))
                st.write(f'{FACILITY_STYLE[t]["icon"]} **{t}**: {cnt}개')
        else:
            st.info("등록된 시설이 없습니다.")

# ===========================================================
# TAB 2 — AI 추천 배치 비교
# ===========================================================
with tab2:
    st.subheader("AI 추천 신규 인프라 배치안")

    style = FACILITY_STYLE[rec_location["suggested_type"]]
    st.markdown(
        f"""
        📍 **추천 결과:** 좌표 **({rec_location['x']}, {rec_location['y']})** [{rec_location['zone']}] 의
        부하율이 {badge_html(rec_location['overload_index'])} 로 가장 심각합니다.
        이 위치에 {style['icon']} **[{rec_location['suggested_type']}]** 신설을 추천합니다.
        """,
        unsafe_allow_html=True,
    )

    # 추천 위치와 같은 유형 기준으로 개선 전/후를 동일한 필터로 비교
    compare_type = rec_location["suggested_type"]
    before_df, _ = calculate_overload(city_df, st.session_state.facilities, compare_type)

    simulated_facs = st.session_state.facilities + [
        {
            "id": "sim-new",
            "name": "★ 추천 신규 시설",
            "type": compare_type,
            "x": rec_location["x"],
            "y": rec_location["y"],
            "capacity": 150,
            "radius": 3.0,
        }
    ]
    after_df, _ = calculate_overload(city_df, simulated_facs, compare_type)

    z_before = to_grid(before_df, "overload_index")
    z_after = to_grid(after_df, "overload_index")
    shared_max = max(z_before.max(), z_after.max(), 1)  # 두 지도 동일 스케일 사용

    col_before, col_after = st.columns(2)

    with col_before:
        st.markdown(f"#### 🔴 개선 전 ({style['icon']} {compare_type} 기준)")
        fig_before = px.imshow(
            z_before, origin="lower", color_continuous_scale="Reds", zmin=0, zmax=shared_max,
            labels=dict(x="X", y="Y", color="부하율(%)"),
        )
        fig_before.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_before, use_container_width=True)

    with col_after:
        st.markdown(f"#### 🟢 추천 시설 추가 후")
        fig_after = px.imshow(
            z_after, origin="lower", color_continuous_scale="Reds", zmin=0, zmax=shared_max,
            labels=dict(x="X", y="Y", color="부하율(%)"),
        )
        fig_after.add_scatter(
            x=[rec_location["x"]], y=[rec_location["y"]],
            mode="markers+text",
            marker=dict(symbol="star", size=20, color="gold", line=dict(width=1, color="black")),
            text=["★ 추천위치"], textposition="top center", showlegend=False,
        )
        fig_after.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_after, use_container_width=True)

    before_avg = before_df["overload_index"].mean()
    after_avg = after_df["overload_index"].mean()
    delta = before_avg - after_avg
    st.success(
        f"이 시설을 추가하면 **{compare_type}** 기준 평균 부하율이 "
        f"**{before_avg:.1f}% → {after_avg:.1f}%** (▼ {delta:.1f}%p) 로 완화될 것으로 예측됩니다."
    )

# ===========================================================
# TAB 3 — 상세 데이터
# ===========================================================
with tab3:
    st.subheader("등록된 시설 목록")
    if st.session_state.facilities:
        fac_table = pd.DataFrame(st.session_state.facilities)[["name", "type", "x", "y", "capacity", "radius"]]
        fac_table.columns = ["이름", "유형", "X", "Y", "용량", "반경"]
        st.dataframe(fac_table, use_container_width=True, hide_index=True)
    else:
        st.info("등록된 시설이 없습니다.")

    st.subheader(f"부하율 상위 10개 구역 (현재 필터: {facility_type_filter})")
    worst = analyzed_df.sort_values("overload_index", ascending=False).head(10).copy()
    worst["등급"] = worst["overload_index"].apply(lambda v: risk_level(v)[0])
    worst_display = worst[["x", "y", "zone", "demand", "supply", "overload_index", "등급"]]
    worst_display.columns = ["X", "Y", "구역", "수요", "공급", "부하율(%)", "등급"]
    st.dataframe(worst_display, use_container_width=True, hide_index=True)

    with st.expander("전체 격자 데이터 보기"):
        full_display = analyzed_df[["x", "y", "zone", "demand", "supply", "overload_index"]].copy()
        full_display.columns = ["X", "Y", "구역", "수요", "공급", "부하율(%)"]
        st.dataframe(full_display, use_container_width=True, hide_index=True)
