# requirements: streamlit>=1.37, plotly>=5.18, pandas, numpy
# 실행: streamlit run smart_city_dashboard.py
#
# 주의: 지도를 "클릭"해서 시설을 설치하는 기능은 streamlit의 on_select 이벤트를 사용합니다.
# streamlit 버전이 낮으면(<=1.36) 클릭 설치가 동작하지 않을 수 있으니,
# 그 경우 사이드바의 "좌표 직접 입력" 방식을 이용해 주세요.

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
GRADE_ORDER = ["양호", "주의", "위험", "심각"]

FACILITY_STYLE = {
    "경찰서": {"color": "#3b82f6", "icon": "🚓"},
    "병원":   {"color": "#ef4444", "icon": "🏥"},
    "회사":   {"color": "#f59e0b", "icon": "🏢"},
}

# 눈이 편안한 부드러운 그라데이션 (진한 빨강 대신 연노랑 -> 연주황 -> 톤 다운된 빨강)
SOFT_SCALE = [
    [0.0, "#f8fafc"],
    [0.30, "#fef3c7"],
    [0.60, "#fdba74"],
    [1.0, "#e07a5f"],
]

DEFAULT_FACILITIES = [
    {"id": "f1", "name": "중앙 경찰서",   "type": "경찰서", "x": 2, "y": 3, "capacity": 150, "radius": 3.0},
    {"id": "f2", "name": "동부 경찰서",   "type": "경찰서", "x": 8, "y": 7, "capacity": 100, "radius": 2.5},
    {"id": "f3", "name": "IT 테크노밸리", "type": "회사",   "x": 5, "y": 5, "capacity": 200, "radius": 3.5},
    {"id": "f4", "name": "시립 종합병원", "type": "병원",   "x": 3, "y": 8, "capacity": 180, "radius": 4.0},
]

st.set_page_config(page_title="스마트 도시 인프라 시뮬레이터", page_icon="🏙️", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #eef0f3; border-radius: 12px;
        padding: 12px 16px 8px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    [data-testid="stMetricLabel"] { font-size: 0.82rem; color: #6b7280; }
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; color: white;
    }
    .cause-box {
        background: #fafafa; border-left: 3px solid #e07a5f;
        padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; font-size: 0.92rem;
    }
    .facility-row {
        display: flex; align-items: center; gap: 6px; padding: 5px 8px;
        border-radius: 8px; border: 1px solid #f0f0f0; margin-bottom: 5px; font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏙️ 스마트 도시 인프라 과부화 분석 · AI 추천 배치")
st.caption("지도를 클릭해 신규 시설을 설치하고, 과부화 원인과 해결 방안을 자동으로 진단합니다.")


def risk_level(value: float) -> tuple[str, str]:
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
# 1. 기본 도시 격자 생성
# ===========================================================
@st.cache_data
def generate_city_grid() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    weights = [0.4, 0.3, 0.2, 0.1]
    demand_ranges = {
        "주거 구역": (60, 95), "상업 구역": (70, 100),
        "공업 구역": (40, 80), "녹지 구역": (10, 30),
    }
    rows = []
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            zone = rng.choice(ZONE_TYPES, p=weights)
            lo, hi = demand_ranges[zone]
            rows.append({"x": x, "y": y, "zone": zone, "demand": int(rng.integers(lo, hi + 1))})
    return pd.DataFrame(rows)


city_df = generate_city_grid()
CITY_AVG_DEMAND = float(city_df["demand"].mean())

# ===========================================================
# 2. 세션 상태
# ===========================================================
if "facilities" not in st.session_state:
    st.session_state.facilities = [dict(f) for f in DEFAULT_FACILITIES]
if "pending_click" not in st.session_state:
    st.session_state.pending_click = None
if "_id_counter" not in st.session_state:
    st.session_state._id_counter = 1000


def next_id() -> str:
    st.session_state._id_counter += 1
    return f"f{st.session_state._id_counter}"


def add_facility(name, ftype, x, y, cap, rad):
    st.session_state.facilities.append(
        {"id": next_id(), "name": name, "type": ftype, "x": int(x), "y": int(y),
         "capacity": int(cap), "radius": float(rad)}
    )


def remove_facilities(ids_to_remove: set):
    st.session_state.facilities = [f for f in st.session_state.facilities if f["id"] not in ids_to_remove]


def reset_facilities():
    st.session_state.facilities = [dict(f) for f in DEFAULT_FACILITIES]
    st.session_state.pending_click = None


# ===========================================================
# 3. 계산 엔진
# ===========================================================
def calculate_overload(grid_df: pd.DataFrame, facilities: list, selected_type: str = "전체"):
    filtered = [f for f in facilities if selected_type == "전체" or f["type"] == selected_type]
    xs, ys = np.arange(GRID_SIZE), np.arange(GRID_SIZE)
    gx, gy = np.meshgrid(xs, ys)

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
    candidate_types = FACILITY_TYPES if selected_type == "전체" else [selected_type]
    best = None
    for t in candidate_types:
        analyzed, _ = calculate_overload(grid_df, facilities, t)
        top = analyzed.loc[analyzed["overload_index"].idxmax()]
        candidate = {
            "x": int(top["x"]), "y": int(top["y"]), "zone": top["zone"],
            "overload_index": float(top["overload_index"]), "suggested_type": t,
        }
        if best is None or candidate["overload_index"] > best["overload_index"]:
            best = candidate
    return best


def best_type_for_cell(grid_df: pd.DataFrame, facilities: list, x: int, y: int) -> str:
    best_t, best_v = FACILITY_TYPES[0], -1
    for t in FACILITY_TYPES:
        analyzed, _ = calculate_overload(grid_df, facilities, t)
        row = analyzed[(analyzed.x == x) & (analyzed.y == y)].iloc[0]
        if row["overload_index"] > best_v:
            best_v, best_t = row["overload_index"], t
    return best_t


# ===========================================================
# 4. 진단(원인 분석 + 해결 방안) 엔진
# ===========================================================
def generate_diagnosis(grid_df: pd.DataFrame, facilities: list, x: int, y: int, ftype: str) -> dict:
    analyzed, filtered = calculate_overload(grid_df, facilities, ftype)
    row = analyzed[(analyzed.x == x) & (analyzed.y == y)].iloc[0]
    demand, supply, overload, zone = row["demand"], row["supply"], row["overload_index"], row["zone"]

    causes = []
    if not filtered:
        causes.append(f"도시 내에 **{ftype}** 유형의 시설이 하나도 없습니다. 이 유형에 대한 공급이 전무한 상태입니다.")
    else:
        dists = [(f, float(np.hypot(f["x"] - x, f["y"] - y))) for f in filtered]
        nearest, nd = min(dists, key=lambda p: p[1])
        if nd > nearest["radius"] * 1.3:
            causes.append(
                f"가장 가까운 {ftype}인 **'{nearest['name']}'**까지 거리가 {nd:.1f}칸으로, "
                f"유효 서비스 반경({nearest['radius']:.1f}칸)을 크게 벗어나 사실상 사각지대입니다."
            )
        elif nd > nearest["radius"]:
            causes.append(
                f"가장 가까운 {ftype} **'{nearest['name']}'**가 서비스 반경 경계 부근"
                f"({nd:.1f} / {nearest['radius']:.1f}칸)에 있어 공급 효율이 크게 떨어집니다."
            )

    if demand >= CITY_AVG_DEMAND * 1.15:
        pct = (demand / CITY_AVG_DEMAND - 1) * 100
        causes.append(
            f"이 구역({zone})의 기본 수요는 {demand:.0f}로, 도시 평균({CITY_AVG_DEMAND:.0f}) 대비 "
            f"**{pct:.0f}% 높은 고밀도 지역**입니다."
        )

    if supply > 0 and overload >= 100:
        causes.append(f"주변 시설의 유효 공급량({supply:.0f})이 수요({demand:.0f})를 따라가지 못해 **공급 부족**이 발생했습니다.")

    if not causes:
        causes.append("현재 이 구역은 뚜렷한 과부화 원인이 발견되지 않았습니다. 대체로 양호한 상태입니다.")

    # 해결 방안 시뮬레이션: 이 위치에 신규 시설을 놓았을 때 예측
    existing_of_type = [f for f in filtered]
    suggested_capacity = int(np.mean([f["capacity"] for f in existing_of_type])) if existing_of_type else 150
    suggested_radius = round(float(np.mean([f["radius"] for f in existing_of_type])), 1) if existing_of_type else 3.0

    sim_facs = facilities + [{
        "id": "sim-preview", "name": f"신규 {ftype}", "type": ftype,
        "x": x, "y": y, "capacity": suggested_capacity, "radius": suggested_radius,
    }]
    sim_analyzed, _ = calculate_overload(grid_df, sim_facs, ftype)
    after_val = float(sim_analyzed[(sim_analyzed.x == x) & (sim_analyzed.y == y)].iloc[0]["overload_index"])

    if overload >= 70:
        solution_text = (
            f"이 위치에 용량 {suggested_capacity}, 반경 {suggested_radius}칸 수준의 신규 **{ftype}**를 배치하면 "
            f"예상 부하율이 **{overload:.0f}% → {after_val:.0f}%** 로 낮아질 것으로 예측됩니다."
        )
    else:
        solution_text = "현재 부하율이 양호한 수준이라 즉시 신규 시설 배치가 시급하지는 않습니다. 다른 고위험 지역을 먼저 검토해 보세요."

    return {
        "demand": demand, "supply": supply, "overload": overload, "zone": zone,
        "causes": causes, "solution_text": solution_text,
        "suggested_capacity": suggested_capacity, "suggested_radius": suggested_radius,
        "predicted_after": after_val,
    }


def get_top_hotspots(analyzed_df: pd.DataFrame, exclude_xy: tuple, n: int = 3) -> list:
    df = analyzed_df[~((analyzed_df.x == exclude_xy[0]) & (analyzed_df.y == exclude_xy[1]))]
    df = df.sort_values("overload_index", ascending=False).head(n)
    return df.to_dict("records")


# ===========================================================
# 5. 사이드바
# ===========================================================
st.sidebar.header("⚙️ 설정")

facility_type_filter = st.sidebar.selectbox("분석할 인프라 유형", ["전체"] + FACILITY_TYPES)

st.sidebar.markdown("**🔥 부하율 표시**")
show_heatmap = st.sidebar.toggle("히트맵 켜기", value=True)
selected_grades = st.sidebar.multiselect(
    "표시할 위험 등급만 필터", GRADE_ORDER, default=GRADE_ORDER,
    help="선택하지 않은 등급의 셀은 지도에서 흐리게 숨겨집니다.", disabled=not show_heatmap,
)

st.sidebar.markdown("---")
st.sidebar.subheader("📋 등록된 시설")
if not st.session_state.facilities:
    st.sidebar.info("등록된 시설이 없습니다.")
else:
    for f in st.session_state.facilities:
        s = FACILITY_STYLE[f["type"]]
        st.sidebar.markdown(
            f'<div class="facility-row">{s["icon"]} <b>{f["name"]}</b>'
            f'<span style="color:#9ca3af">&nbsp;({f["x"]},{f["y"]} · 용량{f["capacity"]})</span></div>',
            unsafe_allow_html=True,
        )
    del_map = {f'{FACILITY_STYLE[f["type"]]["icon"]} {f["name"]}': f["id"] for f in st.session_state.facilities}
    to_del = st.sidebar.multiselect("삭제할 시설", list(del_map.keys()))
    c1, c2 = st.sidebar.columns(2)
    if c1.button("🗑️ 삭제", use_container_width=True, disabled=not to_del):
        remove_facilities({del_map[k] for k in to_del})
        st.rerun()
    if c2.button("↺ 초기화", use_container_width=True):
        reset_facilities()
        st.rerun()

st.sidebar.markdown("---")
with st.sidebar.expander("🔧 좌표 직접 입력으로 설치 (클릭이 안 될 때)"):
    with st.form("manual_add_form", clear_on_submit=True):
        m_name = st.text_input("시설 이름", "")
        m_type = st.selectbox("시설 유형", FACILITY_TYPES, key="manual_type")
        mc1, mc2 = st.columns(2)
        m_x = mc1.number_input("X", 0, GRID_SIZE - 1, 5)
        m_y = mc2.number_input("Y", 0, GRID_SIZE - 1, 5)
        m_cap = st.number_input("수용 용량", 50, 500, 120, step=10)
        m_rad = st.slider("커버리지 반경", 1.0, 5.0, 3.0, step=0.5)
        if st.form_submit_button("설치", use_container_width=True):
            if not m_name.strip():
                st.error("시설 이름을 입력해 주세요.")
            else:
                add_facility(m_name.strip(), m_type, m_x, m_y, m_cap, m_rad)
                st.success(f"'{m_name}' 설치 완료!")
                st.rerun()

# ===========================================================
# 6. 계산 실행
# ===========================================================
analyzed_df, active_facs = calculate_overload(city_df, st.session_state.facilities, facility_type_filter)
rec_location = recommend_best_location(city_df, st.session_state.facilities, facility_type_filter)

avg_overload = analyzed_df["overload_index"].mean()
max_overload = analyzed_df["overload_index"].max()
high_risk_count = int((analyzed_df["overload_index"] >= 100).sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("도시 평균 부하율", f"{avg_overload:.1f}%")
m2.metric("최대 부하 구역", f"{max_overload:.1f}%")
m3.metric("위험 이상 구역 수", f"{high_risk_count} / {GRID_SIZE * GRID_SIZE} 개")
m4.metric("AI 추천 신설 위치", f"({rec_location['x']}, {rec_location['y']})",
          f"{FACILITY_STYLE[rec_location['suggested_type']]['icon']} {rec_location['suggested_type']}")

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["🗺️ 지도 · 클릭 설치", "🤖 AI 진단 & 추천", "📋 상세 데이터"])


def to_grid(df: pd.DataFrame, col: str) -> np.ndarray:
    z = np.zeros((GRID_SIZE, GRID_SIZE))
    z[df["y"].to_numpy(), df["x"].to_numpy()] = df[col].to_numpy()
    return z


def build_map_figure(z_data, active_facs, show_heatmap, selected_grades, pending_click):
    fig = go.Figure()

    if show_heatmap:
        vgrade = np.vectorize(lambda v: risk_level(v)[0])
        grades_grid = vgrade(z_data)
        mask = np.isin(grades_grid, selected_grades) if selected_grades else np.zeros_like(z_data, dtype=bool)
        z_display = np.where(mask, z_data, np.nan)
        zmax = max(120.0, float(np.nanpercentile(z_data, 95)) if z_data.size else 120.0)
        fig.add_trace(go.Heatmap(
            z=z_display, x=list(range(GRID_SIZE)), y=list(range(GRID_SIZE)),
            colorscale=SOFT_SCALE, zmin=0, zmax=zmax,
            colorbar=dict(title="부하율(%)"),
            hovertemplate="X:%{x} Y:%{y}<br>부하율 %{z:.0f}%<extra></extra>",
            xgap=2, ygap=2,
        ))
    else:
        # 눈이 편한 옅은 배경만 표시 (시설 배치에 집중)
        fig.add_trace(go.Heatmap(
            z=np.zeros_like(z_data), x=list(range(GRID_SIZE)), y=list(range(GRID_SIZE)),
            colorscale=[[0, "#f8fafc"], [1, "#f8fafc"]], showscale=False,
            hoverinfo="skip", xgap=2, ygap=2,
        ))

    for ftype in FACILITY_TYPES:
        subset = [f for f in active_facs if f["type"] == ftype]
        if not subset:
            continue
        style = FACILITY_STYLE[ftype]
        fig.add_trace(go.Scatter(
            x=[f["x"] for f in subset], y=[f["y"] for f in subset],
            mode="markers+text",
            marker=dict(symbol="square", size=15, color=style["color"], line=dict(width=1.5, color="white")),
            text=[f["name"] for f in subset], textposition="top center",
            name=f'{style["icon"]} {ftype}',
            hovertext=[f"{f['name']} · 용량 {f['capacity']} · 반경 {f['radius']}" for f in subset],
            hoverinfo="text",
        ))

    if pending_click is not None:
        fig.add_trace(go.Scatter(
            x=[pending_click[0]], y=[pending_click[1]], mode="markers",
            marker=dict(symbol="circle-open", size=26, color="#111827", line=dict(width=3)),
            name="선택된 위치", showlegend=False, hoverinfo="skip",
        ))

    fig.update_layout(
        xaxis=dict(title="X 좌표 (동-서)", dtick=1, showgrid=False),
        yaxis=dict(title="Y 좌표 (남-북)", dtick=1, showgrid=False),
        plot_bgcolor="white", paper_bgcolor="white",
        height=560, margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ===========================================================
# TAB 1 — 지도 & 클릭 설치
# ===========================================================
with tab1:
    col_map, col_panel = st.columns([7, 3])

    with col_map:
        z_data = to_grid(analyzed_df, "overload_index")
        fig = build_map_figure(z_data, active_facs, show_heatmap, selected_grades, st.session_state.pending_click)

        event = None
        try:
            event = st.plotly_chart(
                fig, use_container_width=True, on_select="rerun",
                selection_mode="points", key="city_map",
            )
        except TypeError:
            # 구버전 streamlit: on_select 미지원 → 일반 차트로 표시
            st.plotly_chart(fig, use_container_width=True)
            st.caption("⚠️ 현재 streamlit 버전에서는 지도 클릭 설치가 지원되지 않습니다. 사이드바의 좌표 직접 입력을 이용해 주세요.")

        if event:
            points = (event.get("selection") or {}).get("points") or []
            if points:
                px_ = int(round(points[0]["x"]))
                py_ = int(round(points[0]["y"]))
                px_ = min(max(px_, 0), GRID_SIZE - 1)
                py_ = min(max(py_, 0), GRID_SIZE - 1)
                st.session_state.pending_click = (px_, py_)

    with col_panel:
        st.subheader("📍 구역 분석 & 설치")
        pc = st.session_state.pending_click

        if pc is None:
            st.info("지도의 칸을 클릭하면 해당 구역의 분석 결과와 설치 옵션이 여기에 나타납니다.")
        else:
            x, y = pc
            default_type = facility_type_filter if facility_type_filter != "전체" else \
                best_type_for_cell(city_df, st.session_state.facilities, x, y)
            diag = generate_diagnosis(city_df, st.session_state.facilities, x, y, default_type)

            st.markdown(f"**선택 위치:** ({x}, {y}) · {diag['zone']}")
            st.markdown(badge_html(diag["overload"]), unsafe_allow_html=True)
            st.caption(f"수요 {diag['demand']:.0f} · 공급 {diag['supply']:.0f}")

            with st.expander("🔍 원인 분석 보기", expanded=False):
                for c in diag["causes"]:
                    st.markdown(f'<div class="cause-box">{c}</div>', unsafe_allow_html=True)

            st.markdown("##### 🏗️ 이 위치에 시설 설치")
            with st.form("click_install_form"):
                i_name = st.text_input("시설 이름", f"신규 {default_type}")
                i_type = st.selectbox("시설 유형", FACILITY_TYPES, index=FACILITY_TYPES.index(default_type))
                i_cap = st.number_input("수용 용량", 50, 500, diag["suggested_capacity"], step=10)
                i_rad = st.slider("커버리지 반경", 1.0, 5.0, float(diag["suggested_radius"]), step=0.5)
                if st.form_submit_button("✅ 이 위치에 설치", use_container_width=True):
                    add_facility(i_name.strip() or f"신규 {i_type}", i_type, x, y, i_cap, i_rad)
                    st.success(f"({x},{y})에 '{i_name}' 설치 완료!")
                    st.rerun()

# ===========================================================
# TAB 2 — AI 진단 & 추천
# ===========================================================
with tab2:
    st.subheader("🤖 AI 추천 위치 진단")

    diag = generate_diagnosis(
        city_df, st.session_state.facilities, rec_location["x"], rec_location["y"], rec_location["suggested_type"]
    )
    style = FACILITY_STYLE[rec_location["suggested_type"]]

    st.markdown(f"#### 📍 ({rec_location['x']}, {rec_location['y']}) · {rec_location['zone']}")
    st.markdown(badge_html(rec_location["overload_index"]), unsafe_allow_html=True)
    st.write("")

    st.markdown("**🔍 왜 이 지역의 부하율이 높을까요?**")
    for c in diag["causes"]:
        st.markdown(f'<div class="cause-box">{c}</div>', unsafe_allow_html=True)

    st.markdown("**💡 해결 방안**")
    st.info(f'{style["icon"]} {diag["solution_text"]}')

    st.markdown("---")
    compare_type = rec_location["suggested_type"]
    before_df, _ = calculate_overload(city_df, st.session_state.facilities, compare_type)
    simulated_facs = st.session_state.facilities + [{
        "id": "sim-new", "name": "★ 추천 신규 시설", "type": compare_type,
        "x": rec_location["x"], "y": rec_location["y"],
        "capacity": diag["suggested_capacity"], "radius": diag["suggested_radius"],
    }]
    after_df, _ = calculate_overload(city_df, simulated_facs, compare_type)
    z_before, z_after = to_grid(before_df, "overload_index"), to_grid(after_df, "overload_index")
    shared_max = max(float(z_before.max()), float(z_after.max()), 1.0)

    col_before, col_after = st.columns(2)
    with col_before:
        st.markdown(f"##### 🔴 개선 전 ({style['icon']} {compare_type} 기준)")
        fb = px.imshow(z_before, origin="lower", color_continuous_scale=SOFT_SCALE, zmin=0, zmax=shared_max,
                        labels=dict(x="X", y="Y", color="부하율(%)"))
        fb.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fb, use_container_width=True)
    with col_after:
        st.markdown("##### 🟢 추천 시설 추가 후")
        fa = px.imshow(z_after, origin="lower", color_continuous_scale=SOFT_SCALE, zmin=0, zmax=shared_max,
                        labels=dict(x="X", y="Y", color="부하율(%)"))
        fa.add_scatter(x=[rec_location["x"]], y=[rec_location["y"]], mode="markers+text",
                        marker=dict(symbol="star", size=18, color="#f59e0b", line=dict(width=1, color="#78350f")),
                        text=["★ 추천위치"], textposition="top center", showlegend=False)
        fa.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fa, use_container_width=True)

    before_avg, after_avg = before_df["overload_index"].mean(), after_df["overload_index"].mean()
    st.success(
        f"{compare_type} 기준 평균 부하율이 **{before_avg:.1f}% → {after_avg:.1f}%** "
        f"(▼ {before_avg - after_avg:.1f}%p) 로 완화될 것으로 예측됩니다."
    )

    st.markdown("---")
    st.markdown("**⚠️ 그 외 주의가 필요한 지역 Top 3**")
    hotspots = get_top_hotspots(before_df, (rec_location["x"], rec_location["y"]), n=3)
    if hotspots:
        for h in hotspots:
            hd = generate_diagnosis(city_df, st.session_state.facilities, int(h["x"]), int(h["y"]), compare_type)
            st.markdown(
                f'- **({int(h["x"])},{int(h["y"])})** {h["zone"]} — {badge_html(h["overload_index"])}'
                f'<br><span style="color:#6b7280;font-size:0.85rem">{hd["causes"][0]}</span>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("추가로 주의가 필요한 지역이 없습니다.")

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
