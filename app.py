import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="스마트 도시 인프라 과부화 & 최적 배치 시뮬레이터",
    page_icon="🏙️",
    layout="wide"
)

st.title("🏙️ 스마트 도시 인프라 과부화 분석 및 AI 추천 배치")
st.caption("도시 구역별 인프라 과부화 및 노후화/부족 문제를 시각화하고 최적의 시설 배치를 제안합니다.")

# ---------------------------------------------------------
# 2. 기본 도시 레이아웃 (10x10 Grid) 데이터 생성
# ---------------------------------------------------------
GRID_SIZE = 10

@st.cache_data
def generate_city_grid():
    """도시 기본 구역(주거, 상업, 공업, 녹지) 및 기본 수요 지수 생성"""
    np.random.seed(42)
    zones = ["주거 구역", "상업 구역", "공업 구역", "녹지 구역"]
    weights = [0.4, 0.3, 0.2, 0.1]
    
    grid_data = []
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            zone = np.random.choice(zones, p=weights)
            # 구역별 기본 인프라 수요량 설정
            if zone == "주거 구역":
                base_demand = np.random.randint(60, 95)
            elif zone == "상업 구역":
                base_demand = np.random.randint(70, 100)
            elif zone == "공업 구역":
                base_demand = np.random.randint(40, 80)
            else:  # 녹지
                base_demand = np.random.randint(10, 30)
                
            grid_data.append({
                "x": x,
                "y": y,
                "zone": zone,
                "demand": base_demand
            })
    return pd.DataFrame(grid_data)

city_df = generate_city_grid()

# ---------------------------------------------------------
# 3. 사이드바 - 인프라 시설 관리
# ---------------------------------------------------------
st.sidebar.header("⚙️ 인프라 시설 설정")

# 기본 초기 인프라 시설
if "facilities" not in st.session_state:
    st.session_state.facilities = [
        {"id": 1, "name": "중앙 경찰서", "type": "경찰서", "x": 2, "y": 3, "capacity": 150, "radius": 3.0},
        {"id": 2, "name": "동부 경찰서", "type": "경찰서", "x": 8, "y": 7, "capacity": 100, "radius": 2.5},
        {"id": 3, "name": "IT 테크노밸리", "type": "회사", "x": 5, "y": 5, "capacity": 200, "radius": 3.5},
        {"id": 4, "name": "시립 종합병원", "type": "병원", "x": 3, "y": 8, "capacity": 180, "radius": 4.0},
    ]

facility_type_filter = st.sidebar.selectbox(
    "분석할 인프라 유형 선택",
    ["전체", "경찰서", "회사", "병원"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("➕ 신규 인프라 직접 추가")
with st.sidebar.form("add_facility_form"):
    f_name = st.text_input("시설 이름", "신규 시설")
    f_type = st.selectbox("시설 유형", ["경찰서", "회사", "병원"])
    f_x = st.slider("X 좌표", 0, GRID_SIZE-1, 5)
    f_y = st.slider("Y 좌표", 0, GRID_SIZE-1, 5)
    f_cap = st.number_input("수용 용량", min_value=50, max_value=500, value=120)
    f_rad = st.slider("커버리지 반경", 1.0, 5.0, 3.0, step=0.5)
    
    submitted = st.form_submit_button("시설 추가")
    if submitted:
        new_id = len(st.session_state.facilities) + 1
        st.session_state.facilities.append({
            "id": new_id, "name": f_name, "type": f_type,
            "x": f_x, "y": f_y, "capacity": f_cap, "radius": f_rad
        })
        st.sidebar.success(f"'{f_name}' 시설이 추가되었습니다!")

# ---------------------------------------------------------
# 4. 과부화 계산 엔진 (Distance Decay Model)
# ---------------------------------------------------------
def calculate_overload(grid_df, facilities_list, selected_type="전체"):
    """
    각 Grid 좌표(x, y)에 대해 시설들의 공급량을 계산하고 과부화 지수를 산출합니다.
    - 공급량 = Sum( Capacity / (1 + (거리 / 반경)^2) )
    - 과부화 지수 (%) = (수요량 / 공급량) * 100
    """
    filtered_facs = [
        f for f in facilities_list 
        if selected_type == "전체" or f["type"] == selected_type
    ]
    
    supply_matrix = np.zeros((GRID_SIZE, GRID_SIZE))
    
    for fac in filtered_facs:
        fx, fy = fac["x"], fac["y"]
        cap, rad = fac["capacity"], fac["radius"]
        
        for gx in range(GRID_SIZE):
            for gy in range(GRID_SIZE):
                dist = np.sqrt((gx - fx)**2 + (gy - fy)**2)
                # 거리 감쇄 공식 적용
                supply = cap / (1.0 + (dist / rad)**2)
                supply_matrix[gy, gx] += supply

    # 지수 계산
    grid_analysis = grid_df.copy()
    overload_list = []
    supply_list = []
    
    for idx, row in grid_analysis.iterrows():
        gx, gy = int(row["x"]), int(row["y"])
        sup = supply_matrix[gy, gx]
        dem = row["demand"]
        
        supply_list.append(round(sup, 1))
        if sup == 0:
            overload = 200.0  # 공급이 완전 부재할 경우 높은 과부화값 부여
        else:
            overload = (dem / sup) * 100.0
        overload_list.append(round(overload, 1))
        
    grid_analysis["supply"] = supply_list
    grid_analysis["overload_index"] = overload_list
    return grid_analysis, filtered_facs

analyzed_df, active_facs = calculate_overload(city_df, st.session_state.facilities, facility_type_filter)

# ---------------------------------------------------------
# 5. 최적 추천 배치 알고리즘
# ---------------------------------------------------------
def recommend_best_location(analyzed_df, facility_type):
    """과부화 지수(Overload)가 가장 높고 인프라가 부족한 지점을 찾아 최적 배치 장소 추천"""
    # 과부화 지수가 높을수록 인프라 공급 부족
    sorted_df = analyzed_df.sort_values(by="overload_index", ascending=False)
    top_spot = sorted_df.iloc[0]
    
    return {
        "x": int(top_spot["x"]),
        "y": int(top_spot["y"]),
        "zone": top_spot["zone"],
        "overload_index": top_spot["overload_index"],
        "suggested_type": "경찰서" if facility_type == "전체" else facility_type
    }

rec_location = recommend_best_location(analyzed_df, facility_type_filter)

# ---------------------------------------------------------
# 6. 메인 UI 대시보드
# ---------------------------------------------------------
# Key Metrics
m1, m2, m3, m4 = st.columns(4)
avg_overload = analyzed_df["overload_index"].mean()
max_overload = analyzed_df["overload_index"].max()
high_risk_count = (analyzed_df["overload_index"] > 120).sum()

m1.metric("도시 평균 부하율", f"{avg_overload:.1f}%")
m2.metric("최대 과부화 구역 부하율", f"{max_overload:.1f}%", delta_color="inverse")
m3.metric("고위험(부족) 구역 수", f"{high_risk_count} 개 구역")
m4.metric("추천 신규 배치 좌표", f"({rec_location['x']}, {rec_location['y']})")

st.markdown("---")

tab1, tab2 = st.tabs(["📊 과부화 분석 지도", "💡 AI 추천 배치도 비교"])

with tab1:
    col_map, col_info = st.columns([7, 3])
    
    with col_map:
        st.subheader("도시 격자별 인프라 과부화 히트맵")
        
        # Plotly Heatmap 생성
        z_data = np.zeros((GRID_SIZE, GRID_SIZE))
        for _, row in analyzed_df.iterrows():
            z_data[int(row["y"]), int(row["x"])] = row["overload_index"]
            
        fig = go.Figure()
        
        # 히트맵 레이어
        fig.add_trace(go.Heatmap(
            z=z_data,
            x=list(range(GRID_SIZE)),
            y=list(range(GRID_SIZE)),
            colorscale="Reds",
            colorbar_title="부하율 (%)",
            hovertemplate="X: %{x}<br>Y: %{y}<br>과부화지수: %{z:.1f}%<extra></extra>"
        ))
        
        # 기존 시설 위치 마커
        if active_facs:
            fac_x = [f["x"] for f in active_facs]
            fac_y = [f["y"] for f in active_facs]
            fac_text = [f"{f['name']} ({f['type']})" for f in active_facs]
            
            fig.add_trace(go.Scatter(
                x=fac_x, y=fac_y,
                mode="markers+text",
                marker=dict(symbol="square", size=14, color="blue", line=dict(width=2, color="white")),
                text=[f["name"] for f in active_facs],
                textposition="top center",
                name="현재 인프라 시설",
                hoverinfo="text",
                hovertext=fac_text
            ))

        fig.update_layout(
            xaxis=dict(title="X 좌표 (동-서)", dtick=1),
            yaxis=dict(title="Y 좌표 (남-북)", dtick=1),
            height=550,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.subheader("등록된 시설 목록")
        fac_df = pd.DataFrame(st.session_state.facilities)
        st.dataframe(fac_df[["id", "name", "type", "x", "y", "capacity"]], use_container_width=True, height=250)
        
        st.subheader("구역별 평균 과부화")
        zone_summary = analyzed_df.groupby("zone")["overload_index"].mean().reset_index()
        fig_bar = px.bar(zone_summary, x="zone", y="overload_index", color="zone", title="구역 유형별 과부화 비율")
        fig_bar.update_layout(showlegend=False, height=250)
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.subheader("AI 추천 신규 인프라 설치 배치안")
    st.info(
        f"📍 **추천 결과:** ({rec_location['x']}, {rec_location['y']}) 지점[{rec_location['zone']}]의 "
        f"과부화율이 **{rec_location['overload_index']:.1f}%**로 가장 심각합니다. "
        f"이 지역에 **[{rec_location['suggested_type']}]** 신설을 추천합니다."
    )
    
    # 추천 배치가 적용되었을 때의 예측 계산
    simulated_facs = st.session_state.facilities.copy()
    simulated_facs.append({
        "id": 999, "name": "★ 추천 신규 시설",
        "type": rec_location['suggested_type'],
        "x": rec_location['x'], "y": rec_location['y'],
        "capacity": 150, "radius": 3.0
    })
    
    sim_df, _ = calculate_overload(city_df, simulated_facs, facility_type_filter)
    
    col_before, col_after = st.columns(2)
    
    with col_before:
        st.write("#### 🔴 현재 상태 (개선 전)")
        fig_before = px.imshow(
            z_data, labels=dict(x="X", y="Y", color="부하율(%)"),
            x=list(range(GRID_SIZE)), y=list(range(GRID_SIZE)),
            color_continuous_scale="Reds", origin="lower"
        )
        st.plotly_chart(fig_before, use_container_width=True)
        
    with col_after:
        st.write("#### 🟢 추천 시설 추가 후 (개선 예측)")
        z_sim_data = np.zeros((GRID_SIZE, GRID_SIZE))
        for _, row in sim_df.iterrows():
            z_sim_data[int(row["y"]), int(row["x"])] = row["overload_index"]
            
        fig_after = px.imshow(
            z_sim_data, labels=dict(x="X", y="Y", color="부하율(%)"),
            x=list(range(GRID_SIZE)), y=list(range(GRID_SIZE)),
            color_continuous_scale="Reds", origin="lower"
        )
        # 추천 위치 표시
        fig_after.add_scatter(
            x=[rec_location['x']], y=[rec_location['y']],
            mode="markers+text", marker=dict(symbol="star", size=18, color="gold"),
            text=["★ 추천위치"], textposition="top center"
        )
        st.plotly_chart(fig_after, use_container_width=True)

    st.success(f"시설 추가 시 도시 평균 과부화율이 **{avg_overload:.1f}% ➡️ {sim_df['overload_index'].mean():.1f}%** 로 완화됩니다.")
