# -*- coding: utf-8 -*-
"""
🏙️ 실제 도시 인프라 과부하 분석 시뮬레이터 (Google Maps 기반)

실행 방법
---------
1) 패키지 설치
   pip install streamlit streamlit-folium folium googlemaps python-dotenv numpy pandas plotly
2) API 키 설정 (둘 중 하나)
   - 프로젝트 폴더의 .env 파일:        GOOGLE_MAPS_API_KEY=AIza...
   - .streamlit/secrets.toml 파일:     GOOGLE_MAPS_API_KEY = "AIza..."
3) 실행
   streamlit run app.py

필요한 Google Cloud API: Geocoding API, Places API (places_nearby = Legacy Places API)

구조
----
 A. 상수 · 분석 엔진 (Streamlit 비의존: Haversine, 커버리지 감쇠, 구역별 부하율)
 B. 페이지 설정 · API 키 로드 · Google API 래퍼 (캐싱 적용)
 C. 세션 상태 · 콜백 (시설 추가/삭제/수정/검색)
 D. 사이드바 UI
 E. 지도/차트 생성 함수
 F. 메인 대시보드
"""
from __future__ import annotations

import os
import time
import uuid
from html import escape

import folium
import googlemaps
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

# ===========================================================
# A. 상수 · 분석 엔진 (UI와 독립적인 순수 계산 코드)
# ===========================================================
EARTH_RADIUS_KM = 6371.0088          # 지구 평균 반지름 (km)
KM_PER_DEG_LAT = 111.0               # 위도 1도 ≈ 111 km
KM_PER_DEG_LNG_EQ = 111.32           # 적도 기준 경도 1도 ≈ 111.32 km (실제는 cos(위도)를 곱함)
EPS = 1e-9                           # 0 나눗셈 방지용
OI_CAP = 5.0                         # 부하율 표시 상한 (공급 0 → 상한값으로 처리)

# 시설 유형별 기본 설정: Google Places type, 지도 색상, 기본 용량/반경, 기본 수요율
FACILITY_TYPES = {
    "경찰서": dict(place_type="police", color="blue", hex="#1f77b4", icon="flag",
                 capacity=150, radius_km=2.5, demand_rate=0.04),
    "병원": dict(place_type="hospital", color="red", hex="#d62728", icon="plus-sign",
               capacity=200, radius_km=3.0, demand_rate=0.08),
    "소방서": dict(place_type="fire_station", color="orange", hex="#ff7f0e", icon="fire",
                capacity=120, radius_km=3.0, demand_rate=0.03),
}
DECAY_MODES = ["선형 감쇠", "가우시안 감쇠", "균등 (반경 내 전량)"]
STATUS_COLORS = {"여유": "#2ca02c", "보통": "#f2c744", "과부하": "#ff7f0e", "심각": "#d62728"}
FAC_COLUMNS = ["id", "name", "type", "lat", "lng", "capacity", "radius_km", "source"]


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine 공식: 두 위·경도 좌표 사이의 지표면 거리(km). 스칼라/numpy 배열 모두 지원."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def coverage_weight(dist_km, radius_km, mode):
    """시설과의 거리에 따른 서비스 가중치(0~1). 커버리지 반경 밖이면 0."""
    x = np.asarray(dist_km) / max(float(radius_km), 1e-6)
    if mode == DECAY_MODES[0]:
        w = 1.0 - x                       # 선형: 시설 위치 1.0 → 반경 경계 0.0
    elif mode == DECAY_MODES[1]:
        w = np.exp(-3.0 * x ** 2)         # 가우시안: 경계에서 약 0.05
    else:
        w = np.ones_like(x)               # 균등: 반경 안이면 전량 제공
    return np.where(x <= 1.0, w, 0.0)


def classify_array(oi, over, crit):
    """부하율 → 상태 라벨. 여유 < 0.7×과부하기준 ≤ 보통 < 과부하기준 ≤ 과부하 < 심각기준 ≤ 심각"""
    oi = np.asarray(oi, dtype=float)
    return np.select([oi >= crit, oi >= over, oi >= 0.7 * over], ["심각", "과부하", "보통"], default="여유")


def make_facility(ftype, name, lat, lng, capacity=None, radius_km=None, source="user", fid=None):
    """시설 레코드(dict) 생성. 용량/반경을 생략하면 유형별 기본값을 사용."""
    cfg = FACILITY_TYPES[ftype]
    return {
        "id": fid or f"user_{uuid.uuid4().hex[:8]}",
        "name": name, "type": ftype,
        "lat": float(lat), "lng": float(lng),
        "capacity": int(capacity if capacity is not None else cfg["capacity"]),
        "radius_km": float(radius_km if radius_km is not None else cfg["radius_km"]),
        "source": source,
    }


def facilities_to_df(facs):
    """시설 dict 리스트 → DataFrame (비어 있어도 컬럼 유지)."""
    return pd.DataFrame(facs, columns=FAC_COLUMNS) if facs else pd.DataFrame(columns=FAC_COLUMNS)


def analyze_zones(fac_df, center, p):
    """
    분석 반경을 덮는 N×N 격자 구역별로 수요·공급·부하율(Overload Index)을 계산한다.

      수요(구역, 유형)  = 추정인구 × 유형별 수요율
      공급(구역, 유형)  = Σ 시설 용량 × 용량배율 × 거리감쇠가중치(Haversine 거리 / 커버리지 반경)
      부하율 OI         = 수요 / 공급   (공급 0이면 OI_CAP, 1.0 초과면 수요가 공급을 넘는 상태)
      종합 부하율        = 선택한 유형들의 OI 평균 또는 최댓값
    ※ 실제 인구 통계가 없으므로 추정인구는 '도심 집중 + 잡음' 시뮬레이션 모델이다.
    """
    lat0, lng0, radius_km = center["lat"], center["lng"], center["radius_km"]
    n = int(p["grid_n"])

    # (1) 격자 생성: 중심에서 ±radius_km 범위를 n등분한 각 구역의 중심 좌표
    step_km = 2.0 * radius_km / n
    offsets_km = (np.arange(n) + 0.5) * step_km - radius_km
    km_per_deg_lng = KM_PER_DEG_LNG_EQ * np.cos(np.radians(lat0))
    lats = lat0 + offsets_km / KM_PER_DEG_LAT
    lngs = lng0 + offsets_km / km_per_deg_lng
    LAT, LNG = np.meshgrid(lats, lngs, indexing="ij")          # shape (n, n): [행=위도, 열=경도]
    half_lat = step_km / 2.0 / KM_PER_DEG_LAT
    half_lng = step_km / 2.0 / km_per_deg_lng

    # (2) 추정 인구: 도심으로 갈수록 증가(가우시안) × 로그정규 잡음(시드 고정 → 재현 가능)
    dist_c = haversine_km(lat0, lng0, LAT, LNG)
    in_area = dist_c <= radius_km                              # 원형 분석 반경 밖 구역은 제외
    sigma = max(radius_km * 0.5, 0.1)
    if p["noise"] > 0:
        noise = np.random.default_rng(int(p["seed"])).lognormal(0.0, p["noise"], size=(n, n))
    else:
        noise = np.ones((n, n))
    population = p["base_pop"] * (1.0 + p["conc"] * np.exp(-(dist_c / sigma) ** 2)) * noise

    # (3) 유형별 수요·공급·부하율
    per_type = {}
    for t in p["active"]:
        demand = population * p["rates"][t]
        supply = np.zeros((n, n))
        scale = p["mult"].get(t, 100) / 100.0
        for f in fac_df[fac_df["type"] == t].itertuples(index=False):
            d = haversine_km(f.lat, f.lng, LAT, LNG)           # 시설 → 모든 구역 거리(km)
            supply += f.capacity * scale * coverage_weight(d, f.radius_km, p["decay"])
        oi = np.minimum(demand / np.maximum(supply, EPS), OI_CAP)
        per_type[t] = (demand, supply, oi)

    # (4) 종합 지표 및 사각지대(어느 한 유형이라도 공급 0인 구역)
    stack = np.stack([v[2] for v in per_type.values()])
    oi_total = stack.max(axis=0) if p["agg"] == "max" else stack.mean(axis=0)
    uncovered = np.any(np.stack([v[1] < 1e-6 for v in per_type.values()]), axis=0)

    rows, cols = np.where(in_area)
    data = {
        "구역": [f"R{r + 1:02d}-C{c + 1:02d}" for r, c in zip(rows, cols)],
        "행": rows, "열": cols,
        "위도": LAT[in_area], "경도": LNG[in_area],
        "중심거리(km)": dist_c[in_area], "추정인구": population[in_area],
    }
    for t, (demand, supply, oi) in per_type.items():
        data[f"수요_{t}"] = demand[in_area]
        data[f"공급_{t}"] = supply[in_area]
        data[f"부하율_{t}"] = oi[in_area]
    data["종합 부하율"] = oi_total[in_area]
    data["사각지대"] = uncovered[in_area]

    return {
        "df": pd.DataFrame(data),
        "active": list(p["active"]),
        "grid": {"n": n, "lats": lats, "lngs": lngs, "half_lat": half_lat, "half_lng": half_lng},
    }


# ===========================================================
# B. 페이지 설정 · API 키 · Google API 래퍼
# ===========================================================
# 주의: st.set_page_config()는 첫 번째 Streamlit 명령이어야 한다.
st.set_page_config(page_title="실제 도시 인프라 과부하 분석 시뮬레이터", page_icon="🏙️", layout="wide")
ss = st.session_state


def load_api_key():
    """.env(python-dotenv) 또는 st.secrets에서 GOOGLE_MAPS_API_KEY를 읽는다."""
    load_dotenv()
    key = None
    try:
        key = st.secrets.get("GOOGLE_MAPS_API_KEY")    # secrets.toml이 없으면 예외 → 무시
    except Exception:
        key = None
    if not key:
        key = os.getenv("GOOGLE_MAPS_API_KEY")
    return str(key).strip() if key else None


API_KEY = load_api_key()
if not API_KEY:
    st.error("🔑 Google Maps API Key를 찾을 수 없습니다. `.env` 파일(GOOGLE_MAPS_API_KEY=...) 또는 "
             "`.streamlit/secrets.toml`을 확인해 주세요.")
    st.stop()


def mask_secret(err) -> str:
    """오류 메시지(URL 등)에 API 키가 섞여 화면에 노출되는 것을 방지."""
    return str(err).replace(API_KEY, "***")


@st.cache_resource
def get_gmaps_client(api_key: str):
    """googlemaps 클라이언트는 한 번만 만들어 재사용."""
    return googlemaps.Client(key=api_key)


try:
    get_gmaps_client(API_KEY)
except Exception as e:  # 예: 키 형식 오류(ValueError)
    st.error(f"Google Maps 클라이언트 초기화 실패: {mask_secret(e)}")
    st.stop()


@st.cache_data(ttl=86400, show_spinner=False)
def geocode_address(query: str):
    """Geocoding API: 지역명 → 위·경도 (24시간 캐시). 결과가 없으면 None."""
    res = get_gmaps_client(API_KEY).geocode(query, language="ko", region="kr")
    if not res:
        return None
    loc = res[0]["geometry"]["location"]
    return {"lat": loc["lat"], "lng": loc["lng"], "address": res[0].get("formatted_address", query)}


@st.cache_data(ttl=86400, show_spinner=False)
def reverse_geocode_cached(lat: float, lng: float):
    """Reverse Geocoding: 위·경도 → 주소 (24시간 캐시)."""
    res = get_gmaps_client(API_KEY).reverse_geocode((lat, lng), language="ko")
    return res[0].get("formatted_address") if res else None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_places_by_type(lat: float, lng: float, radius_m: int, place_type: str, max_pages: int):
    """
    Places Nearby Search를 유형 1개에 대해 호출 (1시간 캐시로 API 비용 절감).
    실패 시 예외를 그대로 올린다 → 오류 결과는 캐시되지 않는다.
    """
    client = get_gmaps_client(API_KEY)
    places, token = [], None
    for _ in range(max_pages):
        resp = None
        for attempt in range(4):
            try:
                if token:
                    resp = client.places_nearby(page_token=token)
                else:
                    resp = client.places_nearby(location=(lat, lng), radius=radius_m,
                                                type=place_type, language="ko")
                break
            except googlemaps.exceptions.ApiError as e:
                # next_page_token은 발급 직후 잠시 유효하지 않을 수 있어 재시도
                if token and "INVALID_REQUEST" in str(e) and attempt < 3:
                    time.sleep(2)
                    continue
                raise
        for p in resp.get("results", []):
            if p.get("business_status") == "CLOSED_PERMANENTLY":
                continue                                    # 폐업 시설 제외
            loc = p["geometry"]["location"]
            places.append({"place_id": p.get("place_id"), "name": p.get("name", "이름 없음"),
                           "lat": loc["lat"], "lng": loc["lng"]})
        token = resp.get("next_page_token")
        if not token:
            break
        time.sleep(2)                                       # 다음 페이지 토큰 활성화 대기
    return places


def collect_facilities(lat, lng, radius_m, type_names, max_pages):
    """선택한 유형별로 주변 시설을 수집. (시설 리스트, 오류 메시지 리스트) 반환."""
    facilities, errors, seen = [], [], set()
    for tname in type_names:
        cfg = FACILITY_TYPES[tname]
        try:
            places = fetch_places_by_type(round(lat, 5), round(lng, 5), int(radius_m),
                                          cfg["place_type"], int(max_pages))
        except Exception as e:
            errors.append(f"{tname} 수집 실패: {mask_secret(e)}")
            continue
        for p in places:
            key = (p["place_id"], tname)
            if key in seen:
                continue
            seen.add(key)
            facilities.append(make_facility(tname, p["name"], p["lat"], p["lng"],
                                            source="google", fid=f"{p['place_id']}|{tname}"))
    return facilities, errors


# ===========================================================
# C. 세션 상태 · 콜백
# ===========================================================
def init_state():
    ss.setdefault("center", {"lat": 37.5665, "lng": 126.9780, "label": "서울시청",
                             "address": None, "radius_km": 3.0})
    ss.setdefault("facilities", [])          # 현재(편집 중) 시설
    ss.setdefault("facilities_base", [])     # 수집 직후 원본 (비교/초기화용)
    ss.setdefault("loaded", False)           # 한 번이라도 시설을 수집했는지
    ss.setdefault("view_center", [37.5665, 126.9780])
    ss.setdefault("view_zoom", 13)
    ss.setdefault("clicked", None)           # 마지막 지도 클릭 좌표
    ss.setdefault("last_raw_click", None)


def flash(level: str, msg: str):
    """rerun 이후에도 보이도록 메시지를 세션에 저장 (level: success/info/warning/error)."""
    ss.setdefault("flash", []).append((level, msg))


def show_flash():
    for level, msg in ss.pop("flash", []):
        getattr(st, level)(msg)


def zoom_for_radius(radius_km: float) -> int:
    return 14 if radius_km <= 1.5 else 13 if radius_km <= 3.5 else 12


def center_key(c) -> str:
    return f"{c['lat']:.4f}_{c['lng']:.4f}"


def process_pending_search():
    """
    검색/지도클릭 버튼이 예약한 요청을 위젯 생성 '이전'에 처리한다.
    (위젯 생성 후에는 해당 위젯의 session_state 키를 수정할 수 없기 때문)
    """
    req = ss.pop("pending_search", None)
    if not req:
        return
    types = ss.get("search_types") or []
    if not types:
        flash("warning", "수집할 시설 유형을 하나 이상 선택해 주세요.")
        return
    radius_m = int(ss.get("search_radius_m", 3000))
    max_pages = int(ss.get("search_pages", 1))

    with st.spinner("Google Maps에서 위치와 주변 시설을 조회하는 중..."):
        # (1) 중심 좌표 결정
        if req["mode"] == "query":
            query = req["query"].strip()
            if not query:
                flash("warning", "검색할 지역명을 입력해 주세요.")
                return
            try:
                geo = geocode_address(query)
            except Exception as e:
                flash("error", f"Geocoding 오류: {mask_secret(e)}")
                return
            if not geo:
                flash("error", f"'{query}'에 해당하는 위치를 찾지 못했습니다.")
                return
            lat, lng, label, address = geo["lat"], geo["lng"], query, geo["address"]
        else:  # 지도 클릭 지점
            lat, lng = req["lat"], req["lng"]
            try:
                address = reverse_geocode_cached(round(lat, 5), round(lng, 5))
            except Exception:
                address = None
            label = address or f"지도 클릭 ({lat:.4f}, {lng:.4f})"

        # (2) 주변 시설 수집
        facs, errors = collect_facilities(lat, lng, radius_m, types, max_pages)

    for msg in errors:
        flash("warning", msg)
    if not facs and errors:
        flash("error", "시설을 수집하지 못해 기존 데이터를 유지합니다. (API 활성화/결제/키 제한을 확인하세요)")
        return

    ss["center"] = {"lat": lat, "lng": lng, "label": label, "address": address, "radius_km": radius_m / 1000}
    ss["view_center"], ss["view_zoom"] = [lat, lng], zoom_for_radius(radius_m / 1000)
    ss["facilities"] = [dict(f) for f in facs]
    ss["facilities_base"] = [dict(f) for f in facs]
    ss["loaded"], ss["clicked"] = True, None
    ss["skip_view_sync"] = True                 # 새 중심으로 이동한 직후엔 이전 지도 뷰 값을 덮어쓰지 않음
    for k in ("del_ids", "edit_id"):            # 사라진 시설을 가리키는 위젯 상태 정리
        ss.pop(k, None)
    counts = ", ".join(f"{t} {sum(1 for f in facs if f['type'] == t)}곳" for t in types)
    flash("success", f"'{label}' 반경 {radius_m / 1000:.1f}km 수집 완료 → 총 {len(facs)}곳 ({counts})")
    if not facs:
        flash("warning", "검색 결과가 0건입니다. 반경을 넓히거나 다른 지역을 검색해 보세요.")


def cb_search_query():
    ss["pending_search"] = {"mode": "query", "query": ss.get("search_query", "")}


def cb_search_click():
    clk = ss.get("clicked")
    if clk:
        ss["pending_search"] = {"mode": "click", "lat": clk["lat"], "lng": clk["lng"]}


def cb_add_facility():
    """사이드바 '시설 추가' 버튼 콜백."""
    t = ss["add_type"]
    c = ss["center"]
    mode = ss.get("add_pos_mode", "분석 중심")
    if mode == "분석 중심":
        lat, lng = c["lat"], c["lng"]
    elif mode == "마지막 지도 클릭 지점":
        clk = ss.get("clicked")
        if not clk:
            flash("warning", "먼저 지도에서 위치를 클릭해 주세요.")
            return
        lat, lng = clk["lat"], clk["lng"]
    else:
        ck = center_key(c)
        lat, lng = ss[f"add_lat_{ck}"], ss[f"add_lng_{ck}"]
    n_user = sum(1 for f in ss["facilities"] if f["source"] == "user") + 1
    name = (ss.get("add_name") or "").strip() or f"[추가] {t} {n_user}"
    ss["facilities"].append(make_facility(t, name, lat, lng, ss[f"add_cap_{t}"], ss[f"add_rad_{t}"]))
    flash("success", f"'{name}' 시설을 ({lat:.5f}, {lng:.5f})에 추가했습니다.")


def cb_delete_facilities():
    ids = set(ss.get("del_ids", []))
    ss["facilities"] = [f for f in ss["facilities"] if f["id"] not in ids]
    ss["del_ids"] = []
    ss.pop("edit_id", None)
    flash("success", f"{len(ids)}개 시설을 삭제했습니다.")


def cb_edit_facility(fid: str):
    for f in ss["facilities"]:
        if f["id"] == fid:
            f["capacity"] = int(ss[f"edit_cap_{fid}"])
            f["radius_km"] = float(ss[f"edit_rad_{fid}"])
            flash("success", f"'{f['name']}' 용량/반경을 수정했습니다.")
            return


def cb_reset_facilities():
    ss["facilities"] = [dict(f) for f in ss["facilities_base"]]
    for k in ("del_ids", "edit_id"):
        ss.pop(k, None)
    for t in FACILITY_TYPES:
        ss[f"mult_{t}"] = 100
    flash("info", "시설 목록과 용량 배율을 수집 직후 상태로 되돌렸습니다.")


# ===========================================================
# D. 사이드바 UI
# ===========================================================
init_state()
process_pending_search()

with st.sidebar:
    st.title("🎛️ 컨트롤 패널")

    # ---- 1. 위치 검색 · 데이터 수집 ----
    with st.expander("📍 1. 위치 검색 · 시설 수집", expanded=True):
        st.text_input("도시/지역명 (예: 서울시청, 강남역)", value="서울시청", key="search_query")
        st.slider("시설 검색 반경 (m)", 500, 5000, 3000, 100, key="search_radius_m")
        st.multiselect("수집할 시설 유형", list(FACILITY_TYPES), default=list(FACILITY_TYPES), key="search_types")
        st.slider("유형별 수집 페이지 수 (페이지당 최대 20곳)", 1, 3, 1, key="search_pages",
                  help="페이지를 늘릴수록 더 많은 시설을 수집하지만 API 호출(비용)과 대기 시간이 늘어납니다.")
        st.button("🔍 검색 및 시설 수집", type="primary", on_click=cb_search_query)

    facilities = ss["facilities"]
    fac_label = {f["id"]: f"{f['name']} ({f['type']})" for f in facilities}
    center = ss["center"]

    # ---- 2. 시설 관리 ----
    with st.expander("🏗️ 2. 시설 관리 (추가 · 삭제 · 용량 조정)"):
        tab_add, tab_del, tab_cap = st.tabs(["➕ 추가", "🗑️ 삭제", "🔧 용량"])

        with tab_add:
            add_type = st.selectbox("시설 유형", list(FACILITY_TYPES), key="add_type")
            acfg = FACILITY_TYPES[add_type]
            st.text_input("시설명 (비우면 자동 생성)", key="add_name")
            # 유형별로 위젯 key를 분리해 유형을 바꾸면 해당 유형의 기본값이 나타나도록 함
            st.number_input("수용 용량", 10, 5000, int(acfg["capacity"]), 10, key=f"add_cap_{add_type}")
            st.number_input("커버리지 반경 (km)", 0.5, 10.0, float(acfg["radius_km"]), 0.5, key=f"add_rad_{add_type}")
            pos_mode = st.radio("추가 위치", ["분석 중심", "마지막 지도 클릭 지점", "직접 입력"], key="add_pos_mode")
            if pos_mode == "직접 입력":
                ck = center_key(center)
                st.number_input("위도", -90.0, 90.0, float(center["lat"]), 0.001, format="%.5f", key=f"add_lat_{ck}")
                st.number_input("경도", -180.0, 180.0, float(center["lng"]), 0.001, format="%.5f", key=f"add_lng_{ck}")
            elif pos_mode == "마지막 지도 클릭 지점":
                clk = ss.get("clicked")
                st.caption(f"클릭 좌표: {clk['lat']:.5f}, {clk['lng']:.5f}" if clk else "지도를 클릭하면 좌표가 저장됩니다.")
            st.button("시설 추가", on_click=cb_add_facility, disabled=not ss["loaded"])

        with tab_del:
            if facilities:
                sel = st.multiselect("삭제할 시설", list(fac_label), format_func=fac_label.get, key="del_ids")
                st.button("선택 시설 삭제", on_click=cb_delete_facilities, disabled=not sel)
            else:
                st.caption("삭제할 시설이 없습니다.")

        with tab_cap:
            st.markdown("**유형별 수용 용량 배율 (%)**")
            mult = {t: st.slider(t, 20, 300, 100, 5, key=f"mult_{t}") for t in FACILITY_TYPES}
            st.markdown("**개별 시설 수정**")
            if facilities:
                fid = st.selectbox("시설 선택", list(fac_label), format_func=fac_label.get, key="edit_id")
                cur = next(f for f in facilities if f["id"] == fid)
                st.number_input("수용 용량", 10, 5000, int(cur["capacity"]), 10, key=f"edit_cap_{fid}")
                st.number_input("커버리지 반경 (km)", 0.5, 10.0, float(cur["radius_km"]), 0.5, key=f"edit_rad_{fid}")
                st.button("수정 적용", on_click=cb_edit_facility, args=(fid,))
            else:
                st.caption("수정할 시설이 없습니다.")
            st.button("↩️ 수집 직후 상태로 초기화", on_click=cb_reset_facilities, disabled=not ss["loaded"])

    # ---- 3. 분석 기준 ----
    with st.expander("⚙️ 3. 분석 기준"):
        active_types = st.multiselect("분석 대상 시설 유형", list(FACILITY_TYPES),
                                      default=list(FACILITY_TYPES), key="active_types")
        grid_n = st.slider("격자 해상도 (N×N 구역)", 5, 20, 10)
        decay = st.selectbox("커버리지 감쇠 방식", DECAY_MODES)
        agg = st.radio("종합 부하율 산출", ["mean", "max"], horizontal=True,
                       format_func=lambda x: "평균" if x == "mean" else "최댓값 (가장 취약한 서비스)")
        base_pop = st.number_input("구역당 기본 추정 인구 (명)", 500, 50000, 5000, 500)
        conc = st.slider("도심 집중도", 0.0, 4.0, 1.5, 0.1, help="0이면 모든 구역 인구가 균일, 클수록 중심부에 집중")
        noise_sigma = st.slider("수요 변동성 (잡음)", 0.0, 0.6, 0.2, 0.05)
        seed = st.number_input("난수 시드", 0, 9999, 42)
        st.markdown("**유형별 수요율 (인구 대비)**")
        rates = {t: st.slider(t, 0.0, 0.30, float(FACILITY_TYPES[t]["demand_rate"]), 0.005,
                              format="%.3f", key=f"rate_{t}") for t in FACILITY_TYPES}
        over_thr = st.slider("과부하 기준 (부하율 ≥)", 0.5, 2.0, 1.0, 0.05)
        crit_thr = max(st.slider("심각 기준 (부하율 ≥)", 1.0, 4.0, 1.5, 0.05), over_thr + 0.1)

    # ---- 4. 지도 표시 ----
    with st.expander("🗺️ 4. 지도 표시"):
        view_options = ["종합"] + list(active_types)
        if ss.get("view_metric") not in view_options:
            ss.pop("view_metric", None)
        view_metric = st.selectbox("지도에 표시할 지표", view_options, key="view_metric")
        tile_choice = st.radio("지도 타일", ["Google Maps", "OpenStreetMap"], horizontal=True)
        show_zones = st.checkbox("구역별 부하율 오버레이", True)
        show_cover = st.checkbox("시설 커버리지 반경 표시", False)


# ===========================================================
# E. 지도 · 차트 생성 함수
# ===========================================================
def build_map(center, facilities, result, view_col, view_label, over, crit,
              tile_choice, show_zones, show_cover, clicked):
    """Folium 지도 생성: 타일 → 부하율 구역 → 분석 반경 → 시설 마커 → 클릭 지점."""
    m = folium.Map(location=ss["view_center"], zoom_start=ss["view_zoom"], tiles=None, control_scale=True)
    if tile_choice == "Google Maps":
        folium.TileLayer(
            tiles=f"https://mt1.google.com/vt/lyrs=m&hl=ko&x={{x}}&y={{y}}&z={{z}}&key={API_KEY}",
            attr="Google Maps", name="Google Maps", max_zoom=20).add_to(m)
    else:
        folium.TileLayer("OpenStreetMap").add_to(m)

    # 구역별 부하율 오버레이 (색상 = 상태)
    if show_zones and result is not None:
        g = result["grid"]
        df = result["df"]
        statuses = classify_array(df[view_col], over, crit)
        for r, status in zip(df.to_dict("records"), statuses):
            color = STATUS_COLORS[status]
            lines = [f"<b>{r['구역']}</b>", f"{view_label} 부하율: <b>{r[view_col]:.2f}</b> ({status})",
                     f"추정 인구: {r['추정인구']:,.0f}명"]
            lines += [f"{t}: 수요 {r[f'수요_{t}']:.0f} / 공급 {r[f'공급_{t}']:.0f}" for t in result["active"]]
            folium.Rectangle(
                bounds=[[r["위도"] - g["half_lat"], r["경도"] - g["half_lng"]],
                        [r["위도"] + g["half_lat"], r["경도"] + g["half_lng"]]],
                color=color, weight=1, fill=True, fill_color=color, fill_opacity=0.35,
                tooltip=folium.Tooltip("<br>".join(lines)),
            ).add_to(m)

    # 분석 반경 + 중심점
    folium.Circle([center["lat"], center["lng"]], radius=center["radius_km"] * 1000, color="#444",
                  weight=1.5, dash_array="6", fill=False).add_to(m)
    folium.CircleMarker([center["lat"], center["lng"]], radius=6, color="black", fill=True,
                        fill_opacity=1, tooltip="분석 중심").add_to(m)

    # 시설 마커 (경찰서 파랑 / 병원 빨강 / 소방서 주황)
    for f in facilities:
        cfg = FACILITY_TYPES[f["type"]]
        origin = "사용자 추가" if f["source"] == "user" else "Google Places"
        popup = (f"<b>{escape(f['name'])}</b> ({f['type']})<br>용량 {f['capacity']} · 반경 {f['radius_km']:.1f}km"
                 f"<br><small>{origin}</small>")
        folium.Marker([f["lat"], f["lng"]], popup=folium.Popup(popup, max_width=260),
                      tooltip=f"{f['name']} ({f['type']})",
                      icon=folium.Icon(color=cfg["color"], icon=cfg["icon"])).add_to(m)
        if show_cover:
            folium.Circle([f["lat"], f["lng"]], radius=f["radius_km"] * 1000, color=cfg["hex"],
                          weight=1, fill=True, fill_opacity=0.04).add_to(m)

    # 마지막 클릭 지점
    if clicked:
        folium.Marker([clicked["lat"], clicked["lng"]], tooltip="클릭한 지점",
                      icon=folium.Icon(color="purple", icon="screenshot")).add_to(m)
    return m


def build_heatmap(result, view_col, view_label, facilities, over, crit):
    """Plotly 히트맵: 부하율 격자 + 시설 위치 (색상 구간은 상태 기준과 동일)."""
    g, df = result["grid"], result["df"]
    z = np.full((g["n"], g["n"]), np.nan)                 # 분석 반경 밖 구역은 NaN(빈칸)
    z[df["행"].to_numpy(), df["열"].to_numpy()] = df[view_col].to_numpy()
    zmax = float(max(crit * 1.5, np.nanmax(z)))
    low = 0.7 * over
    c = STATUS_COLORS
    scale = [[0, c["여유"]], [low / zmax, c["여유"]], [low / zmax, c["보통"]], [over / zmax, c["보통"]],
             [over / zmax, c["과부하"]], [crit / zmax, c["과부하"]], [crit / zmax, c["심각"]], [1, c["심각"]]]
    fig = go.Figure(go.Heatmap(
        z=z, x=np.round(g["lngs"], 5), y=np.round(g["lats"], 5), colorscale=scale, zmin=0, zmax=zmax,
        colorbar=dict(title="부하율"), hovertemplate="경도 %{x}<br>위도 %{y}<br>부하율 %{z:.2f}<extra></extra>"))
    for t, cfg in FACILITY_TYPES.items():
        sub = [f for f in facilities if f["type"] == t]
        if sub:
            fig.add_trace(go.Scatter(
                x=[f["lng"] for f in sub], y=[f["lat"] for f in sub], mode="markers", name=t,
                text=[f["name"] for f in sub], hovertemplate="%{text}<extra>" + t + "</extra>",
                marker=dict(size=9, color=cfg["hex"], line=dict(color="white", width=1.5))))
    fig.update_xaxes(range=[g["lngs"].min() - g["half_lng"], g["lngs"].max() + g["half_lng"]], title="경도")
    fig.update_yaxes(range=[g["lats"].min() - g["half_lat"], g["lats"].max() + g["half_lat"]], title="위도")
    fig.update_layout(title=f"{view_label} 부하율 히트맵", height=560, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.12))
    return fig


def summarize(result, view_col, over):
    df = result["df"]
    return {"avg": float(df[view_col].mean()), "over": int((df[view_col] >= over).sum()),
            "gap": int(df["사각지대"].sum()), "zones": len(df)}


# ===========================================================
# F. 메인 대시보드
# ===========================================================
st.title("🏙️ 실제 도시 인프라 과부하 분석 시뮬레이터")
st.caption("Google Maps(Geocoding · Places) 실데이터와 Haversine 거리 기반으로 구역별 인프라 수요·공급·부하율을 시뮬레이션합니다.")
show_flash()

center, facilities = ss["center"], ss["facilities"]
st.markdown(f"**분석 중심:** {center['label']} · 위도 `{center['lat']:.5f}` / 경도 `{center['lng']:.5f}` "
            f"· 분석 반경 `{center['radius_km']:.1f} km`")

# ---- 분석 실행 ----
view_col = "종합 부하율" if view_metric == "종합" else f"부하율_{view_metric}"
result = base_result = None
if not ss["loaded"]:
    st.info("👈 사이드바에서 지역명을 입력하고 **검색 및 시설 수집**을 눌러 주세요. 지도를 클릭해 새 중심을 정할 수도 있습니다.")
elif not active_types:
    st.warning("분석 대상 시설 유형을 하나 이상 선택해 주세요. (사이드바 3. 분석 기준)")
else:
    params = dict(grid_n=grid_n, decay=decay, agg=agg, base_pop=base_pop, conc=conc, noise=noise_sigma,
                  seed=seed, rates=rates, mult=mult, active=active_types)
    result = analyze_zones(facilities_to_df(facilities), center, params)
    # 비교 기준: 수집 직후 원본 시설(용량 배율 100%)로 같은 분석을 한 번 더 수행
    base_params = {**params, "mult": {t: 100 for t in FACILITY_TYPES}}
    base_result = analyze_zones(facilities_to_df(ss["facilities_base"]), center, base_params)

    cur, base = summarize(result, view_col, over_thr), summarize(base_result, view_col, over_thr)
    for t in active_types:
        if not any(f["type"] == t for f in facilities):
            st.warning(f"'{t}' 시설이 없어 해당 유형은 전 구역이 사각지대(부하율 상한)로 계산됩니다.")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("보유 시설", f"{len(facilities)}곳", f"{len(facilities) - len(ss['facilities_base']):+d} (원본 대비)")
    m2.metric("분석 구역", f"{cur['zones']}개")
    m3.metric(f"평균 {view_metric} 부하율", f"{cur['avg']:.2f}", f"{cur['avg'] - base['avg']:+.2f}", delta_color="inverse")
    m4.metric("과부하 구역", f"{cur['over']}개", f"{cur['over'] - base['over']:+d}", delta_color="inverse")
    m5.metric("사각지대 구역", f"{cur['gap']}개", f"{cur['gap'] - base['gap']:+d}", delta_color="inverse")

# ---- 지도 + 우측 패널 ----
col_map, col_side = st.columns([7, 3])

with col_map:
    st.subheader("🗺️ 지도")
    fmap = build_map(center, facilities, result, view_col, view_metric, over_thr, crit_thr,
                     tile_choice, show_zones, show_cover, ss.get("clicked"))
    # returned_objects를 제한해 불필요한 rerun을 줄이고, 지도 뷰(center/zoom)는 세션에 보존
    map_data = st_folium(fmap, key="main_map", height=600, use_container_width=True,
                         returned_objects=["last_clicked", "center", "zoom"],
                         center=ss["view_center"], zoom=ss["view_zoom"])
    if map_data:
        raw = map_data.get("last_clicked")
        if raw and raw != ss["last_raw_click"]:       # 새 클릭일 때만 반영 (이전 클릭 재처리 방지)
            ss["last_raw_click"] = raw
            ss["clicked"] = {"lat": raw["lat"], "lng": raw["lng"]}
        if ss.pop("skip_view_sync", False):
            pass
        else:
            if map_data.get("center"):
                ss["view_center"] = [map_data["center"]["lat"], map_data["center"]["lng"]]
            if map_data.get("zoom"):
                ss["view_zoom"] = int(map_data["zoom"])

with col_side:
    st.subheader("📌 클릭 지점")
    clk = ss.get("clicked")
    if clk:
        st.info(f"👉 위도 `{clk['lat']:.5f}`, 경도 `{clk['lng']:.5f}`")
        try:
            addr = reverse_geocode_cached(round(clk["lat"], 5), round(clk["lng"], 5))
            st.caption(f"📍 {addr}" if addr else "주소 정보를 찾지 못했습니다.")
        except Exception as e:
            st.caption(f"주소 조회 실패: {mask_secret(e)}")
        st.button("🎯 이 지점을 새 분석 중심으로 설정", on_click=cb_search_click)
    else:
        st.caption("지도를 클릭하면 좌표와 주소가 표시됩니다.")

    st.subheader("🚦 부하율 범례")
    low = 0.7 * over_thr
    st.markdown(f"🟢 여유 (< {low:.2f})  \n🟡 보통 ({low:.2f} ~ {over_thr:.2f})  \n"
                f"🟠 과부하 ({over_thr:.2f} ~ {crit_thr:.2f})  \n🔴 심각 (≥ {crit_thr:.2f})")

    if result is not None:
        st.subheader(f"🔥 위험 구역 TOP 5 ({view_metric})")
        top = result["df"].nlargest(5, view_col)[["구역", view_col]].copy()
        top["상태"] = classify_array(top[view_col], over_thr, crit_thr)
        st.dataframe(top.round(2), hide_index=True)

# ---- 상세 탭 ----
tab_heat, tab_fac, tab_zone, tab_info = st.tabs(["🔥 히트맵", "🏢 시설 목록", "🧮 구역 상세", "ℹ️ 계산 방식"])

with tab_heat:
    if result is not None:
        st.plotly_chart(build_heatmap(result, view_col, view_metric, facilities, over_thr, crit_thr))
    else:
        st.caption("시설을 수집하면 히트맵이 표시됩니다.")

with tab_fac:
    if facilities:
        fdf = facilities_to_df(facilities)
        fdf["유효 용량"] = (fdf["capacity"] * fdf["type"].map(lambda t: mult[t] / 100.0)).round(0)
        fdf["출처"] = fdf["source"].map({"google": "Google Places", "user": "사용자 추가"})
        st.dataframe(fdf.rename(columns={"name": "시설명", "type": "유형", "lat": "위도", "lng": "경도",
                                         "capacity": "기본 용량", "radius_km": "반경(km)"})
                     [["시설명", "유형", "위도", "경도", "기본 용량", "유효 용량", "반경(km)", "출처"]],
                     hide_index=True)
    else:
        st.caption("표시할 시설이 없습니다.")

with tab_zone:
    if result is not None:
        zdf = result["df"].drop(columns=["행", "열"]).round(3)
        zdf["상태"] = classify_array(result["df"][view_col], over_thr, crit_thr)
        st.dataframe(zdf, hide_index=True)
        st.download_button("⬇️ 구역 분석 결과 CSV 다운로드", zdf.to_csv(index=False).encode("utf-8-sig"),
                           file_name="zone_analysis.csv", mime="text/csv")
    else:
        st.caption("분석 결과가 없습니다.")

with tab_info:
    st.markdown("""
**계산 순서**
1. 분석 반경 안을 N×N 격자로 나누고, 각 구역 중심 좌표를 구합니다.
2. 추정 인구 = `기본 인구 × (1 + 도심집중도 × exp(-(중심거리/σ)²)) × 로그정규 잡음` (σ = 반경의 50%)
3. 수요(유형) = `추정 인구 × 유형별 수요율`
4. 공급(유형) = `Σ 시설 용량 × 용량 배율 × 거리 가중치(Haversine 거리 / 커버리지 반경)`
5. 부하율(OI) = `수요 / 공급` (공급 0 → 상한 5.0, 사각지대로 집계) → 종합 = 유형별 OI의 평균 또는 최댓값

**주의**: Google은 실제 인구/이용량을 제공하지 않으므로 인구·수요율·시설 용량은 *가정값*입니다.
실제 통계(예: 격자 인구 CSV)를 `analyze_zones()`의 `population` 계산 부분에 연결하면 정확도를 높일 수 있습니다.
분석 반경 바깥 시설은 수집하지 않으므로 가장자리 구역의 공급은 다소 과소 추정될 수 있습니다.
""")
