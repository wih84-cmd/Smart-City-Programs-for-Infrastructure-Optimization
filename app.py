# -*- coding: utf-8 -*-
"""도시 인프라 과부하 분석 + 신규 시설 시뮬레이터

설치:
    pip install streamlit streamlit-folium folium branca requests python-dotenv numpy pandas plotly

API 키 설정(서버용/브라우저용 각각 필요):
    .env:
        GOOGLE_MAPS_API_KEY=YOUR_SERVER_API_KEY
        GOOGLE_MAPS_BROWSER_API_KEY=YOUR_BROWSER_API_KEY
    .streamlit/secrets.toml:
        GOOGLE_MAPS_API_KEY = "YOUR_SERVER_API_KEY"
        GOOGLE_MAPS_BROWSER_API_KEY = "YOUR_BROWSER_API_KEY"

서버용 키는 IP 제한 + Places API/Geocoding API 제한, 브라우저용 키는
HTTP referrer 제한 + Maps JavaScript API 제한을 각각 적용하세요.
브라우저용 키는 지도 표시를 위해 웹 브라우저에 전달되므로 반드시 별도 키를 사용하세요.

실행:
    streamlit run app.py

Google Cloud에서 Places API (New)를 활성화해야 합니다. Geocoding API는 선택 사항이며, Maps JavaScript API는 Google 지도 바탕을 쓸 때만 필요합니다.
주의: Google Places는 시설의 이름/위치/유형을 제공하지만, 실제 수용 인원이나
지역별 인구·이용량을 제공하지 않습니다. 따라서 용량·수요·커버리지 값은
명시적으로 설정하는 시뮬레이션 가정이며, 공인된 행정 통계가 아닙니다.
"""
from __future__ import annotations

import os
import time
import uuid
from html import escape
from typing import Any

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from branca.element import MacroElement, Template
from dotenv import load_dotenv
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# -----------------------------------------------------------------------------
# 1. 설정 및 분석 모델
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="도시 인프라 과부하 분석 시뮬레이터",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

EARTH_RADIUS_KM = 6371.0088
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LNG_EQ = 111.32
EPS = 1e-9
OI_CAP = 5.0
PLACES_NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
PLACES_TEXT_URL = "https://places.googleapis.com/v1/places:searchText"
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

FACILITY_TYPES: dict[str, dict[str, Any]] = {
    "경찰서": {
        "place_type": "police", "color": "blue", "hex": "#2878B5", "icon": "shield",
        "capacity": 150, "radius_km": 2.5, "demand_rate": 0.04,
    },
    "병원": {
        "place_type": "hospital", "color": "red", "hex": "#D64545", "icon": "hospital",
        "capacity": 200, "radius_km": 3.0, "demand_rate": 0.08,
    },
    "소방서": {
        "place_type": "fire_station", "color": "orange", "hex": "#E58A27", "icon": "fire",
        "capacity": 120, "radius_km": 3.0, "demand_rate": 0.03,
    },
}
FAC_COLUMNS = ["id", "name", "type", "lat", "lng", "capacity", "radius_km", "source", "place_id", "google_primary_type", "google_types"]
DECAY_MODES = ["선형 감쇠", "가우시안 감쇠", "균등 (반경 내 전량)"]
STATUS_COLORS = {"여유": "#2ca02c", "보통": "#e5bd35", "과부하": "#f28e2b", "심각": "#d62728"}


def haversine_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> Any:
    """Haversine 공식. 스칼라 및 numpy 배열 입력을 지원한다."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def coverage_weight(dist_km: Any, radius_km: float, mode: str) -> np.ndarray:
    """거리별 시설 기여도(0~1). 지정 반경 밖은 0으로 처리한다."""
    x = np.asarray(dist_km, dtype=float) / max(float(radius_km), 1e-6)
    if mode == DECAY_MODES[0]:
        weight = 1.0 - x
    elif mode == DECAY_MODES[1]:
        weight = np.exp(-3.0 * x**2)
    else:
        weight = np.ones_like(x)
    return np.where(x <= 1.0, np.clip(weight, 0.0, 1.0), 0.0)


def classify_array(oi: Any, overload: float, critical: float) -> np.ndarray:
    arr = np.asarray(oi, dtype=float)
    return np.select(
        [arr >= critical, arr >= overload, arr >= 0.7 * overload],
        ["심각", "과부하", "보통"], default="여유",
    )


def make_facility(
    ftype: str,
    name: str,
    lat: float,
    lng: float,
    capacity: int | None = None,
    radius_km: float | None = None,
    source: str = "user",
    fid: str | None = None,
    place_id: str | None = None,
    google_primary_type: str | None = None,
    google_types: list[str] | None = None,
) -> dict[str, Any]:
    cfg = FACILITY_TYPES[ftype]
    return {
        "id": fid or f"{source}_{uuid.uuid4().hex[:10]}",
        "name": str(name).strip(),
        "type": ftype,
        "lat": float(lat),
        "lng": float(lng),
        "capacity": int(capacity if capacity is not None else cfg["capacity"]),
        "radius_km": float(radius_km if radius_km is not None else cfg["radius_km"]),
        "source": source,
        "place_id": place_id or "",
        "google_primary_type": google_primary_type or "",
        "google_types": ", ".join(google_types or []),
    }


def facilities_to_df(facilities: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(facilities, columns=FAC_COLUMNS) if facilities else pd.DataFrame(columns=FAC_COLUMNS)


def analyze_zones(facility_df: pd.DataFrame, center: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """같은 격자·수요 조건에서 유형별 수요/공급과 OI(수요÷공급)를 계산한다."""
    lat0, lng0, radius_km = center["lat"], center["lng"], center["radius_km"]
    n = int(params["grid_n"])
    step_km = 2.0 * radius_km / n
    offsets = (np.arange(n) + 0.5) * step_km - radius_km
    km_per_deg_lng = max(KM_PER_DEG_LNG_EQ * abs(np.cos(np.radians(lat0))), 1e-4)
    lats = lat0 + offsets / KM_PER_DEG_LAT
    lngs = lng0 + offsets / km_per_deg_lng
    lat_grid, lng_grid = np.meshgrid(lats, lngs, indexing="ij")
    half_lat = step_km / 2.0 / KM_PER_DEG_LAT
    half_lng = step_km / 2.0 / km_per_deg_lng

    center_distance = haversine_km(lat0, lng0, lat_grid, lng_grid)
    in_area = center_distance <= radius_km
    sigma = max(radius_km * 0.5, 0.1)
    if params["noise"] > 0:
        noise = np.random.default_rng(int(params["seed"])).lognormal(
            mean=0.0, sigma=params["noise"], size=(n, n)
        )
    else:
        noise = np.ones((n, n))
    population = params["base_pop"] * (
        1.0 + params["concentration"] * np.exp(-(center_distance / sigma) ** 2)
    ) * noise

    per_type: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for ftype in params["active_types"]:
        demand = population * params["demand_rates"][ftype]
        supply = np.zeros((n, n), dtype=float)
        scale = params["capacity_scale"].get(ftype, 100) / 100.0
        if not facility_df.empty:
            for f in facility_df[facility_df["type"] == ftype].itertuples(index=False):
                distance = haversine_km(f.lat, f.lng, lat_grid, lng_grid)
                supply += float(f.capacity) * scale * coverage_weight(distance, f.radius_km, params["decay"])
        oi = np.minimum(demand / np.maximum(supply, EPS), OI_CAP)
        per_type[ftype] = (demand, supply, oi)

    if not per_type:
        raise ValueError("분석할 시설 유형을 하나 이상 선택해야 합니다.")
    oi_stack = np.stack([v[2] for v in per_type.values()])
    total_oi = oi_stack.max(axis=0) if params["aggregate"] == "max" else oi_stack.mean(axis=0)
    uncovered = np.any(np.stack([v[1] < 1e-6 for v in per_type.values()]), axis=0)

    rows, cols = np.where(in_area)
    data: dict[str, Any] = {
        "구역": [f"R{r + 1:02d}-C{c + 1:02d}" for r, c in zip(rows, cols)],
        "행": rows, "열": cols,
        "위도": lat_grid[in_area], "경도": lng_grid[in_area],
        "중심거리(km)": center_distance[in_area], "추정인구": population[in_area],
    }
    for ftype, (demand, supply, oi) in per_type.items():
        data[f"수요_{ftype}"] = demand[in_area]
        data[f"공급_{ftype}"] = supply[in_area]
        data[f"부하율_{ftype}"] = oi[in_area]
    data["종합 부하율"] = total_oi[in_area]
    data["사각지대"] = uncovered[in_area]

    return {
        "df": pd.DataFrame(data),
        "active_types": list(params["active_types"]),
        "grid": {"n": n, "lats": lats, "lngs": lngs, "half_lat": half_lat, "half_lng": half_lng},
    }


def summarize(result: dict[str, Any], view_col: str, overload: float, critical: float) -> dict[str, Any]:
    df = result["df"]
    values = df[view_col].to_numpy(dtype=float)
    return {
        "avg": float(np.mean(values)),
        "over": int(np.sum(values >= overload)),
        "critical": int(np.sum(values >= critical)),
        "gap": int(df["사각지대"].sum()),
        "zones": len(df),
    }


# -----------------------------------------------------------------------------
# 2. API 키와 Google API (Places API New + Geocoding REST)
# -----------------------------------------------------------------------------

def load_secret(name: str) -> str | None:
    """st.secrets를 먼저 확인하고, 없으면 .env/환경변수에서 읽는다."""
    load_dotenv()
    value = None
    try:
        value = st.secrets.get(name)
    except Exception:
        pass
    value = value or os.getenv(name)
    return str(value).strip() if value else None


API_KEY = load_secret("GOOGLE_MAPS_API_KEY")
BROWSER_API_KEY = load_secret("GOOGLE_MAPS_BROWSER_API_KEY")


def mask_api_key(message: Any) -> str:
    text = str(message)
    if API_KEY:
        text = text.replace(API_KEY, "***")
    if BROWSER_API_KEY:
        text = text.replace(BROWSER_API_KEY, "***")
    return text


def safe_api_error(response: requests.Response) -> str:
    try:
        body = response.json()
        message = body.get("error", {}).get("message") or body.get("error_message") or response.text
    except Exception:
        message = response.text
    return f"HTTP {response.status_code}: {str(message)[:500]}"


def places_text_geocode(query: str, api_key: str) -> dict[str, Any] | None:
    """Places API (New) Text Search로 지역명/주소를 좌표로 변환한다."""
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.location,places.formattedAddress",
    }
    search_query = query if any(x in query for x in ("대한민국", "한국", "Korea")) else f"{query}, 대한민국"
    body = {"textQuery": search_query, "languageCode": "ko", "regionCode": "KR", "pageSize": 1}
    response = requests.post(PLACES_TEXT_URL, headers=headers, json=body, timeout=25)
    if not response.ok:
        raise RuntimeError(safe_api_error(response))
    places = response.json().get("places", [])
    if not places:
        return None
    place = places[0]
    loc = place.get("location") or {}
    if "latitude" not in loc or "longitude" not in loc:
        return None
    label = (place.get("displayName") or {}).get("text") or query
    return {
        "lat": float(loc["latitude"]),
        "lng": float(loc["longitude"]),
        "address": place.get("formattedAddress") or label,
        "label": label,
    }


def resolve_location(query: str, api_key: str) -> dict[str, Any] | None:
    """Places Text Search를 우선 사용하고, 실패하면 Geocoding API를 대체 경로로 시도한다."""
    text_error = None
    try:
        result = places_text_geocode(query, api_key)
        if result:
            return result
    except Exception as exc:
        text_error = exc
    try:
        result = geocode_address(query, api_key)
        if result:
            return result
    except Exception as geo_error:
        if text_error:
            raise RuntimeError(
                f"Places Text Search 실패: {mask_api_key(text_error)} / "
                f"Geocoding 실패: {mask_api_key(geo_error)}"
            ) from geo_error
        raise
    if text_error:
        raise RuntimeError(f"Places Text Search 실패: {mask_api_key(text_error)}")
    return None


def geocode_address(query: str, api_key: str) -> dict[str, Any] | None:
    response = requests.get(
        GEOCODE_URL,
        params={"address": query, "key": api_key, "language": "ko", "region": "kr"},
        timeout=20,
    )
    if not response.ok:
        raise RuntimeError(safe_api_error(response))
    payload = response.json()
    status = payload.get("status")
    if status == "ZERO_RESULTS":
        return None
    if status != "OK":
        raise RuntimeError(payload.get("error_message", f"Geocoding 상태: {status}"))
    item = payload["results"][0]
    loc = item["geometry"]["location"]
    return {"lat": loc["lat"], "lng": loc["lng"], "address": item.get("formatted_address", query)}


def reverse_geocode(lat: float, lng: float, api_key: str) -> str | None:
    response = requests.get(
        GEOCODE_URL,
        params={"latlng": f"{lat},{lng}", "key": api_key, "language": "ko"},
        timeout=20,
    )
    if not response.ok:
        raise RuntimeError(safe_api_error(response))
    payload = response.json()
    if payload.get("status") == "ZERO_RESULTS":
        return None
    if payload.get("status") != "OK":
        raise RuntimeError(payload.get("error_message", f"Reverse Geocoding 상태: {payload.get('status')}"))
    return payload["results"][0].get("formatted_address")


def fetch_places_by_type(
    lat: float, lng: float, radius_m: int, place_type: str, api_key: str
) -> list[dict[str, Any]]:
    """Places API (New) Nearby Search. 유형별 최대 20개를 조회한다."""
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "places.id,places.displayName,places.location,places.primaryType,"
            "places.types,places.businessStatus"
        ),
    }
    body = {
        "includedTypes": [place_type],
        "maxResultCount": 20,
        "languageCode": "ko",
        "locationRestriction": {
            "circle": {
                "center": {"latitude": float(lat), "longitude": float(lng)},
                "radius": float(radius_m),
            }
        },
    }
    last_error = None
    for attempt in range(3):
        try:
            response = requests.post(PLACES_NEARBY_URL, headers=headers, json=body, timeout=25)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                last_error = RuntimeError(safe_api_error(response))
                time.sleep(1.2 * (attempt + 1))
                continue
            if not response.ok:
                raise RuntimeError(safe_api_error(response))
            payload = response.json()
            results = []
            for place in payload.get("places", []):
                if place.get("businessStatus") == "CLOSED_PERMANENTLY":
                    continue
                loc = place.get("location") or {}
                display = place.get("displayName") or {}
                if "latitude" not in loc or "longitude" not in loc:
                    continue
                results.append({
                    "place_id": place.get("id", ""),
                    "name": display.get("text") or "이름 미제공",
                    "lat": float(loc["latitude"]),
                    "lng": float(loc["longitude"]),
                    "primary_type": place.get("primaryType", ""),
                    "types": place.get("types", []),
                })
            return results
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt < 2 and isinstance(exc, requests.RequestException):
                time.sleep(1.0 * (attempt + 1))
                continue
            break
    raise RuntimeError(str(last_error or "Places API 요청에 실패했습니다."))


def collect_facilities(
    lat: float, lng: float, radius_m: int, selected_types: list[str], api_key: str
) -> tuple[list[dict[str, Any]], list[str]]:
    facilities: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    for ftype in selected_types:
        place_type = FACILITY_TYPES[ftype]["place_type"]
        try:
            places = fetch_places_by_type(round(lat, 5), round(lng, 5), int(radius_m), place_type, api_key)
            for place in places:
                pid = place.get("place_id")
                if pid and pid in seen_ids:
                    continue
                if pid:
                    seen_ids.add(pid)
                facilities.append(make_facility(
                    ftype, place["name"], place["lat"], place["lng"],
                    source="google", fid=f"google_{pid or uuid.uuid4().hex[:10]}", place_id=pid,
                    google_primary_type=place.get("primary_type") or place_type,
                    google_types=place.get("types") or [],
                ))
        except Exception as exc:
            errors.append(f"{ftype} 수집 실패: {mask_api_key(exc)[:400]}")
    return facilities, errors


# -----------------------------------------------------------------------------
# 3. 세션 상태와 검색/추가 액션
# -----------------------------------------------------------------------------
ss = st.session_state

def init_state() -> None:
    defaults = {
        "center": {"lat": 37.5665, "lng": 126.9780, "label": "서울시청", "address": "", "radius_km": 3.0},
        "base_facilities": [],       # Google Places에서 수집한 기준 시설(보존)
        "sim_facilities": [],        # 사용자가 추가한 시뮬레이션 시설
        "loaded": False,
        "clicked": None,
        "last_click_token": None,
        "view_center": [37.5665, 126.9780],
        "view_zoom": 13,
        "flash_messages": [],
    }
    for key, value in defaults.items():
        ss.setdefault(key, value)


def notify(level: str, message: str) -> None:
    ss.setdefault("flash_messages", []).append((level, message))


def show_notifications() -> None:
    for level, message in ss.pop("flash_messages", []):
        getattr(st, level if level in ("success", "info", "warning", "error") else "info")(message)


def run_search(query: str, radius_m: int, selected_types: list[str]) -> None:
    query = (query or "").strip()
    if not API_KEY:
        notify("error", "서버용 GOOGLE_MAPS_API_KEY가 없습니다. .env 또는 .streamlit/secrets.toml을 확인해 주세요.")
        return
    if not selected_types:
        notify("warning", "수집할 시설 유형을 하나 이상 선택해 주세요.")
        return
    if not query:
        notify("warning", "검색할 지역명을 입력해 주세요.")
        return
    try:
        with st.spinner("Google Geocoding 및 Places API에서 주변 시설을 수집하고 있습니다..."):
            geo = resolve_location(query, API_KEY)
            if not geo:
                notify("error", f"'{query}' 위치를 찾지 못했습니다.")
                return
            facilities, errors = collect_facilities(
                geo["lat"], geo["lng"], int(radius_m), selected_types, API_KEY
            )
    except Exception as exc:
        notify("error", f"위치/시설 조회에 실패했습니다: {mask_api_key(exc)[:500]}")
        return

    # 모든 유형이 실패했을 때는 이전 정상 데이터를 지우지 않는다.
    if not facilities and errors:
        for message in errors:
            notify("warning", message)
        notify("error", "시설 데이터를 가져오지 못했습니다. API 활성화, 결제 계정, 키 제한 및 Places API (New) 권한을 확인해 주세요. 기존 분석 데이터는 유지했습니다.")
        return

    ss["center"] = {
        "lat": geo["lat"], "lng": geo["lng"], "label": geo.get("label") or query,
        "address": geo.get("address", query), "radius_km": float(radius_m) / 1000.0,
    }
    ss["base_facilities"] = facilities
    ss["sim_facilities"] = []
    ss["loaded"] = True
    ss["clicked"] = None
    # 마지막 지도 클릭 토큰은 유지해 st_folium이 이전 클릭값을 반환해도 재처리하지 않는다.
    ss["view_center"] = [geo["lat"], geo["lng"]]
    ss["view_zoom"] = 13 if radius_m <= 5000 else 11
    for message in errors:
        notify("warning", message)
    counts = ", ".join(f"{t} {sum(f['type'] == t for f in facilities)}곳" for t in selected_types)
    notify("success", f"{query} 주변 Google Places 시설 {len(facilities)}곳을 기준 데이터로 수집했습니다. ({counts})")
    if not facilities:
        notify("warning", "검색 결과가 0건입니다. 검색 반경을 넓히거나 다른 지역을 선택해 보세요.")


def add_simulated_facility(name: str, ftype: str, capacity: int, radius_km: float,
                           position_mode: str, lat: float, lng: float) -> None:
    if position_mode == "지도 클릭 지점":
        clicked = ss.get("clicked")
        if not clicked:
            notify("warning", "먼저 지도에서 신규 시설을 놓을 위치를 클릭해 주세요.")
            return
        lat, lng = float(clicked["lat"]), float(clicked["lng"])
    elif position_mode == "분석 중심":
        lat, lng = ss["center"]["lat"], ss["center"]["lng"]
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        notify("error", "위도/경도 범위가 올바르지 않습니다.")
        return
    clean_name = (name or "").strip()
    if not clean_name:
        n = sum(f["type"] == ftype for f in ss["sim_facilities"]) + 1
        clean_name = f"신설 {ftype} {n}"
    facility = make_facility(ftype, clean_name, lat, lng, capacity, radius_km, source="user")
    ss["sim_facilities"].append(facility)
    notify("success", f"'{clean_name}'을(를) 시뮬레이션 시설로 추가했습니다. 비교 결과에 즉시 반영됩니다.")


init_state()

# -----------------------------------------------------------------------------
# 4. 사이드바: 사용 흐름 1 -> 3
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎛️ 분석 컨트롤")
    st.caption("① 위치/실데이터 수집 → ② 과부하 확인 → ③ 신규 시설 실험")

    with st.expander("① 지역 검색 및 Google 데이터 수집", expanded=True):
        with st.form("search_form", clear_on_submit=False):
            search_query = st.text_input("중심 위치", value=ss.get("search_query", "서울시청"), key="search_query")
            search_radius_m = st.slider("시설 수집 반경", 500, 10000, 3000, 500, format="%d m", key="search_radius_m")
            search_types = st.multiselect(
                "수집할 시설 유형", list(FACILITY_TYPES), default=list(FACILITY_TYPES), key="search_types"
            )
            search_submit = st.form_submit_button("🔍 Google 시설 수집", type="primary", use_container_width=True)
        st.caption("Places API (New)의 Nearby Search를 유형별 1회 호출합니다. 유형별 최대 20개 결과를 사용합니다.")
        if search_submit:
            run_search(search_query, search_radius_m, search_types)

    with st.expander("② 분석 기준 조정", expanded=True):
        active_types = st.multiselect(
            "분석 대상 유형", list(FACILITY_TYPES), default=list(FACILITY_TYPES), key="active_types"
        )
        grid_n = st.slider("격자 해상도", 5, 20, 10, help="N×N 격자. 값이 높을수록 세밀하지만 계산량이 커집니다.")
        decay = st.selectbox("시설 거리 가중치", DECAY_MODES, index=0)
        aggregate = st.radio(
            "종합 부하율", ["mean", "max"], horizontal=True,
            format_func=lambda x: "유형 평균" if x == "mean" else "가장 취약한 유형(최댓값)",
        )
        base_pop = st.number_input("구역당 기본 추정 인구", min_value=500, max_value=50000, value=5000, step=500)
        concentration = st.slider("중심부 인구 집중도", 0.0, 4.0, 1.5, 0.1)
        noise = st.slider("수요 변동성", 0.0, 0.6, 0.2, 0.05)
        seed = st.number_input("난수 시드(전·후 동일 조건)", 0, 9999, 42)
        st.markdown("**유형별 인구 대비 수요율**")
        demand_rates = {
            t: st.slider(t, 0.0, 0.30, float(cfg["demand_rate"]), 0.005, key=f"rate_{t}")
            for t, cfg in FACILITY_TYPES.items()
        }
        st.markdown("**유형별 시설 용량 배율**")
        capacity_scale = {t: st.slider(f"{t} 용량 배율", 20, 300, 100, 5, key=f"scale_{t}") for t in FACILITY_TYPES}
        overload_threshold = st.slider("과부하 기준(OI)", 0.5, 2.0, 1.0, 0.05)
        critical_threshold = st.slider("심각 기준(OI)", 1.0, 4.0, max(1.5, overload_threshold + 0.1), 0.05)
        critical_threshold = max(critical_threshold, overload_threshold + 0.05)

    with st.expander("③ 신규 시설 추가 시뮬레이션", expanded=True):
        if not ss["loaded"]:
            st.info("먼저 ①에서 지역을 검색해 Google 시설 데이터를 수집하세요.")
        with st.form("add_facility_form", clear_on_submit=True):
            sim_name = st.text_input("신규 시설명", placeholder="예: 신설 강남병원")
            sim_type = st.selectbox("신규 시설 유형", list(FACILITY_TYPES), key="sim_type")
            cfg = FACILITY_TYPES[sim_type]
            sim_capacity = st.number_input("시뮬레이션 수용 용량", 10, 5000, int(cfg["capacity"]), 10)
            sim_radius = st.number_input("서비스 반경(km)", 0.5, 10.0, float(cfg["radius_km"]), 0.5)
            position_mode = st.radio("시설 위치 지정", ["지도 클릭 지점", "직접 입력", "분석 중심"], index=0)
            if position_mode == "직접 입력":
                sim_lat = st.number_input("위도", -90.0, 90.0, float(ss["center"]["lat"]), 0.0001, format="%.5f")
                sim_lng = st.number_input("경도", -180.0, 180.0, float(ss["center"]["lng"]), 0.0001, format="%.5f")
            else:
                sim_lat, sim_lng = ss["center"]["lat"], ss["center"]["lng"]
            add_submit = st.form_submit_button("➕ 신규 시설 추가 및 비교", use_container_width=True, disabled=not ss["loaded"])
        if position_mode == "지도 클릭 지점":
            if ss.get("clicked"):
                st.caption(f"선택된 지도 좌표: {ss['clicked']['lat']:.5f}, {ss['clicked']['lng']:.5f}")
            else:
                st.caption("지도에서 추가할 위치를 클릭하면 좌표가 여기에 표시됩니다.")
        if add_submit:
            add_simulated_facility(sim_name, sim_type, int(sim_capacity), float(sim_radius), position_mode, sim_lat, sim_lng)

        if ss["sim_facilities"]:
            st.markdown("**추가된 시뮬레이션 시설**")
            sim_labels = {f["id"]: f"{f['name']} · {f['type']}" for f in ss["sim_facilities"]}
            delete_ids = st.multiselect("삭제할 시설 선택", list(sim_labels), format_func=sim_labels.get, key="delete_sim_ids")
            if st.button("선택한 시뮬레이션 시설 삭제", disabled=not delete_ids, use_container_width=True):
                ss["sim_facilities"] = [f for f in ss["sim_facilities"] if f["id"] not in delete_ids]
                ss.pop("delete_sim_ids", None)
                notify("success", f"시뮬레이션 시설 {len(delete_ids)}곳을 삭제했습니다.")
                st.rerun()
            if st.button("시뮬레이션 시설 전체 초기화", use_container_width=True):
                ss["sim_facilities"] = []
                ss.pop("delete_sim_ids", None)
                notify("info", "신규 시설을 모두 제거했습니다. 기준 데이터는 유지됩니다.")
                st.rerun()

    with st.expander("지도 표시 설정", expanded=False):
        show_zones = st.checkbox("구역별 부하율 색상 표시", True)
        show_coverage = st.checkbox("시설 서비스 반경 표시", False)
        map_scenario = st.radio("지도에 표시할 시설", ["추가 후(시나리오)", "추가 전(기준)"], horizontal=False)

# -----------------------------------------------------------------------------
# 5. 대시보드 계산 및 렌더링
# -----------------------------------------------------------------------------
st.title("🏙️ 도시 인프라 과부하 분석 시뮬레이터")
st.caption("Google Places 실제 시설 위치·명칭 + 거리 기반 공급 모델 + 신규 시설 추가 전/후 시뮬레이션")
show_notifications()

center = ss["center"]
base_facilities = [dict(f) for f in ss["base_facilities"]]
sim_facilities = [dict(f) for f in ss["sim_facilities"]]
scenario_facilities = base_facilities + sim_facilities

if not API_KEY:
    st.error("서버용 GOOGLE_MAPS_API_KEY가 없습니다. .env 또는 .streamlit/secrets.toml에 설정하세요.")
    st.code('GOOGLE_MAPS_API_KEY="YOUR_SERVER_API_KEY"', language="toml")
    st.info("서버용 키에는 Places API (New)를 허용하세요. 위치 검색은 Places Text Search를 우선 사용하므로 Geocoding API는 선택 사항입니다.")
    st.stop()

if not BROWSER_API_KEY:
    st.warning(
        "GOOGLE_MAPS_BROWSER_API_KEY가 없어 지도는 OpenStreetMap으로 표시됩니다. "
        "시설 검색·분석은 계속 사용할 수 있습니다. Google 지도를 표시하려면 별도 브라우저용 키를 추가하세요."
    )
with st.expander("Google 지도 / 위치 검색이 실패할 때 확인할 항목", expanded=False):
    st.markdown(
        """
- 서버용 키: **Places API (New)** 활성화 및 API 제한 허용 여부 확인
- 서버용 키를 IP 제한했다면 Streamlit 실행 PC/서버의 공인 IP 허용
- 브라우저용 키: **Maps JavaScript API** 활성화 및 HTTP referrer에 `http://localhost:8501/*`, `http://127.0.0.1:8501/*` 등록
- 배포 중이면 실제 서비스 도메인도 브라우저용 키의 허용 referrer에 추가
- 서버용 키와 브라우저용 키는 반드시 분리
- 키/권한 변경 후 몇 분 기다린 뒤 Streamlit 재시작
        """
    )

if not ss["loaded"]:
    st.info("왼쪽 ①에서 서울시청, 강남역 등 분석할 중심 위치를 검색하고 Google 시설 데이터를 수집하세요.")

st.markdown(
    f"**분석 중심:** {escape(center['label'])}  ·  **주소:** {escape(center.get('address') or '주소 정보 없음')}  ·  "
    f"**좌표:** `{center['lat']:.5f}, {center['lng']:.5f}`  ·  **분석 반경:** `{center['radius_km']:.1f} km`"
)

params = {
    "grid_n": grid_n,
    "decay": decay,
    "aggregate": aggregate,
    "base_pop": float(base_pop),
    "concentration": float(concentration),
    "noise": float(noise),
    "seed": int(seed),
    "demand_rates": demand_rates,
    "capacity_scale": capacity_scale,
    "active_types": active_types,
}
base_result = scenario_result = None
view_metric = st.selectbox("비교할 지표", ["종합", *active_types] if active_types else ["종합"], key="metric_view")
view_col = "종합 부하율" if view_metric == "종합" else f"부하율_{view_metric}"

if ss["loaded"]:
    if not active_types:
        st.warning("분석 대상 시설 유형을 하나 이상 선택하세요.")
    else:
        base_result = analyze_zones(facilities_to_df(base_facilities), center, params)
        scenario_result = analyze_zones(facilities_to_df(scenario_facilities), center, params)

        before = summarize(base_result, view_col, overload_threshold, critical_threshold)
        after = summarize(scenario_result, view_col, overload_threshold, critical_threshold)

        st.subheader("② 시설 추가 전 · 후 과부하 비교")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Google 실제 시설", f"{len(base_facilities)}곳")
        c2.metric("신규 시뮬레이션", f"+{len(sim_facilities)}곳")
        c3.metric("평균 부하율(OI)", f"{before['avg']:.2f} → {after['avg']:.2f}", f"{after['avg'] - before['avg']:+.2f}", delta_color="inverse")
        c4.metric("과부하 구역", f"{before['over']} → {after['over']}", f"{after['over'] - before['over']:+d}", delta_color="inverse")
        c5.metric("사각지대 구역", f"{before['gap']} → {after['gap']}", f"{after['gap'] - before['gap']:+d}", delta_color="inverse")

        if sim_facilities:
            improvement = (before["avg"] - after["avg"]) / max(before["avg"], EPS) * 100
            st.success(
                f"신규 시설 {len(sim_facilities)}곳 추가 결과: 평균 부하율이 {improvement:.1f}% "
                f"{'감소' if improvement >= 0 else '증가'}했습니다. (동일한 인구·수요·격자 조건 기준)"
            )
        else:
            st.info("아직 신규 시설이 없습니다. 지도에서 위치를 클릭한 뒤 왼쪽 ③에서 시설을 추가하면 변화가 즉시 계산됩니다.")

        # 구역별 전/후 지표 비교
        before_df = base_result["df"][["구역", view_col]].rename(columns={view_col: "추가 전 OI"})
        after_df = scenario_result["df"][["구역", view_col]].rename(columns={view_col: "추가 후 OI"})
        compare_df = before_df.merge(after_df, on="구역", how="inner")
        compare_df["개선량(전-후)"] = compare_df["추가 전 OI"] - compare_df["추가 후 OI"]
        compare_df["추가 전 상태"] = classify_array(compare_df["추가 전 OI"], overload_threshold, critical_threshold)
        compare_df["추가 후 상태"] = classify_array(compare_df["추가 후 OI"], overload_threshold, critical_threshold)

        col_chart, col_status = st.columns([1.55, 1])
        with col_chart:
            st.subheader("📊 평균 부하율 비교")
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(name="시설 추가 전", x=["평균 OI"], y=[before["avg"]], marker_color="#78909C"))
            fig_compare.add_trace(go.Bar(name="시설 추가 후", x=["평균 OI"], y=[after["avg"]], marker_color="#2E8B75"))
            fig_compare.update_layout(barmode="group", height=310, margin=dict(l=10, r=10, t=25, b=10),
                                      yaxis_title="Overload Index (수요 ÷ 공급)", legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig_compare, use_container_width=True)
        with col_status:
            st.subheader("구역 상태 변화")
            status_df = pd.DataFrame({
                "상태": ["여유", "보통", "과부하", "심각"],
                "추가 전": [int((classify_array(compare_df["추가 전 OI"], overload_threshold, critical_threshold) == s).sum()) for s in ["여유", "보통", "과부하", "심각"]],
                "추가 후": [int((classify_array(compare_df["추가 후 OI"], overload_threshold, critical_threshold) == s).sum()) for s in ["여유", "보통", "과부하", "심각"]],
            })
            st.dataframe(status_df, hide_index=True, use_container_width=True)

        no_google_type = [t for t in active_types if not any(f["type"] == t for f in base_facilities)]
        if no_google_type:
            st.warning("Google 기준 데이터에 시설이 없는 유형: " + ", ".join(no_google_type) + ". 기준 분석에서는 해당 유형의 공급이 0으로 계산됩니다.")

# -----------------------------------------------------------------------------
# 6. 지도와 지도 클릭 좌표 수집
# -----------------------------------------------------------------------------
map_facilities = scenario_facilities if map_scenario == "추가 후(시나리오)" else base_facilities
map_result = scenario_result if map_scenario == "추가 후(시나리오)" else base_result
map_col = "종합 부하율" if view_metric == "종합" else f"부하율_{view_metric}"

left_map, right_info = st.columns([2.0, 1.0])
with left_map:
    st.subheader("🗺️ 지도 — 마커에 마우스를 올리면 실제 시설명을 확인할 수 있습니다")
    st.caption("Google Maps 브라우저 키가 정상 작동하면 Google 지도가 표시됩니다. Google 지도가 뜨지 않으면 레이어 메뉴에서 OpenStreetMap을 선택할 수 있습니다.")
    fmap = folium.Map(location=ss["view_center"], zoom_start=ss["view_zoom"], tiles=None, control_scale=True)
    # GoogleMutant가 실패해도 지도가 빈 화면이 되지 않도록 OSM을 기본 바탕으로 먼저 추가한다.
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap (대체 지도)",
        control=True,
    ).add_to(fmap)

    if BROWSER_API_KEY:
        browser_key_html = escape(BROWSER_API_KEY, quote=True)
        fmap.get_root().header.add_child(
            folium.Element(
                f'<script src="https://maps.googleapis.com/maps/api/js?key={browser_key_html}&v=weekly"></script>'
            ),
            name="google_maps_javascript_api",
        )
        fmap.get_root().header.add_child(
            folium.Element(
                '<script src="https://unpkg.com/leaflet.gridlayer.googlemutant@0.16.0/dist/Leaflet.GoogleMutant.js"></script>'
            ),
            name="leaflet_google_mutant",
        )

        class GoogleMapsBaseLayer(MacroElement):
            _template = Template(
                """{% macro script(this, kwargs) %}
                if (L.gridLayer && L.gridLayer.googleMutant && window.google && window.google.maps) {
                    L.gridLayer.googleMutant({type: 'roadmap'}).addTo({{ this._parent.get_name() }});
                } else {
                    console.warn('Google Maps unavailable; OpenStreetMap fallback remains visible.');
                }
                {% endmacro %}"""
            )

        fmap.add_child(GoogleMapsBaseLayer())

    if show_zones and map_result is not None:
        grid = map_result["grid"]
        zdf = map_result["df"]
        statuses = classify_array(zdf[map_col], overload_threshold, critical_threshold)
        for record, status in zip(zdf.to_dict("records"), statuses):
            color = STATUS_COLORS[status]
            tooltip_html = (
                f"<b>{escape(str(record['구역']))}</b><br>"
                f"{escape(view_metric)} OI: <b>{record[map_col]:.2f}</b> ({status})<br>"
                f"추정 인구: {record['추정인구']:,.0f}명"
            )
            folium.Rectangle(
                bounds=[
                    [record["위도"] - grid["half_lat"], record["경도"] - grid["half_lng"]],
                    [record["위도"] + grid["half_lat"], record["경도"] + grid["half_lng"]],
                ],
                color=color, weight=0.8, fill=True, fill_color=color, fill_opacity=0.28,
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
            ).add_to(fmap)

    folium.Circle(
        [center["lat"], center["lng"]], radius=center["radius_km"] * 1000,
        color="#333333", weight=1.5, dash_array="6", fill=False,
        tooltip=f"분석 반경 {center['radius_km']:.1f}km",
    ).add_to(fmap)
    folium.Marker(
        [center["lat"], center["lng"]], tooltip=f"분석 중심: {escape(center['label'])}",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(fmap)

    marker_cluster = MarkerCluster(name="시설 마커").add_to(fmap)
    for facility in map_facilities:
        cfg = FACILITY_TYPES[facility["type"]]
        origin_label = "Google Places 실제 등록 정보" if facility["source"] == "google" else "사용자 시뮬레이션"
        popup_html = (
            f"<b>{escape(facility['name'])}</b><br>분석 유형: {escape(facility['type'])}<br>"
            f"Google primaryType: {escape(facility.get('google_primary_type') or '사용자 지정')}<br>"
            f"출처: {origin_label}<br>시뮬레이션 용량: {facility['capacity']}<br>"
            f"모델 반경: {facility['radius_km']:.1f}km"
        )
        marker_color = cfg["color"] if facility["source"] == "google" else "purple"
        icon_name = cfg["icon"] if facility["source"] == "google" else "plus"
        folium.Marker(
            [facility["lat"], facility["lng"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=folium.Tooltip(f"{escape(facility['name'])} ({facility['type']})", sticky=True),
            icon=folium.Icon(color=marker_color, icon=icon_name, prefix="fa"),
        ).add_to(marker_cluster)
        if show_coverage:
            folium.Circle(
                [facility["lat"], facility["lng"]], radius=facility["radius_km"] * 1000,
                color=cfg["hex"], weight=1, fill=True, fill_opacity=0.035,
            ).add_to(fmap)

    if ss.get("clicked"):
        folium.Marker(
            [ss["clicked"]["lat"], ss["clicked"]["lng"]], tooltip="신규 시설 위치 후보",
            icon=folium.Icon(color="purple", icon="map-marker", prefix="fa"),
        ).add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    map_data = st_folium(
        fmap, key="infrastructure_map", height=610, use_container_width=True,
        returned_objects=["last_clicked", "center", "zoom"],
    )
    if map_data:
        # 먼저 지도 뷰를 저장한 뒤 클릭 이벤트를 처리해야 클릭 시에도 이동/확대 상태가 보존된다.
        if map_data.get("center"):
            ss["view_center"] = [map_data["center"]["lat"], map_data["center"]["lng"]]
        if map_data.get("zoom"):
            ss["view_zoom"] = int(map_data["zoom"])
        raw_click = map_data.get("last_clicked")
        if raw_click and "lat" in raw_click and "lng" in raw_click:
            token = f"{float(raw_click['lat']):.6f},{float(raw_click['lng']):.6f}"
            if token != ss.get("last_click_token"):
                ss["last_click_token"] = token
                ss["clicked"] = {"lat": float(raw_click["lat"]), "lng": float(raw_click["lng"])}
                st.rerun()

with right_info:
    st.subheader("📍 지도 선택 위치")
    clicked = ss.get("clicked")
    if clicked:
        st.write(f"위도: `{clicked['lat']:.6f}`")
        st.write(f"경도: `{clicked['lng']:.6f}`")
        if st.button("선택 지점을 새 분석 중심으로 검색", use_container_width=True):
            try:
                try:
                    addr = reverse_geocode(round(clicked["lat"], 6), round(clicked["lng"], 6), API_KEY)
                except Exception:
                    addr = None  # 역지오코딩은 선택 기능이므로 실패해도 시설 수집을 계속한다.
                facilities, errors = collect_facilities(
                    clicked["lat"], clicked["lng"], int(center["radius_km"] * 1000),
                    list(ss.get("search_types", list(FACILITY_TYPES))), API_KEY,
                )
                if not facilities and errors:
                    for err in errors:
                        notify("warning", err)
                    notify("error", "새 중심의 시설 수집에 실패해 기존 분석을 유지합니다.")
                else:
                    ss["center"] = {
                        "lat": clicked["lat"], "lng": clicked["lng"],
                        "label": addr or f"지도 선택 ({clicked['lat']:.5f}, {clicked['lng']:.5f})", "address": addr or "",
                        "radius_km": center["radius_km"],
                    }
                    ss["base_facilities"] = facilities
                    ss["sim_facilities"] = []
                    ss["loaded"] = True
                    ss["view_center"] = [clicked["lat"], clicked["lng"]]
                    ss["view_zoom"] = 13
                    ss["clicked"] = None
                    for err in errors:
                        notify("warning", err)
                    notify("success", f"새 중심 주변 시설 {len(facilities)}곳을 수집했습니다.")
                st.rerun()
            except Exception as exc:
                st.error(f"선택 위치 조회 실패: {mask_api_key(exc)[:400]}")
    else:
        st.caption("지도 빈 곳을 클릭하면 좌표가 저장됩니다. 왼쪽 ③에서 '지도 클릭 지점'을 선택해 시설을 추가하세요.")

    st.subheader("🚦 부하율 범례")
    st.markdown(
        f"🟢 여유: OI < {0.7 * overload_threshold:.2f}  \n"
        f"🟡 보통: {0.7 * overload_threshold:.2f} ~ {overload_threshold:.2f}  \n"
        f"🟠 과부하: {overload_threshold:.2f} ~ {critical_threshold:.2f}  \n"
        f"🔴 심각: OI ≥ {critical_threshold:.2f}"
    )
    st.caption("OI = 추정 수요 ÷ 모델상 공급. 값이 1보다 크면 수요가 공급을 초과하는 상태로 해석합니다.")

# -----------------------------------------------------------------------------
# 7. 상세 탭 및 다운로드
# -----------------------------------------------------------------------------
st.divider()
tab_facilities, tab_zones, tab_heat, tab_method = st.tabs(
    ["🏢 시설 데이터", "🧮 구역별 전/후 비교", "🔥 부하율 히트맵", "ℹ️ 계산 가정 및 한계"]
)

with tab_facilities:
    all_facilities = base_facilities + sim_facilities
    if all_facilities:
        display_df = facilities_to_df(all_facilities).copy()
        display_df["출처"] = display_df["source"].map({"google": "Google Places 실제 등록", "user": "사용자 시뮬레이션"})
        display_df = display_df.rename(columns={
            "name": "시설명", "type": "시설 유형", "lat": "위도", "lng": "경도",
            "capacity": "모델 용량", "radius_km": "모델 반경(km)",
            "google_primary_type": "Google primaryType", "google_types": "Google 유형 목록",
        })
        st.markdown("**Google Maps 장소 정보** — 아래의 Google 시설명 및 유형은 현재 검색 응답에서 불러온 데이터입니다.")
        st.dataframe(
            display_df[["시설명", "시설 유형", "Google primaryType", "Google 유형 목록", "위도", "경도", "모델 용량", "모델 반경(km)", "출처"]],
            use_container_width=True, hide_index=True,
        )
        st.download_button(
            "⬇️ 시설 데이터 CSV 다운로드",
            display_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="facility_data.csv", mime="text/csv",
        )
    else:
        st.info("표시할 시설 데이터가 없습니다.")

with tab_zones:
    if base_result is not None and scenario_result is not None:
        show_df = compare_df.sort_values("개선량(전-후)", ascending=False).copy()
        st.caption("개선량이 양수이면 신규 시설 추가 후 부하율이 감소한 구역입니다. 음수이면 부하율이 증가했습니다.")
        st.dataframe(show_df.round(3), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ 구역별 전/후 비교 CSV 다운로드",
            show_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="zone_before_after_comparison.csv", mime="text/csv",
        )
        worst = compare_df.nlargest(5, "추가 후 OI")[["구역", "추가 전 OI", "추가 후 OI", "개선량(전-후)"]]
        st.markdown("**신규 시설 추가 후에도 부하율이 높은 구역 TOP 5**")
        st.dataframe(worst.round(3), use_container_width=True, hide_index=True)
    else:
        st.info("지역 검색과 시설 수집을 완료하면 구역별 비교표가 표시됩니다.")

with tab_heat:
    if scenario_result is not None:
        heat = scenario_result["grid"]
        heat_df = scenario_result["df"]
        z = np.full((heat["n"], heat["n"]), np.nan)
        z[heat_df["행"].to_numpy(), heat_df["열"].to_numpy()] = heat_df[map_col].to_numpy()
        max_value = max(float(np.nanmax(z)), critical_threshold * 1.2, 0.1)
        heat_fig = go.Figure(go.Heatmap(
            z=z, x=np.round(heat["lngs"], 5), y=np.round(heat["lats"], 5),
            zmin=0, zmax=max_value, colorscale="RdYlGn_r", colorbar={"title": "OI"},
            hovertemplate="경도 %{x}<br>위도 %{y}<br>부하율 %{z:.2f}<extra></extra>",
        ))
        heat_fig.update_layout(
            title=f"시설 추가 후 {view_metric} 부하율", height=560,
            xaxis_title="경도", yaxis_title="위도", margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(heat_fig, use_container_width=True)
        if base_result is not None:
            bdf = base_result["df"]
            z_before = np.full((heat["n"], heat["n"]), np.nan)
            z_before[bdf["행"].to_numpy(), bdf["열"].to_numpy()] = bdf[map_col].to_numpy()
            delta_z = z_before - z
            delta_fig = go.Figure(go.Heatmap(
                z=delta_z, x=np.round(heat["lngs"], 5),
                y=np.round(heat["lats"], 5), colorscale="RdYlGn", colorbar={"title": "OI 감소량"},
                hovertemplate="경도 %{x}<br>위도 %{y}<br>개선량 %{z:.2f}<extra></extra>",
            ))
            delta_fig.update_layout(title="신규 시설 추가에 따른 부하율 개선량 (전 - 후)", height=430,
                                    xaxis_title="경도", yaxis_title="위도", margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(delta_fig, use_container_width=True)
    else:
        st.info("분석 데이터가 아직 없습니다.")

with tab_method:
    st.markdown(
        """
**데이터 출처 구분**
- **Google Places 실제 시설:** Google Places API (New) Nearby Search가 반환한 장소 ID, 시설명, 좌표 및 유형을 사용합니다.
- **사용자 시뮬레이션 시설:** 사용자가 지정한 이름·좌표·유형으로 생성한 가상 시설입니다. Google Maps에 실제 등록된 시설로 취급하지 않습니다.

**계산 과정**
1. 분석 중심으로부터 지정 반경 내에 N×N 격자를 구성합니다.
2. 인구는 `기본 추정 인구 × (1 + 중심 집중도 × exp(-(중심거리/σ)²)) × 로그정규 잡음`으로 추정합니다. 여기서 σ는 분석 반경의 절반입니다.
3. 유형별 수요는 `추정 인구 × 유형별 수요율`로 계산합니다.
4. 유형별 공급은 `Σ(시설 모델 용량 × 용량 배율 × 거리 가중치)`로 계산합니다.
5. 부하율(Overload Index)은 `수요 ÷ 공급`이며, 공급이 0인 경우 5.0으로 상한 처리합니다.
6. **추가 전**은 Google Places 기준 시설만, **추가 후**는 Google Places 기준 시설 + 사용자 시뮬레이션 시설로 계산합니다. 전/후에 동일한 격자·수요·난수 시드를 적용합니다.

**해석 시 주의사항**
- Places API는 시설의 실제 병상 수, 인력 수, 출동 가능 대수, 이용량 또는 행정구역 인구를 제공하지 않습니다.
- 따라서 시설 용량, 서비스 반경, 수요율 및 인구는 사용자가 조정할 수 있는 모델 가정입니다. OI는 실제 행정기관의 공식 과부하율이나 실시간 혼잡도가 아닙니다.
- Places Nearby Search는 시설 유형별 최대 20개를 반환합니다. 검색 반경·유형·Google API의 제공 범위에 따라 누락될 수 있습니다.
- 지도 바탕은 Google Maps JavaScript API를 사용합니다. Google Places 콘텐츠를 다른 지도 바탕 위에 혼합하지 않도록 구성했습니다. 브라우저 키는 페이지 소스에 노출되므로 HTTP referrer와 Maps JavaScript API로 제한해야 합니다.
- Places 콘텐츠의 저장·캐싱·파생 데이터 사용에는 Google Maps Platform 약관 제한이 적용됩니다. 이 앱은 서버 캐시/파일 DB에 Places 응답을 저장하지 않고 현재 Streamlit 세션에서만 표시하지만, 좌표로 부하 지표를 산출하는 사용 방식은 실제 배포 전에 프로젝트의 약관 및 사용권 범위를 별도로 확인해야 합니다.
- 실제 서비스 배포 전에는 Google 요구사항에 맞는 공개 이용약관 및 개인정보처리방침 링크를 앱 사용자에게 제공해야 합니다.
"""
    )

st.caption("서버용 API 키는 서버 측에서만 사용합니다. 지도 표시용 브라우저 키는 Maps JavaScript API 로드를 위해 브라우저에 전달되므로 반드시 별도 발급하고 HTTP referrer 제한을 적용하세요.")
