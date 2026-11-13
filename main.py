import os
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics.pairwise import haversine_distances
import numpy as np

# 1. 브라우저 탭 제목 및 아이콘 설정
st.set_page_config(
    page_title="주변 화장실",
    page_icon="🚻",
    layout="wide",
)

# 2. 화면 맨 위 앱 제목 및 어울리는 아이콘 표시
st.title("🚻 주변 화장실")
st.caption("현재 위치(위도, 경도)를 입력하면 가장 가까운 공중화장실 10곳의 위치와 개방 형태(상시/일시 개방)를 알려드립니다.")

CSV_FILENAME = "경기도 남양주시_공중화장실 현황_20260126.csv"


@st.cache_data
def load_data(file_source) -> pd.DataFrame:
    """CSV 데이터를 불러와 위도·경도 결측치를 정제하고 개방 유형을 분류합니다."""
    try:
        df = pd.read_csv(file_source, encoding="utf-8")
    except UnicodeDecodeError:
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        df = pd.read_csv(file_source, encoding="cp949")

    # 위도·경도 숫자 변환 및 결측치 제거
    df["위도"] = pd.to_numeric(df["위도"], errors="coerce")
    df["경도"] = pd.to_numeric(df["경도"], errors="coerce")
    df = df.dropna(subset=["위도", "경도"]).copy()

    # 주소 통합 컬럼 생성 (도로명주소 우선, 없으면 지번주소 사용)
    road_addr = df["소재지도로명주소"].fillna("").astype(str).str.strip()
    jibun_addr = df["소재지지번주소"].fillna("").astype(str).str.strip()
    df["주소"] = road_addr.where(road_addr != "", jibun_addr)
    df["주소"] = df["주소"].replace("", "주소 정보 없음")

    # 개방시간('상시' vs '정시'/'불규칙')을 기준으로 항상 개방 / 일시적 개방 분류
    df["개방분류"] = df["개방시간"].apply(
        lambda x: "항상 개방 (상시)" if str(x).strip() == "상시" else "일시적 개방 (정시/불규칙)"
    )
    df["개방시간상세"] = df["개방시간상세"].fillna("-").astype(str).str.strip()

    return df


# 사이드바: 데이터 로드 및 위도·경도 입력
with st.sidebar:
    st.header("📍 내 위치 설정")
    default_lat = 37.6360  # 남양주시청 부근 기본 위도
    default_lon = 127.2165  # 남양주시청 부근 기본 경도

    user_lat = st.number_input(
        "위도 (Latitude)",
        min_value=33.0,
        max_value=44.0,
        value=default_lat,
        step=0.0001,
        format="%.6f",
    )
    user_lon = st.number_input(
        "경도 (Longitude)",
        min_value=124.0,
        max_value=132.0,
        value=default_lon,
        step=0.0001,
        format="%.6f",
    )

    st.divider()
    st.subheader("📂 데이터 파일")
    uploaded_file = st.file_uploader(
        "CSV 파일 변경 (선택 사항)",
        type=["csv"],
        help=f"기본적으로 같은 폴더의 '{CSV_FILENAME}' 파일을 불러옵니다.",
    )

# 데이터 불러오기
if uploaded_file is not None:
    df = load_data(uploaded_file)
elif os.path.exists(CSV_FILENAME):
    df = load_data(CSV_FILENAME)
else:
    st.error(f"'{CSV_FILENAME}' 파일을 찾을 수 없습니다. 같은 폴더에 파일을 두거나 왼쪽 사이드바에서 CSV 파일을 업로드해 주세요.")
    st.stop()

# scikit-learn haversine_distances를 활용한 실제 지구 표면 거리(km) 계산
EARTH_RADIUS_KM = 6371.0
user_rad = np.radians([[user_lat, user_lon]])
toilets_rad = np.radians(df[["위도", "경도"]].to_numpy())
distances_km = haversine_distances(user_rad, toilets_rad)[0] * EARTH_RADIUS_KM

df["거리(km)"] = np.round(distances_km, 3)

# 가장 가까운 화장실 10개 추출
nearest_10 = df.sort_values("거리(km)", ascending=True).head(10).reset_index(drop=True)
nearest_10.insert(0, "순위", range(1, len(nearest_10) + 1))

# 상단 요약 지표
always_open_cnt = int((nearest_10["개방분류"] == "항상 개방 (상시)").sum())
temp_open_cnt = int((nearest_10["개방분류"] == "일시적 개방 (정시/불규칙)").sum())

col1, col2, col3 = st.columns(3)
col1.metric("가장 가까운 화장실 거리", f"{nearest_10.loc[0, '거리(km)']:.3f} km", nearest_10.loc[0, "화장실명"])
col2.metric("항상 개방 (주변 10곳 중)", f"{always_open_cnt} 곳")
col3.metric("일시적 개방 (주변 10곳 중)", f"{temp_open_cnt} 곳")

st.divider()

# 섹션 1: 주변 화장실 10개 위치 표 및 지도
st.subheader("📋 주변 화장실 10곳 위치 표")

location_table = nearest_10[
    ["순위", "화장실명", "구분", "주소", "위도", "경도", "거리(km)", "전화번호"]
]
st.dataframe(location_table, use_container_width=True, hide_index=True)

# Plotly 지도 시각화 (내 위치 + 주변 화장실 10곳)
map_df = nearest_10[["화장실명", "위도", "경도", "거리(km)", "개방분류", "주소"]].copy()
user_point = pd.DataFrame(
    [
        {
            "화장실명": "📍 현재 입력 위치",
            "위도": user_lat,
            "경도": user_lon,
            "거리(km)": 0.0,
            "개방분류": "내 위치",
            "주소": f"위도 {user_lat:.6f}, 경도 {user_lon:.6f}",
        }
    ]
)
plot_df = pd.concat([user_point, map_df], ignore_index=True)

fig = px.scatter_mapbox(
    plot_df,
    lat="위도",
    lon="경도",
    color="개방분류",
    hover_name="화장실명",
    hover_data={"주소": True, "거리(km)": True, "위도": ":.5f", "경도": ":.5f"},
    zoom=13,
    height=420,
    color_discrete_map={
        "내 위치": "#E63946",
        "항상 개방 (상시)": "#2A9D8F",
        "일시적 개방 (정시/불규칙)": "#F4A261",
    },
)
fig.update_layout(
    mapbox_style="open-street-map",
    margin={"r": 0, "t": 10, "l": 0, "b": 0},
    legend_title_text="구분",
)
fig.update_traces(marker=dict(size=13))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# 섹션 2: 분류를 사용한 항상 개방 vs 일시적 개방 표
st.subheader("🕒 개방 형태 분류 표 (항상 개방 vs 일시적 개방)")

open_filter = st.radio(
    "개방 분류 필터",
    options=["전체 (10곳)", "항상 개방 (상시)", "일시적 개방 (정시/불규칙)"],
    horizontal=True,
)

classification_table = nearest_10[
    ["순위", "화장실명", "개방분류", "개방시간", "개방시간상세", "관리기관명", "거리(km)"]
].rename(
    columns={
        "개방분류": "개방 분류 (항상/일시)",
        "개방시간": "원본 개방구분",
        "개방시간상세": "상세 개방시간",
    }
)

if open_filter != "전체 (10곳)":
    classification_table = classification_table[
        classification_table["개방 분류 (항상/일시)"] == open_filter
    ]

st.dataframe(classification_table, use_container_width=True, hide_index=True)

# 분류별 요약 집계표
summary_df = (
    nearest_10.groupby("개방분류", as_index=False)
    .agg(
        화장실_수=("화장실명", "count"),
        평균_거리_km=("거리(km)", lambda s: round(s.mean(), 3)),
        화장실_목록=("화장실명", lambda s: ", ".join(s)),
    )
    .rename(
        columns={
            "개방분류": "개방 분류",
            "화장실_수": "개소 수",
            "평균_거리_km": "평균 거리(km)",
            "화장실_목록": "해당 화장실 목록",
        }
    )
)
st.markdown("**개방 분류별 요약**")
st.dataframe(summary_df, use_container_width=True, hide_index=True)
