import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import haversine_distances
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# 1. 브라우저 탭 제목 및 아이콘 설정
st.set_page_config(
    page_title="주변 화장실",
    page_icon="🚻",
    layout="wide",
)

# 2. 화면 맨 위 앱 제목 및 어울리는 아이콘 표시
st.title("🚻 주변 화장실 및 개방 형태 머신러닝 분류기")
st.caption("현재 위치(위도, 경도)를 입력하면 가장 가까운 공중화장실 10곳을 찾고, 머신러닝(지도 학습)을 통해 개방 형태를 예측·분석합니다.")

DATA_URL = "https://raw.githubusercontent.com/pqpqpoiqwe-creator/info/refs/heads/main/bathroom.csv"


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
        help="기본적으로 GitHub 저장소의 bathroom.csv 데이터를 불러옵니다.",
    )

# 데이터 불러오기
try:
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        df = load_data(DATA_URL)
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
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

# Streamlit 내장 st.map을 활용한 지도 시각화
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

st.markdown("**🗺️ 주변 화장실 위치 지도**")
st.map(plot_df, latitude="위도", longitude="경도", size=30, zoom=13)

st.divider()

# 섹션 2: 머신러닝(지도 학습) 분류 모델을 활용한 개방 형태 예측 및 오분류표(Confusion Matrix)
st.subheader("🤖 지도 학습(Classification) 기반 개방 형태 예측 및 오분류표")
st.caption("화장실의 시설 특성(대변기수, 소변기수 등)을 바탕으로 로지스틱 회귀 모델을 학습시켜 항상 개방 여부를 예측하고 성능을 평가합니다.")

# 모델 학습용 특성(X)과 타겟(y) 준비 (전체 데이터 기준 학습)
feature_cols = [
    "남성용대변기수", "남성용소변기수", "남성용장애인용대변기수", 
    "여성용대변기수", "여성용장애인용대변기수"
]

# 결측치 처리 및 feature 추출
train_df = df.dropna(subset=feature_cols + ["개방분류"]).copy()
X_train = train_df[feature_cols]
y_train = train_df["개방분류"]

# 지도 학습 모델(Logistic Regression) 학습
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 가장 가까운 10개 화장실에 대해 모델 예측 수행
X_nearest = nearest_10[feature_cols].fillna(0)
nearest_10["예측개방분류"] = model.predict(X_nearest)

# 10개 화장실 기준 실제값 vs 예측값 오분류표(Confusion Matrix) 생성
labels = ["항상 개방 (상시)", "일시적 개방 (정시/불규칙)"]
cm = confusion_matrix(nearest_10["개방분류"], nearest_10["예측개방분류"], labels=labels)
acc = accuracy_score(nearest_10["개방분류"], nearest_10["예측개방분류"])

# 결과 출력 탭 구성
tab1, tab2, tab3 = st.tabs(["📋 주변 10곳 예측 결과 표", "📊 오분류표 (Confusion Matrix)", "📈 분류 요약 집계"])

with tab1:
    st.markdown("**가장 가까운 화장실 10곳의 실제 개방 형태 vs 머신러닝 예측 결과**")
    pred_table = nearest_10[
        ["순위", "화장실명", "개방분류", "예측개방분류", "개방시간상세", "관리기관명", "거리(km)"]
    ].rename(columns={
        "개방분류": "실제 개방 형태",
        "예측개방분류": "모델 예측 형태",
        "개방시간상세": "상세 시간"
    })
    st.dataframe(pred_table, use_container_width=True, hide_index=True)

with tab2:
    st.markdown(f"**🔍 오분류표 (Confusion Matrix)** — 모델 정확도: `{acc * 100:.1f}%`")
    cm_df = pd.DataFrame(
        cm,
        index=[f"실제: {l}" for l in labels],
        columns=[f"예측: {l}" for l in labels]
    )
    st.dataframe(cm_df, use_container_width=True)
    st.info("💡 오분류표는 머신러닝 모델이 화장실의 실제 개방 상태를 올바르게 맞추었는지 행(실제)과 열(예측) 기준으로 비교해 주는 표입니다.")

with tab3:
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
