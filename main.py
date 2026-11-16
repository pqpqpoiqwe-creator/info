import re
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import haversine_distances

# 1. 브라우저 탭 제목 및 아이콘 설정
st.set_page_config(
    page_title="주변 화장실",
    page_icon="🚻",
    layout="wide",
)

# 2. 화면 맨 위 앱 제목 및 어울리는 아이콘 표시
st.title("🚻 주변 화장실")
st.caption("현재 위치(위도, 경도)를 입력하면 가장 가까운 공중화장실 10곳의 위치, 현재 개방 여부, 그리고 지도 학습 기반 개방 형태 분류 및 오분류표를 보여줍니다.")

DATA_URL = "https://raw.githubusercontent.com/pqpqpoiqwe-creator/info/refs/heads/main/bathroom.csv"
KST = timezone(timedelta(hours=9))


def parse_time_range(text: str):
    """문자열에서 'HH:MM~HH:MM' 또는 'HH:MM~익일HH:MM' 시간 구간을 분 단위로 추출합니다."""
    m = re.search(r"(\d{1,2}):(\d{2})\s*~\s*(익일)?\s*(\d{1,2}):(\d{2})", text)
    if not m:
        return None
    sh, sm = int(m.group(1)), int(m.group(2))
    next_day = m.group(3) is not None
    eh, em = int(m.group(4)), int(m.group(5))
    start_min = sh * 60 + sm
    end_min = eh * 60 + em
    if next_day or end_min == 0:
        end_min += 24 * 60
    return start_min, end_min


def check_currently_open(open_type: str, detail: str, now_dt: datetime) -> str:
    """개방시간 구분과 상세 시간을 바탕으로 현재 시각(KST) 기준 개방 여부를 판별합니다."""
    open_type = str(open_type).strip()
    detail = str(detail).strip()

    if open_type == "상시":
        return "🟢 현재 개방 중"

    cur_min = now_dt.hour * 60 + now_dt.minute
    is_weekend = now_dt.weekday() >= 5  # 5: 토요일, 6: 일요일
    month = now_dt.month
    day = now_dt.day

    # 불규칙(계절·특정일 개방) 처리
    if open_type == "불규칙":
        if "7~8월" in detail:
            if month not in (7, 8):
                return "🔴 현재 미개방 (7~8월 개방)"
        elif "6월~10월" in detail:
            if not (6 <= month <= 10):
                return "🔴 현재 미개방 (6~10월 개방)"
            return "🟢 현재 개방 중 (행락철 기간)"
        elif "현충일" in detail:
            if not (month == 6 and day == 6):
                return "🔴 현재 미개방 (현충일 개방)"
            return "🟢 현재 개방 중 (현충일)"

    # 평일/주중/주말/공휴일 분기 처리
    target_text = detail
    if "평일" in detail or "주중" in detail:
        if is_weekend:
            if "미 개방" in detail or "미개방" in detail:
                return "🔴 현재 미개방 (주말 미개방)"
            m_wknd = re.search(
                r"(?:주말|토요일)[^0-9]*(\d{1,2}:\d{2}\s*~\s*(?:익일)?\s*\d{1,2}:\d{2})",
                detail,
            )
            if m_wknd:
                target_text = m_wknd.group(1)
            elif "~20:00(토요일)" in detail and now_dt.weekday() == 5:
                target_text = "06:00~20:00"
            elif "9:00~17:00 공휴일" in detail and now_dt.weekday() == 6:
                target_text = "09:00~17:00"
            else:
                return "🔴 현재 미개방 (평일 개방)"
        else:
            m_wkdy = re.search(
                r"(?:평일|주중)[^0-9]*(\d{1,2}:\d{2}\s*~\s*(?:익일)?\s*\d{1,2}:\d{2})",
                detail,
            )
            if m_wkdy:
                target_text = m_wkdy.group(1)

    tr = parse_time_range(target_text)
    if tr is None:
        return "⚪ 확인 불가 (상세시간 참조)"

    start_min, end_min = tr
    if end_min > 24 * 60:
        is_open = (cur_min >= start_min) or (cur_min < (end_min - 24 * 60))
    else:
        is_open = start_min <= cur_min < end_min

    return "🟢 현재 개방 중" if is_open else "🔴 현재 미개방"


@st.cache_data
def load_data(file_source) -> pd.DataFrame:
    """CSV 데이터를 불러와 위도·경도 결측치를 정제하고 개방 유형을 정리합니다."""
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

    # 실제 개방 유형 레이블 ('항상 개방' vs '일시적 개방')
    df["개방분류"] = df["개방시간"].apply(
        lambda x: "항상 개방" if str(x).strip() == "상시" else "일시적 개방"
    )
    df["개방시간상세"] = df["개방시간상세"].fillna("-").astype(str).str.strip()

    return df


# 현재 한국 표준시(KST) 기준 시각
now_kst = datetime.now(KST)
weekday_names = ["월", "화", "수", "목", "금", "토", "일"]
current_time_str = f"{now_kst.strftime('%Y-%m-%d')} ({weekday_names[now_kst.weekday()]}) {now_kst.strftime('%H:%M')} KST"

# 사이드바: 위도·경도 입력 및 기준 시각 표시
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
    st.subheader("⏰ 기준 시각")
    st.info(f"현재 시각: **{current_time_str}**")

# 데이터 불러오기
try:
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

# 현재 개방 여부 판별 컬럼 추가
nearest_10["현재개방여부"] = nearest_10.apply(
    lambda row: check_currently_open(row["개방시간"], row["개방시간상세"], now_kst),
    axis=1,
)

# 상단 요약 지표
open_now_cnt = int(nearest_10["현재개방여부"].str.startswith("🟢").sum())
always_open_cnt = int((nearest_10["개방분류"] == "항상 개방").sum())
temp_open_cnt = int((nearest_10["개방분류"] == "일시적 개방").sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("가장 가까운 화장실", f"{nearest_10.loc[0, '거리(km)']:.3f} km", nearest_10.loc[0, "화장실명"])
col2.metric("현재 개방 중 (10곳 중)", f"{open_now_cnt} 곳", f"{now_kst.strftime('%H:%M')} 기준")
col3.metric("항상 개방 (상시)", f"{always_open_cnt} 곳")
col4.metric("일시적 개방 (정시/불규칙)", f"{temp_open_cnt} 곳")

st.divider()

# 섹션 1: 주변 화장실 10개 위치 표 (현재 개방 여부 포함)
st.subheader("📋 주변 화장실 10곳 위치 및 현재 개방 여부 표")
st.caption(f"기준 시각: {current_time_str}")

location_table = nearest_10[
    ["순위", "화장실명", "구분", "현재개방여부", "개방시간상세", "주소", "위도", "경도", "거리(km)", "전화번호"]
].rename(
    columns={
        "현재개방여부": "현재 개방 여부",
        "개방시간상세": "개방 시간 안내",
    }
)
st.dataframe(location_table, use_container_width=True, hide_index=True)

st.divider()

# 섹션 2: 머신러닝(지도 학습) 분류 모델을 활용한 개방 형태 분류 표 및 오분류표
st.subheader("🤖 지도 학습(분류) 기반 항상 개방 vs 일시적 개방 표 및 오분류표")
st.caption("화장실 시설 특성(대변기수, 소변기수, 장애인용 변기수 등)으로 로지스틱 회귀(Logistic Regression) 지도 학습 모델을 학습시켜 항상 개방 여부를 분류합니다.")

feature_cols = [
    "남성용대변기수",
    "남성용소변기수",
    "남성용장애인용대변기수",
    "여성용대변기수",
    "여성용장애인용대변기수",
]

train_df = df.dropna(subset=feature_cols + ["개방분류"]).copy()
X_train = train_df[feature_cols].fillna(0)
y_train = train_df["개방분류"]

# 지도 학습 분류 모델 학습
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# 주변 10개 화장실 예측
X_nearest = nearest_10[feature_cols].fillna(0)
nearest_10["예측개방분류"] = clf.predict(X_nearest)

# 오분류표(Confusion Matrix) 계산 (주변 10개 기준)
labels = ["항상 개방", "일시적 개방"]
cm = confusion_matrix(nearest_10["개방분류"], nearest_10["예측개방분류"], labels=labels)
acc = accuracy_score(nearest_10["개방분류"], nearest_10["예측개방분류"])

# 2-1. 분류 결과 표
st.markdown("**1) 주변 10개 화장실 개방 형태 분류 표 (실제 vs 지도 학습 예측)**")
classification_table = nearest_10[
    ["순위", "화장실명", "개방분류", "예측개방분류", "현재개방여부", "개방시간", "개방시간상세", "거리(km)"]
].rename(
    columns={
        "개방분류": "실제 개방 분류",
        "예측개방분류": "모델 예측 분류",
        "현재개방여부": "현재 개방 여부",
        "개방시간": "원본 구분",
        "개방시간상세": "상세 개방시간",
    }
)
st.dataframe(classification_table, use_container_width=True, hide_index=True)

# 2-2. 오분류표 및 시각화
left_col, right_col = st.columns([1.1, 1])

with left_col:
    st.markdown(f"**2) 오분류표 (Confusion Matrix) — 정확도: `{acc * 100:.1f}%`**")
    cm_df = pd.DataFrame(
        cm,
        index=[f"실제: {lbl}" for lbl in labels],
        columns=[f"예측: {lbl}" for lbl in labels],
    )
    st.dataframe(cm_df, use_container_width=True)
    st.caption("행(실제 개방 형태)과 열(모델 예측 개방 형태)을 교차 비교한 오분류표입니다.")

with right_col:
    st.markdown("**3) 주변 10곳 현재 개방 현황 요약**")
    status_counts = (
        nearest_10["현재개방여부"]
        .value_counts()
        .reset_index()
    )
    status_counts.columns = ["현재 개방 상태", "화장실 수"]
    fig = px.bar(
        status_counts,
        x="현재 개방 상태",
        y="화장실 수",
        color="현재 개방 상태",
        text="화장실 수",
        height=260,
    )
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
