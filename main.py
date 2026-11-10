import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 1. 페이지 기본 설정 (브라우저 탭 제목 및 아이콘)
st.set_page_config(
    page_title="주변 화장실",
    page_icon="🚽",
    layout="wide"
)

# 2. 메인 화면 타이틀
st.title("🚽 주변 화장실 찾기")
st.caption("경기도 남양주시 공중화장실 위치 기반 안내 서비스")

# 3. 데이터 파일 로드 함수
DEFAULT_FILE = "경기도 남양주시_공중화장실 현황_20260126.csv"

st.sidebar.header("📁 데이터 설정")
uploaded_file = st.sidebar.file_uploader(
    "CSV 파일 업로드 (선택사항)", 
    type=['csv'],
    help="파일이 프로젝트 폴더에 없거나 새로 업로드하려면 여기에 올려주세요."
)

@st.cache_data
def load_data_from_path(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='cp949')
        return df
    except Exception:
        return None

@st.cache_data
def load_data_from_upload(file):
    try:
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding='cp949')
        return df
    except Exception:
        return None

# 데이터 로드
df = None
if uploaded_file is not None:
    df = load_data_from_upload(uploaded_file)
else:
    df = load_data_from_path(DEFAULT_FILE)

# 데이터 예외 처리 및 기본 전처리
if df is None or df.empty:
    st.error(f"⚠️ 데이터 파일을 불러올 수 없습니다.")
    st.info(f"현재 폴더에 `{DEFAULT_FILE}` 파일이 없으면, 왼쪽 사이드바의 **파일 업로드** 버튼으로 CSV 파일을 등록해 주세요.")
else:
    # 위도, 경도 컬럼 숫자형 변환 및 결측치 처리
    df['위도'] = pd.to_numeric(df['위도'], errors='coerce')
    df['경도'] = pd.to_numeric(df['경도'], errors='coerce')
    df = df.dropna(subset=['위도', '경도'])

    # 4. 기준 위치 입력 및 필터 설정
    st.sidebar.markdown("---")
    st.sidebar.header("📍 기준 위치 입력")
    
    # 남양주시청 기준 기본 위치
    default_lat = 37.6360
    default_lon = 127.2165
    
    user_lat = st.sidebar.number_input("위도 (Latitude)", value=default_lat, format="%.6f")
    user_lon = st.sidebar.number_input("경도 (Longitude)", value=default_lon, format="%.6f")
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 추가 필터")
    selected_gu = st.sidebar.selectbox("구분 선택", ["전체"] + list(df['구분'].unique()))
    only_bell = st.sidebar.checkbox("비상벨 설치 화장실만 보기")
    only_baby = st.sidebar.checkbox("기저귀교환대 있는 화장실만 보기")

    # 5. 조건에 따른 데이터 필터링
    filtered_df = df.copy()
    if selected_gu != "전체":
        filtered_df = filtered_df[filtered_df['구분'] == selected_gu]
    if only_bell:
        filtered_df = filtered_df[filtered_df['비상벨설치여부'] == 'Y']
    if only_baby:
        filtered_df = filtered_df[filtered_df['기저귀교환대유무'] == 'Y']

    if filtered_df.empty:
        st.warning("조건에 맞는 화장실이 없습니다. 사이드바의 필터를 조정해 보세요.")
    else:
        # 6. 거리 계산 (단순 유클리드 거리 기준) 및 상위 10개 추출
        filtered_df['거리차'] = ((filtered_df['위도'] - user_lat) ** 2 + (filtered_df['경도'] - user_lon) ** 2) ** 0.5
        nearest_df = filtered_df.sort_values(by='거리차').head(10).copy()
        nearest_df['대략적거리(km)'] = (nearest_df['거리차'] * 111).round(2)

        st.subheader(f"📍 입력 위치 ({user_lat:.4f}, {user_lon:.4f})에서 가장 가까운 화장실 TOP 10")

        # 표로 출력할 항목 정리
        display_columns = [
            '화장실명', '소재지도로명주소', '대략적거리(km)', 
            '개방시간', '비상벨설치여부', '기저귀교환대유무', '전화번호', '관리기관명'
        ]
        available_cols = [col for col in display_columns if col in nearest_df.columns]
        st.dataframe(nearest_df[available_cols].reset_index(drop=True), use_container_width=True)

        # 7. 지도 시각화 (px.scatter_map 하위호환성 호환 처리)
        st.subheader("🗺️ 주변 화장실 지도 위치")
        
        user_point = pd.DataFrame({
            '화장실명': ['[기준 위치]'],
            '위도': [user_lat],
            '경도': [user_lon],
            '대략적거리(km)': [0],
            '소재지도로명주소': ['입력하신 기준 좌표']
        })
        
        map_df = pd.concat([nearest_df, user_point], ignore_index=True)

        # 최신 Plotly에서는 scatter_map, 이전 버전에서는 scatter_mapbox 호환 사용
        if hasattr(px, "scatter_map"):
            fig = px.scatter_map(
                map_df,
                lat="위도",
                lon="경도",
                hover_name="화장실명",
                hover_data=["소재지도로명주소", "대략적거리(km)"],
                zoom=12,
                height=500
            )
            fig.update_layout(map_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        else:
            fig = px.scatter_mapbox(
                map_df,
                lat="위도",
                lon="경도",
                hover_name="화장실명",
                hover_data=["소재지도로명주소", "대략적거리(km)"],
                zoom=12,
                height=500
            )
            fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        
        fig.update_traces(marker=dict(size=12, color='red'))
        st.plotly_chart(fig, use_container_width=True)
