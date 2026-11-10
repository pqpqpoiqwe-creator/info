import streamlit as st
import pandas as pd
import plotly.express as px

# 1. 페이지 설정 (브라우저 탭 제목 및 아이콘)
st.set_page_config(
    page_title="주변 화장실",
    page_icon="🚽",
    layout="wide"
)

# 2. 메인 화면 제목 및 아이콘
st.title("🚽 주변 화장실 찾기")
st.caption("경기도 남양주시 공중화장실 위치 기반 안내 서비스")

# 3. 데이터 로드 함수 (캐싱 적용)
DATA_PATH = "/content/drive/MyDrive/경기도 남양주시_공중화장실 현황_20260126.csv"

@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # 위도/경도 숫자형 변환
        df['위도'] = pd.to_numeric(df['위도'], errors='coerce')
        df['경도'] = pd.to_numeric(df['경도'], errors='coerce')
        # 좌표가 없는 데이터 제외
        df = df.dropna(subset=['위도', '경도'])
        return df
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

df = load_data(DATA_PATH)

if df.empty:
    st.warning("데이터가 비어 있거나 파일을 불러올 수 없습니다. 경로를 확인해주세요.")
else:
    # 4. 사이드바 - 사용자 위치 입력 및 필터
    st.sidebar.header("📍 기준 위치 입력")
    st.sidebar.write("찾고자 하는 위치의 위도와 경도를 입력하세요.")
    
    # 남양주시청 근처 기본 좌표 설정
    default_lat = 37.6360
    default_lon = 127.2165
    
    user_lat = st.sidebar.number_input("위도 (Latitude)", value=default_lat, format="%f")
    user_lon = st.sidebar.number_input("경도 (Longitude)", value=default_lon, format="%f")
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 추가 필터")
    selected_gu = st.sidebar.selectbox("구분 선택", ["전체"] + list(df['구분'].unique()))
    only_bell = st.sidebar.checkbox("비상벨 설치 화장실만 보기")
    only_baby = st.sidebar.checkbox("기저귀교환대 있는 화장실만 보기")

    # 5. 필터 적용
    filtered_df = df.copy()
    if selected_gu != "전체":
        filtered_df = filtered_df[filtered_df['구분'] == selected_gu]
    if only_bell:
        filtered_df = filtered_df[filtered_df['비상벨설치여부'] == 'Y']
    if only_baby:
        filtered_df = filtered_df[filtered_df['기저귀교환대유무'] == 'Y']

    if filtered_df.empty:
        st.warning("조건에 맞는 화장실이 없습니다. 필터를 조정해 보세요.")
    else:
        # 6. 거리 계산 (단순 유클리드 거리 기반 근사치 계산)
        # 1도(degree) 당 거리는 위/경도가 다르지만 가까운 거리 내 순위 비교용으로는 충분합니다.
        filtered_df['거리차'] = ((filtered_df['위도'] - user_lat) ** 2 + (filtered_df['경도'] - user_lon) ** 2) ** 0.5
        
        # 가까운 순서대로 상위 10개 추출
        nearest_df = filtered_df.sort_values(by='거리차').head(10).copy()
        
        # 대략적인 거리(km) 안내용 컬럼 생성 (위도 1도 약 111km 기준 단순 산식)
        nearest_df['대략적거리(km)'] = (nearest_df['거리차'] * 111).round(2)

        st.subheader(f"📍 입력하신 위치($\{user_lat:.4f}, \{user_lon:.4f}\$)에서 가장 가까운 화장실 TOP 10")

        # 7. 표 출력용 컬럼 정리
        display_columns = [
            '화장실명', '소재지도로명주소', '대략적거리(km)', 
            '개방시간', '비상벨설치여부', '기저귀교환대유무', '전화번호', '관리기관명'
        ]
        
        # 존재하는 컬럼만 선택하여 표로 표시
        available_cols = [col for col in display_columns if col in nearest_df.columns]
        st.dataframe(nearest_df[available_cols].reset_index(drop=True), use_container_width=True)

        # 8. Plotly 지도 시각화
        st.subheader("🗺️ 주변 화장실 지도 위치")
        
        # 사용자가 입력한 위치도 시각화에 포함하기 위해 임시 행 추가
        user_point = pd.DataFrame({
            '화장실명': ['[기준 위치]'],
            '위도': [user_lat],
            '경도': [user_lon],
            '대략적거리(km)': [0],
            '소재지도로명주소': ['입력하신 기준 좌표']
        })
        
        # 지도에 표시할 데이터 합치기
        map_df = pd.concat([nearest_df, user_point], ignore_index=True)

        fig = px.scatter_mapbox(
            map_df,
            lat="위도",
            lon="경도",
            hover_name="화장실명",
            hover_data=["소재지도로명주소", "대략적거리(km)"],
            zoom=13,
            height=500
        )
        
        fig.update_traces(marker=dict(size=12, color='red'))
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        
        st.plotly_chart(fig, use_container_width=True)
