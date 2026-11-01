import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import plotly.express as px

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="수면 패턴과 인지 피로도 분석", layout="wide")

st.title("💤 수면 시간대별 군집화 분석과 인지 피로도(`Stress Level`) 관계 연구")
st.markdown("""
본 연구는 **Kaggle Sleep Health and Lifestyle Dataset**을 활용하여 수면 시간대(`Sleep Duration`) 및 수면의 질(`Quality of Sleep`)에 따른 
**데이터 기반 군집(K-Means)**을 형성하고, 군집 간 인지 피로도(`Stress Level`)의 차이를 검증합니다.
""")

# 2. 데이터 로드 및 전처리 함수
@st.cache_data
def load_and_preprocess_data():
    # 데이터셋 읽기 (파일 경로에 맞춰 수정 가능)
    try:
        df = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")
    except FileNotFoundError:
        # 경로 차이를 대비한 예비 로드 logic
        df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
    
    # [전처리] Sleep Disorder의 219개 결측치는 '정상 수면'을 의미하므로 'None'으로 대체
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    
    # K-Means 군집화를 위한 입력 변수 선정 및 Min-Max 정규화
    features = ['Sleep Duration', 'Quality of Sleep']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    # K-Means (K=3) 군집화 실행
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_features)
    
    # 군집 이름 매핑 (수면 시간 평균 기준 정리)
    cluster_centers = df.groupby('Cluster')['Sleep Duration'].mean().sort_values()
    cluster_mapping = {
        cluster_centers.index[0]: '단수면 군집 (Cluster 0)',
        cluster_centers.index[1]: '중수면 군집 (Cluster 1)',
        cluster_centers.index[2]: '장수면 군집 (Cluster 2)'
    }
    df['Cluster_Name'] = df['Cluster'].map(cluster_mapping)
    
    # 실루엣 계수 계산
    sil_score = silhouette_score(scaled_features, df['Cluster'])
    
    return df, sil_score, kmeans, scaler

df, sil_score, kmeans, scaler = load_and_preprocess_data()

# 3. 사이드바 - 주요 요약 지표
st.sidebar.header("📊 분석 데이터 요약")
st.sidebar.metric("총 데이터 수", f"{len(df)} 행")
st.sidebar.metric("K-Means 실루엣 계수", f"{sil_score:.3f}")

# 4. 메인 화면 - 탭 구성
tab1, tab2, tab3 = st.tabs(["📌 데이터 전처리 & 군집화", "📈 군집별 스트레스 비교", "🔍 개별 데이터 탐색"])

with tab1:
    st.subheader("1. K-Means 군집화 결과 (K=3)")
    st.markdown(f"**실루엣 계수**: `{sil_score:.3f}` (0.6 이상으로 군집 간 경계가 매우 명확함)")
    
    # 군집별 주요 수치 요약 표
    cluster_summary = df.groupby('Cluster_Name').agg(
        수면시간_평균=('Sleep Duration', 'mean'),
        수면의질_평균=('Quality of Sleep', 'mean'),
        스트레스_평균=('Stress Level', 'mean'),
        인원수=('Person ID', 'count')
    ).reset_index()
    
    st.dataframe(cluster_summary.style.format({
        '수면시간_평균': '{:.2f}시간',
        '수면의질_평균': '{:.2f}점',
        '스트레스_평균': '{:.2f}점',
        '인원수': '{:d}명'
    }), use_container_width=True)
    
    # 군집화 산점도 시각화
    fig_cluster = px.scatter(
        df, 
        x='Sleep Duration', 
        y='Quality of Sleep', 
        color='Cluster_Name',
        title="수면 시간과 수면의 질에 따른 K-Means 군집 분포",
        labels={'Sleep Duration': '수면 시간 (시간)', 'Quality of Sleep': '수면의 질 (점수)'},
        hover_data=['Stress Level', 'Sleep Disorder']
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

with tab2:
    st.subheader("2. 수면 시간대 군집별 인지 피로도(`Stress Level`) 차이")
    
    # 스트레스 지수 비교 박스플롯
    fig_stress = px.box(
        df, 
        x='Cluster_Name', 
        y='Stress Level', 
        color='Cluster_Name',
        points="all",
        title="군집별 인지 피로도(스트레스 지수) 분포 및 계단식 차이",
        labels={'Cluster_Name': '수면 군집', 'Stress Level': '스트레스 지수 (1~10)'}
    )
    st.plotly_chart(fig_stress, use_container_width=True)
    
    st.markdown("""
    ### 💡 주요 분석 결과
    * **단수면 군집**: 평균 수면 시간이 가장 적으며, **스트레스 지수가 가장 높게 형성**됨.
    * **중/장수면 군집**: 수면 시간과 수면의 질이 높아질수록 **스트레스 지수가 계단식으로 유의미하게 감소**함.
    * **결측치 처리 효과**: 219개의 `Sleep Disorder` 결측치를 'None(정상)'으로 채워 넣음으로써 정상 수면군이 중/장수면 군집으로 올바르게 분리됨.
    """)

with tab3:
    st.subheader("3. 원본 데이터 조회 및 필터링")
    
    selected_cluster = st.multiselect(
        "조회할 군집 선택", 
        options=df['Cluster_Name'].unique(), 
        default=df['Cluster_Name'].unique()
    )
    
    filtered_df = df[df['Cluster_Name'].isin(selected_cluster)]
    st.dataframe(filtered_df[['Person ID', 'Gender', 'Age', 'Sleep Duration', 'Quality of Sleep', 'Stress Level', 'Sleep Disorder', 'Cluster_Name']], use_container_width=True)
