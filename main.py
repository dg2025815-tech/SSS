import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import plotly.express as px

# 1. 페이지 설정
st.set_page_config(
    page_title="수면 패턴과 인지 피로도 분석",
    page_icon="💤",
    layout="wide"
)

st.title("💤 수면 시간대별 군집화 분석과 인지 피로도(`Stress Level`) 관계 연구")
st.markdown("""
본 연구는 수면 시간대 및 수면의 질에 따른 **데이터 기반 군집(K-Means)**을 형성하고, 
군집 간 인지 피로도(`Stress Level`)의 차이를 검증합니다.
""")

# 2. 데이터 로드 (리포지토리에 있는 모든 CSV 파일 탐색)
@st.cache_data
def load_and_preprocess_data():
    possible_paths = [
        "human_cognitive_performance.csv",
        "Sleep_health_and_lifestyle_dataset.csv",
        "data/human_cognitive_performance.csv",
        "data/Sleep_health_and_lifestyle_dataset.csv"
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
            
    if file_path is None:
        st.error("❌ 데이터 파일을 찾을 수 없습니다. CSV 파일이 리포지토리에 정상적으로 올랐는지 확인해주세요.")
        st.stop()
        
    df = pd.read_csv(file_path)
    
    # 변수명이 다른 경우 자동 매핑
    column_mapping = {
        'sleep_duration': 'Sleep Duration',
        'quality_of_sleep': 'Quality of Sleep',
        'stress_level': 'Stress Level',
        'sleep_disorder': 'Sleep Disorder'
    }
    df = df.rename(columns=column_mapping)
    
    # 결측치 처리 (Sleep Disorder)
    if 'Sleep Disorder' in df.columns:
        df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    else:
        df['Sleep Disorder'] = 'None'
        
    # K-Means 군집화 (Sleep Duration, Quality of Sleep 기준)
    features = ['Sleep Duration', 'Quality of Sleep']
    
    # 열 존재 여부 체크
    for col in features:
        if col not in df.columns:
            st.error(f"❌ 데이터셋에 '{col}' 열이 존재하지 않습니다.")
            st.stop()
            
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_features)
    
    # 군집 이름 정리
    cluster_centers = df.groupby('Cluster')['Sleep Duration'].mean().sort_values()
    cluster_mapping = {
        cluster_centers.index[0]: '단수면 군집 (Cluster 0)',
        cluster_centers.index[1]: '중수면 군집 (Cluster 1)',
        cluster_centers.index[2]: '장수면 군집 (Cluster 2)'
    }
    df['Cluster_Name'] = df['Cluster'].map(cluster_mapping)
    
    sil_score = silhouette_score(scaled_features, df['Cluster'])
    
    return df, sil_score

df, sil_score = load_and_preprocess_data()

# 3. 사이드바 지표
st.sidebar.header("📊 데이터 요약")
st.sidebar.metric("총 데이터 수", f"{len(df)} 행")
st.sidebar.metric("K-Means 실루엣 계수", f"{sil_score:.3f}")

# 4. 탭 구성
tab1, tab2, tab3 = st.tabs(["📌 데이터 전처리 & 군집화", "📈 군집별 스트레스 비교", "🔍 개별 데이터 탐색"])

with tab1:
    st.subheader("1. K-Means 군집화 결과 (K=3)")
    st.markdown(f"**실루엣 계수**: `{sil_score:.3f}`")
    
    cluster_summary = df.groupby('Cluster_Name').agg(
        수면시간_평균=('Sleep Duration', 'mean'),
        수면의질_평균=('Quality of Sleep', 'mean'),
        스트레스_평균=('Stress Level', 'mean'),
        인원수=('Cluster', 'count')
    ).reset_index()
    
    st.dataframe(cluster_summary, use_container_width=True)
    
    fig_cluster = px.scatter(
        df, 
        x='Sleep Duration', 
        y='Quality of Sleep', 
        color='Cluster_Name',
        title="수면 시간과 수면의 질에 따른 K-Means 군집 분포"
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

with tab2:
    st.subheader("2. 수면 시간대 군집별 인지 피로도(`Stress Level`) 차이")
    fig_stress = px.box(
        df, 
        x='Cluster_Name', 
        y='Stress Level', 
        color='Cluster_Name',
        title="군집별 인지 피로도(스트레스 지수) 분포"
    )
    st.plotly_chart(fig_stress, use_container_width=True)

with tab3:
    st.subheader("3. 원본 데이터 탐색")
    st.dataframe(df, use_container_width=True)
