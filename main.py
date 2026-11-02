import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import plotly.express as px

st.set_page_config(page_title="수면 패턴과 인지 피로도 분석", page_icon="💤", layout="wide")

st.title("💤 수면 시간대별 군집화 분석과 인지 피로도(`Stress Level`) 관계 연구")
st.markdown("본 연구는 수면 시간대 및 수면의 질에 따른 **데이터 기반 군집(K-Means)**을 형성하고 인지 피로도과의 관계를 검증합니다.")

@st.cache_data
def load_and_preprocess_data():
    file_path = "human_cognitive_performance.csv"
    if not os.path.exists(file_path):
        # 파일이 없을 경우 예비 탐색
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                file_path = f
                break
                
    df = pd.read_csv(file_path)
    
    # 컬럼명 대소문자 및 띄어쓰기 유연 처리
    cols = {col.lower().replace(' ', '_'): col for col in df.columns}
    
    # 필요한 핵심 변수 탐색 및 매핑
    sleep_dur_col = next((df.columns[i] for i, c in enumerate(df.columns) if 'sleep' in c.lower() and 'dur' in c.lower()), None)
    sleep_qual_col = next((df.columns[i] for i, c in enumerate(df.columns) if 'qual' in c.lower()), None)
    stress_col = next((df.columns[i] for i, c in enumerate(df.columns) if 'stress' in c.lower() or 'score' in c.lower() or 'performance' in c.lower()), None)
    
    # 탐색 실패 시 기본 위치 기반 지정
    if not sleep_dur_col: sleep_dur_col = df.columns[3] if len(df.columns) > 3 else df.columns[0]
    if not sleep_qual_col: sleep_qual_col = df.columns[4] if len(df.columns) > 4 else df.columns[1]
    if not stress_col: stress_col = df.columns[5] if len(df.columns) > 5 else df.columns[-1]

    df['Sleep Duration'] = pd.to_numeric(df[sleep_dur_col], errors='coerce')
    df['Quality of Sleep'] = pd.to_numeric(df[sleep_qual_col], errors='coerce')
    df['Stress Level'] = pd.to_numeric(df[stress_col], errors='coerce')

    # 결측치 평균 대체
    df['Sleep Duration'] = df['Sleep Duration'].fillna(df['Sleep Duration'].mean())
    df['Quality of Sleep'] = df['Quality of Sleep'].fillna(df['Quality of Sleep'].mean())
    df['Stress Level'] = df['Stress Level'].fillna(df['Stress Level'].mean())
    
    # K-Means 군집화
    features = ['Sleep Duration', 'Quality of Sleep']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_features)
    
    cluster_centers = df.groupby('Cluster')['Sleep Duration'].mean().sort_values()
    cluster_mapping = {
        cluster_centers.index[0]: '단수면 군집 (Cluster 0)',
        cluster_centers.index[1]: '중수면 군집 (Cluster 1)',
        cluster_centers.index[2]: '장수면 군집 (Cluster 2)'
    }
    df['Cluster_Name'] = df['Cluster'].map(cluster_mapping)
    sil_score = silhouette_score(scaled_features, df['Cluster'])
    
    return df, sil_score

try:
    df, sil_score = load_and_preprocess_data()

    st.sidebar.header("📊 데이터 요약")
    st.sidebar.metric("총 데이터 수", f"{len(df)} 행")
    st.sidebar.metric("K-Means 실루엣 계수", f"{sil_score:.3f}")

    tab1, tab2, tab3 = st.tabs(["📌 군집화 결과", "📈 스트레스 비교", "🔍 데이터 확인"])

    with tab1:
        st.subheader("1. K-Means 군집화 분포 (K=3)")
        st.markdown(f"**실루엣 계수**: `{sil_score:.3f}`")
        fig_cluster = px.scatter(df, x='Sleep Duration', y='Quality of Sleep', color='Cluster_Name', title="수면 시간 & 수면의 질 군집")
        st.plotly_chart(fig_cluster, use_container_width=True)

    with tab2:
        st.subheader("2. 군집별 스트레스 지수 분포")
        fig_stress = px.box(df, x='Cluster_Name', y='Stress Level', color='Cluster_Name', title="군집별 인지 피로도(스트레스)")
        st.plotly_chart(fig_stress, use_container_width=True)

    with tab3:
        st.subheader("3. 원본 데이터")
        st.dataframe(df, use_container_width=True)

except Exception as e:
    st.error(f"❌ 데이터 읽기 오류 발생: {e}")
