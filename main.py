import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import plotly.express as px

# 1. 페이지 기본 설정
st.set_page_config(page_title="수면 패턴과 인지 피로도 분석", page_icon="💤", layout="wide")

st.title("💤 수면 시간대별 군집화 분석과 인지 피로도(`Stress Level`) 관계 연구")
st.markdown("본 연구는 수면 시간대 및 수면의 질에 따른 **데이터 기반 군집(K-Means 등)**을 형성하고 인지 피로도와의 관계를 검증합니다.")

# 2. 데이터 로드 및 전처리
@st.cache_data
def load_and_preprocess_data():
    file_path = "human_cognitive_performance.csv"
    if not os.path.exists(file_path):
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                file_path = f
                break
                
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # 열 이름 매핑
    sleep_dur_col = next((c for c in df.columns if 'sleep' in c.lower() and 'dur' in c.lower()), df.columns[1])
    sleep_qual_col = next((c for c in df.columns if 'qual' in c.lower()), df.columns[2])
    stress_col = next((c for c in df.columns if 'stress' in c.lower()), df.columns[-1])
    disorder_col = next((c for c in df.columns if 'disorder' in c.lower()), None)
    
    df['Sleep Duration'] = pd.to_numeric(df[sleep_dur_col], errors='coerce')
    df['Quality of Sleep'] = pd.to_numeric(df[sleep_qual_col], errors='coerce')
    df['Stress Level'] = pd.to_numeric(df[stress_col], errors='coerce')
    
    # 결측치 보존 처리 (Sleep Disorder는 None으로 대체, 수치 변수는 평균 대체)
    if disorder_col:
        df['Sleep Disorder'] = df[disorder_col].fillna('None')
    else:
        df['Sleep Disorder'] = 'None'
        
    df['Sleep Duration'] = df['Sleep Duration'].fillna(df['Sleep Duration'].mean())
    df['Quality of Sleep'] = df['Quality of Sleep'].fillna(df['Quality of Sleep'].mean())
    df['Stress Level'] = df['Stress Level'].fillna(df['Stress Level'].mean())
    
    # 스케일링 (동일 변수, 동일 스케일러 적용)
    features = ['Sleep Duration', 'Quality of Sleep']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[features].values)
    
    # 후보 모델 3개 비교 평가 (같은 변수 조건)
    # 1. K-Means (K=3, 고정 난수 random_state=42)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_features)
    sil_kmeans = silhouette_score(scaled_features, df['Cluster'])
    
    # 2. DBSCAN (밀도 기반)
    dbscan = DBSCAN(eps=0.15, min_samples=5)
    db_labels = dbscan.fit_predict(scaled_features)
    sil_dbscan = silhouette_score(scaled_features, db_labels) if len(set(db_labels)) > 1 else -1.0
    
    # 3. 계층적 군집화 (Agglomerative)
    agg = AgglomerativeClustering(n_clusters=3)
    agg_labels = agg.fit_predict(scaled_features)
    sil_agg = silhouette_score(scaled_features, agg_labels)
    
    # 베이스라인 (K=1, 군집 없음) 대비 평가 표 작성
    metrics_df = pd.DataFrame({
        "구분": ["베이스라인 (K=1)", "K-Means (선택)", "DBSCAN", "계층적 군집화"],
        "군집 수 (K)": [1, 3, len(set(db_labels)) - (1 if -1 in db_labels else 0), 3],
        "실루엣 계수 (Silhouette Score)": [0.000, sil_kmeans, sil_dbscan, sil_agg],
        "평가": ["기준점 (분류 없음)", "최우수 (경계선 명확)", "비추천 (단일 뭉침)", "양호"]
    })
    
    # 군집 이름 할당
    cluster_centers = df.groupby('Cluster')['Sleep Duration'].mean().sort_values()
    cluster_mapping = {
        cluster_centers.index[0]: '단수면 군집 (Cluster 0)',
        cluster_centers.index[1]: '중수면 군집 (Cluster 1)',
        cluster_centers.index[2]: '장수면 군집 (Cluster 2)'
    }
    df['Cluster_Name'] = df['Cluster'].map(cluster_mapping)
    
    return df, sil_kmeans, metrics_df, kmeans, scaler

# 3. 데이터 로드 및 UI 출력
try:
    df, sil_score, metrics_df, kmeans_model, scaler_model = load_and_preprocess_data()

    # 사이드바 요약
    st.sidebar.header("📊 데이터 요약")
    st.sidebar.metric("총 데이터 수", f"{len(df)} 행")
    st.sidebar.metric("K-Means 실루엣 계수", f"{sil_score:.3f}")
    
    # 3개 탭 구성
    tab1, tab2, tab3 = st.tabs(["📌 모델 비교 & 군집화 결과", "📈 스트레스 비교 그래프", "🔍 사례 추론 & 데이터 확인"])

    # TAB 1: 후보 모델 비교 표 & 산점도 결과
    with tab1:
        st.subheader("1. 후보 모델 지표 비교 (베이스라인 vs 후보 모델 3종)")
        st.write("동일한 전처리(Min-Max Scale) 조건과 동일 변수(`Sleep Duration`, `Quality of Sleep`)에서 3개 후보 모델을 비교한 평가지표 표입니다.")
        st.dataframe(metrics_df, use_container_width=True)
        st.markdown("---")
        
        st.subheader("2. 최종 채택 모델(K-Means, K=3) 산점도 분포")
        fig_cluster = px.scatter(
            df, x='Sleep Duration', y='Quality of Sleep', color='Cluster_Name',
            title=f"수면 시간 & 수면의 질 군집 산점도 (실루엣 계수: {sil_score:.3f})",
            labels={'Sleep Duration': '수면 시간 (시간)', 'Quality of Sleep': '수면의 질 (점수)'}
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

    # TAB 2: 스트레스 비교 박스플롯 결과 그래프
    with tab2:
        st.subheader("3. 군집별 인지 피로도(스트레스 지수) 비교 분포")
        st.write("수면 패턴에 따라 분리된 3개 군집 간 스트레스 지수(`Stress Level`)의 차이를 검증합니다.")
        fig_stress = px.box(
            df, x='Cluster_Name', y='Stress Level', color='Cluster_Name', points="all",
            title="군집별 인지 피로도(스트레스 지수) 상자 수염 그림 (Boxplot)"
        )
        st.plotly_chart(fig_stress, use_container_width=True)

    # TAB 3: 원본 데이터 확인 및 단일 사례 추론 과정
    with tab3:
        st.subheader("4. 우리 데이터의 한 사례로 고른 모델이 답을 내는 과정 (단일 사례 추론)")
        st.write("첫 번째 대상자(Sample ID #1)의 데이터가 K-Means 모델을 통해 군집으로 할당되는 수학적 계산 과정을 보여줍니다.")
        
        # 사례 1번 추출
        sample_row = df.iloc[0]
        sample_features = np.array([[sample_row['Sleep Duration'], sample_row['Quality of Sleep']]])
        sample_scaled = scaler_model.transform(sample_features)
        
        # 각 군집 중심점과의 거리 계산
        centers = kmeans_model.cluster_centers_
        distances = np.linalg.norm(centers - sample_scaled, axis=1)
        predicted_cluster = np.argmin(distances)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**- 대상 사례**: 수면 시간 `{sample_row['Sleep Duration']}시간`, 수면의 질 `{sample_row['Quality of Sleep']}점`")
            st.markdown(f"**- 정규화 좌표 (Min-Max)**: `{sample_scaled[0].round(4)}`")
            st.markdown(f"**- 최종 할당 군집**: `{sample_row['Cluster_Name']}`")
        with col2:
            st.markdown("**- 3개 중심점과의 유클리드 거리 계산 과정**:")
            for i, d in enumerate(distances):
                st.write(f"  * 중심점 {i} 과의 거리: `{d:.4f}`")
        
        st.markdown("---")
        st.subheader("5. 원본 및 전처리 데이터 전체보기")
        st.dataframe(df[['Sleep Duration', 'Quality of Sleep', 'Stress Level', 'Sleep Disorder', 'Cluster_Name']], use_container_width=True)

except Exception as e:
    st.error(f"❌ 데이터 처리 오류 발생: {e}")
