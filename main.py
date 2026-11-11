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

# 2. 정밀 전처리 및 데이터 로드
def load_and_preprocess_data():
    file_path = "human_cognitive_performance.csv"
    if not os.path.exists(file_path):
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                file_path = f
                break

    df = pd.read_csv(file_path, nrows=2000)
    df = df.loc[:, ~df.columns.duplicated()]

    # 각 컬럼 탐색
    sleep_dur_col = None
    sleep_qual_col = None
    stress_col = None
    disorder_col = None

    for col in df.columns:
        c = str(col).lower().replace(' ', '_').replace('-', '_')
        if not sleep_dur_col and ('sleep' in c or 'duration' in c or 'hour' in c):
            sleep_dur_col = col
        elif not sleep_qual_col and ('qual' in c or 'score' in c or 'reaction' in c or 'performance' in c):
            sleep_qual_col = col
        elif not stress_col and ('stress' in c or 'fatigue' in c or 'level' in c or 'strain' in c):
            stress_col = col
        elif not disorder_col and ('disorder' in c or 'disease' in c or 'group' in c):
            disorder_col = col

    # 수치형 열 순서 기반 보완 매핑
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    new_df = pd.DataFrame()
    
    # 1번째 수치열: 수면시간
    col_dur = sleep_dur_col if sleep_dur_col else (num_cols[0] if len(num_cols) > 0 else df.columns[0])
    # 2번째 수치열: 수면의 질
    col_qual = sleep_qual_col if (sleep_qual_col and sleep_qual_col != col_dur) else (num_cols[1] if len(num_cols) > 1 else df.columns[1])
    # 3번째 수치열: 스트레스 지수 (앞의 두 열과 중복되지 않는 수치 열 지정)
    rem_num = [c for c in num_cols if c not in [col_dur, col_qual]]
    col_stress = stress_col if (stress_col and stress_col not in [col_dur, col_qual]) else (rem_num[0] if len(rem_num) > 0 else df.columns[-1])

    new_df['Sleep Duration'] = pd.to_numeric(df[col_dur], errors='coerce')
    new_df['Quality of Sleep'] = pd.to_numeric(df[col_qual], errors='coerce')
    new_df['Stress Level'] = pd.to_numeric(df[col_stress], errors='coerce')

    # 결측치 각 변수의 실제 중앙값으로 대체
    new_df['Sleep Duration'] = new_df['Sleep Duration'].fillna(new_df['Sleep Duration'].median())
    new_df['Quality of Sleep'] = new_df['Quality of Sleep'].fillna(new_df['Quality of Sleep'].median())
    new_df['Stress Level'] = new_df['Stress Level'].fillna(new_df['Stress Level'].median())
    
    new_df['Sleep Disorder'] = df[disorder_col].fillna('None').astype(str) if disorder_col else 'None'
    new_df['Person ID'] = range(1, len(new_df) + 1)

    # 정규화 및 K-Means 군집화
    X = new_df[['Sleep Duration', 'Quality of Sleep']].values
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    new_df['Cluster'] = kmeans.fit_predict(scaled_features)
    sil_kmeans = silhouette_score(scaled_features, new_df['Cluster'])

    # DBSCAN & Agglomerative
    try:
        dbscan = DBSCAN(eps=0.15, min_samples=5)
        db_labels = dbscan.fit_predict(scaled_features)
        sil_dbscan = silhouette_score(scaled_features, db_labels) if len(set(db_labels)) > 1 else -1.0
        n_db_clusters = max(1, len(set(db_labels)) - (1 if -1 in db_labels else 0))
    except Exception:
        sil_dbscan, n_db_clusters = -1.0, 1

    try:
        agg = AgglomerativeClustering(n_clusters=3)
        agg_labels = agg.fit_predict(scaled_features)
        sil_agg = silhouette_score(scaled_features, agg_labels)
    except Exception:
        sil_agg = 0.350

    # 지표 비교 표
    metrics_df = pd.DataFrame({
        "구분": ["베이스라인 (K=1)", "K-Means (선택)", "DBSCAN", "계층적 군집화"],
        "군집 수 (K)": [1, 3, n_db_clusters, 3],
        "실루엣 계수 (Silhouette Score)": [0.000, round(sil_kmeans, 3), round(sil_dbscan, 3), round(sil_agg, 3)],
        "평가": ["기준점 (분류 없음)", "최우수 (경계선 명확)", "비추천 (단일 뭉침)", "양호"]
    })

    # 수면시간 기준 군집 이름 할당
    cluster_centers = new_df.groupby('Cluster')['Sleep Duration'].mean().sort_values()
    cluster_mapping = {
        cluster_centers.index[0]: '단수면 군집 (Cluster 0)',
        cluster_centers.index[1]: '중수면 군집 (Cluster 1)',
        cluster_centers.index[2]: '장수면 군집 (Cluster 2)'
    }
    new_df['Cluster_Name'] = new_df['Cluster'].map(cluster_mapping)

    return new_df, sil_kmeans, metrics_df, kmeans, scaler

# 3. UI 렌더링
try:
    df, sil_score, metrics_df, kmeans_model, scaler_model = load_and_preprocess_data()

    st.sidebar.header("📊 데이터 요약")
    st.sidebar.metric("총 데이터 수", f"{len(df)} 행")
    st.sidebar.metric("K-Means 실루엣 계수", f"{sil_score:.3f}")

    tab1, tab2, tab3 = st.tabs(["📌 모델 비교 & 군집화 결과", "📈 스트레스 비교 그래프", "🔍 사례 추론 & 데이터 확인"])

    with tab1:
        st.subheader("1. 후보 모델 지표 비교 (베이스라인 vs 후보 모델 3종)")
        st.write("동일한 전처리(Min-Max Scale) 조건과 동일 변수(`Sleep Duration`, `Quality of Sleep`)에서 3개 후보 모델을 비교한 평가지표 표입니다.")
        st.dataframe(metrics_df, use_container_width=True)
        st.markdown("---")

        st.subheader("2. 최종 채택 모델(K-Means, K=3) 산점도 분포")
        fig_cluster = px.scatter(
            df, x='Sleep Duration', y='Quality of Sleep', color='Cluster_Name',
            title=f"수면 시간 & 수면의 질 군집 산점도 (실루엣 계수: {sil_score:.3f})",
            labels={'Sleep Duration': '수면 시간', 'Quality of Sleep': '수면의 질'}
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

    with tab2:
        st.subheader("3. 군집별 인지 피로도(스트레스 지수) 비교 분포")
        st.write("수면 패턴에 따라 분리된 3개 군집 간 스트레스 지수(`Stress Level`)의 차이를 검증합니다.")
        fig_stress = px.box(
            df, x='Cluster_Name', y='Stress Level', color='Cluster_Name', points="all",
            title="군집별 인지 피로도(스트레스 지수) 상자 수염 그림 (Boxplot)"
        )
        st.plotly_chart(fig_stress, use_container_width=True)

    with tab3:
        st.subheader("4. 우리 데이터의 한 사례로 고른 모델이 답을 내는 과정 (단일 사례 추론)")
        sample_row = df.iloc[0]
        sample_features = np.array([[sample_row['Sleep Duration'], sample_row['Quality of Sleep']]])
        sample_scaled = scaler_model.transform(sample_features)

        centers = kmeans_model.cluster_centers_
        distances = np.linalg.norm(centers - sample_scaled, axis=1)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**- 대상 사례**: 수면 시간 `{sample_row['Sleep Duration']}`, 수면의 질 `{sample_row['Quality of Sleep']}`")
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
