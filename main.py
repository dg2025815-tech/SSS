import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import plotly.express as px

# 1. 페이지 기본 설정
st.set_page_config(page_title="수면 패턴과 인지 피로도 심층 분석", page_icon="💤", layout="wide")

st.title("💤 수면 시간대별 군집화 분석과 인지 피로도(`Stress Level`) 관계 연구")
st.markdown("본 연구는 수면 시간대 및 수면의 질에 따른 **데이터 기반 군집(K-Means)**을 형성하고, 인지 피로도 및 상관관계를 종합 검증합니다.")

# 2. 데이터 로드 및 정밀 전처리
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
    
    # 열 이름 유연 탐색
    sleep_dur_col = next((c for c in df.columns if 'sleep' in str(c).lower() and ('dur' in str(c).lower() or 'hour' in str(c).lower())), df.columns[1])
    sleep_qual_col = next((c for c in df.columns if 'qual' in str(c).lower()), df.columns[2])
    stress_col = next((c for c in df.columns if 'stress' in str(c).lower()), df.columns[-1])
    disorder_col = next((c for c in df.columns if 'disorder' in str(c).lower()), None)
    occupation_col = next((c for c in df.columns if 'occupation' in str(c).lower() or 'job' in str(c).lower()), None)

    new_df = pd.DataFrame()
    new_df['Sleep Duration'] = pd.to_numeric(df[sleep_dur_col], errors='coerce').fillna(df[sleep_dur_col].median()).fillna(7.0)
    new_df['Quality of Sleep'] = pd.to_numeric(df[sleep_qual_col], errors='coerce').fillna(df[sleep_qual_col].median()).fillna(6.0)
    new_df['Stress Level'] = pd.to_numeric(df[stress_col], errors='coerce').fillna(df[stress_col].median()).fillna(5.0)
    
    new_df['Sleep Disorder'] = df[disorder_col].fillna('None').astype(str) if disorder_col else 'None'
    new_df['Occupation'] = df[occupation_col].fillna('General').astype(str) if occupation_col else 'General'
    new_df['Person ID'] = range(1, len(new_df) + 1)
    
    # Min-Max Scaling & K-Means
    X = new_df[['Sleep Duration', 'Quality of Sleep']].values
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    new_df['Cluster'] = kmeans.fit_predict(scaled_features)
    sil_score = silhouette_score(scaled_features, new_df['Cluster'])
    
    # 군집 이름 매핑
    cluster_centers = new_df.groupby('Cluster')['Sleep Duration'].mean().sort_values()
    cluster_mapping = {
        cluster_centers.index[0]: '단수면 군집 (Cluster 0)',
        cluster_centers.index[1]: '중수면 군집 (Cluster 1)',
        cluster_centers.index[2]: '장수면 군집 (Cluster 2)'
    }
    new_df['Cluster_Name'] = new_df['Cluster'].map(cluster_mapping)
    
    return new_df, sil_score, kmeans, scaler

# 3. 메인 앱 실행 및 탭 구성
try:
    df, sil_score, kmeans_model, scaler_model = load_and_preprocess_data()

    # 사이드바 요약 지표
    st.sidebar.header("📊 데이터 요약")
    st.sidebar.metric("총 데이터 수", f"{len(df)} 행")
    st.sidebar.metric("K-Means 실루엣 계수", f"{sil_score:.3f}")
    st.sidebar.metric("평균 수면 시간", f"{df['Sleep Duration'].mean():.2f} 시간")
    st.sidebar.metric("평균 스트레스 지수", f"{df['Stress Level'].mean():.2f} 점")

    # 4개 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 군집 분석 결과", 
        "📊 변수 간 상관관계 & 종합 분석", 
        "📈 스트레스 & 집단별 비교", 
        "🔍 원본 데이터 및 추론"
    ])

    # TAB 1: K-Means 군집 분석
    with tab1:
        st.subheader("1. K-Means 군집화 분포 (K=3)")
        st.markdown(f"수면 시간과 수면의 질을 Min-Max Scaler로 정규화한 뒤 형성한 3개 군집 분포입니다. **(실루엣 계수: `{sil_score:.3f}`)**")
        
        fig_cluster = px.scatter(
            df, x='Sleep Duration', y='Quality of Sleep', color='Cluster_Name',
            hover_data=['Stress Level', 'Sleep Disorder'],
            title="수면 시간 & 수면의 질 기반 K-Means 군집 분포",
            labels={'Sleep Duration': '수면 시간 (시간)', 'Quality of Sleep': '수면의 질 (점수)'}
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        st.markdown("---")
        st.subheader("2. 군집별 주요 수치 요약")
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

    # TAB 2: 데이터 심층 상관분석 그래프
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("3. 변수 간 상관관계 히트맵 (Correlation)")
            corr_df = df[['Sleep Duration', 'Quality of Sleep', 'Stress Level']].corr().round(3)
            fig_corr = px.imshow(
                corr_df, text_auto=True, color_continuous_scale='RdBu_r',
                title="수치 변수 간 상관계수 히트맵",
                labels=dict(color="상관계수")
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("💡 **해석**: 수면 시간/수면의 질과 스트레스 지수 간에 강한 **음의 상관관계**가 나타납니다.")

        with col2:
            st.subheader("4. 수면 장애(Sleep Disorder) 비중 분포")
            disorder_counts = df['Sleep Disorder'].value_counts().reset_index()
            fig_pie = px.pie(
                disorder_counts, names='Sleep Disorder', values='count', hole=0.4,
                title="수면 장애 범주별 인원 비율",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.caption("💡 **해석**: 결측치를 대체한 'None(정상 수면)' 비중이 높으며, 불면증(Insomnia) 및 수면무호흡증(Sleep Apnea)이 뒤를 이읍니다.")

    # TAB 3: 스트레스 및 직업군 비교 그래프
    with tab3:
        st.subheader("5. 군집별 인지 피로도(스트레스 지수) 비교 분포")
        fig_stress = px.box(
            df, x='Cluster_Name', y='Stress Level', color='Cluster_Name', points="all",
            title="군집별 인지 피로도 분포 상자 수염 그림 (Boxplot)",
            labels={'Stress Level': '스트레스 지수 (1~10)', 'Cluster_Name': '군집명'}
        )
        st.plotly_chart(fig_stress, use_container_width=True)

        if df['Occupation'].nunique() > 1:
            st.markdown("---")
            st.subheader("6. 직업군별 평균 수면 시간 및 스트레스 수준")
            occ_summary = df.groupby('Occupation')[['Sleep Duration', 'Stress Level']].mean().reset_index()
            fig_occ = px.bar(
                occ_summary, x='Occupation', y=['Sleep Duration', 'Stress Level'], barmode='group',
                title="직업군별 수면 시간 vs 스트레스 지수 비교",
                labels={'value': '수치', 'variable': '지표'}
            )
            st.plotly_chart(fig_occ, use_container_width=True)

    # TAB 4: 데이터 원본 및 계산 과정
    with tab4:
        st.subheader("7. 단일 사례 K-Means 할당 추론 과정")
        sample_row = df.iloc[0]
        sample_features = np.array([[sample_row['Sleep Duration'], sample_row['Quality of Sleep']]])
        sample_scaled = scaler_model.transform(sample_features)
        
        centers = kmeans_model.cluster_centers_
        distances = np.linalg.norm(centers - sample_scaled, axis=1)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**- 대상 사례**: 수면 시간 `{sample_row['Sleep Duration']}시간`, 수면의 질 `{sample_row['Quality of Sleep']}점`")
            st.markdown(f"**- 정규화 좌표 (Min-Max)**: `{sample_scaled[0].round(4)}`")
            st.markdown(f"**- 최종 군집**: `{sample_row['Cluster_Name']}`")
        with c2:
            st.markdown("**- 3개 중심점과의 유클리드 거리**:")
            for i, d in enumerate(distances):
                st.write(f"  * 중심점 {i} 과의 거리: `{d:.4f}`")

        st.markdown("---")
        st.subheader("8. 원본 데이터 세부 확인")
        st.dataframe(df, use_container_width=True)

except Exception as e:
    st.error(f"❌ 데이터 분석 중 오류가 발생했습니다: {e}")
