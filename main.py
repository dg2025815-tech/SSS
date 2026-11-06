import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import plotly.express as px

# 페이지 설정 및 사이드바 구성
st.set_page_config(page_title="수면 패턴과 인지 피로도 분석", page_icon="💤", layout="wide")

# 사이드바 메뉴 구성
st.sidebar.title("메뉴 Navigation")
menu = st.sidebar.radio(
    "페이지 선택",
    ["main", "데이터 살펴보기", "전처리", "모델 후보", "평가지표", "모델 원리", "해석 한계", "구술 연습"]
)

# 데이터 로드 함수 (Kaggle 수면 데이터셋 처리)
@st.cache_data
def load_data():
    file_path = "human_cognitive_performance.csv"
    if not os.path.exists(file_path):
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                file_path = f
                break
    df = pd.read_csv(file_path)
    
    # 열 이름 표준화
    col_map = {}
    for col in df.columns:
        c_lower = col.lower().replace(' ', '_')
        if 'sleep' in c_lower and 'dur' in c_lower: col_map[col] = 'Sleep Duration'
        elif 'qual' in c_lower: col_map[col] = 'Quality of Sleep'
        elif 'stress' in c_lower: col_map[col] = 'Stress Level'
        elif 'disorder' in c_lower: col_map[col] = 'Sleep Disorder'
        elif 'person' in c_lower or 'id' in c_lower: col_map[col] = 'Person ID'
    df = df.rename(columns=col_map)
    return df

df_raw = load_data()

# -----------------------------------------------------------------------------
# 1. MAIN 페이지 (프로젝트 대시보드 메인)
# -----------------------------------------------------------------------------
if menu == "main":
    st.title("🧪 데이터 과학 실습실: 수면 분석 프로젝트")
    st.write("수면 패턴(수면 시간, 수면의 질)과 인지 피로도(`Stress Level`)의 관계를 K-Means 군집 분석을 통해 탐구하는 실습실입니다.")
    st.markdown("---")
    
    st.subheader("① 주제 선정·데이터 수집")
    st.info("수면 패턴에 따라 데이터 기반의 군집을 형성하고, 그룹 간 스트레스 지수의 차이가 나타나는지 분석합니다.")
    
    st.subheader("② 데이터 시각화·인사이트 & 전처리")
    st.info("결측치(`Sleep Disorder` 219개)를 'None(정상 수면)'으로 대체 보존하고 Min-Max Scale 정규화를 수행합니다.")
    
    st.subheader("③ 모델 후보·베이스라인")
    st.info("군집화 모델 후보(K-Means, DBSCAN, 계층적 군집화)를 비교하고 K-Means(K=3)를 최종 채택했습니다.")

# -----------------------------------------------------------------------------
# 2. 데이터 살펴보기 페이지
# -----------------------------------------------------------------------------
elif menu == "데이터 살펴보기":
    st.title("📊 데이터 살펴보기")
    st.write("데이터를 처음 열었을 때 크기, 빈 값, 자료형을 확인합니다.")
    
    st.markdown("### 🧑‍🎓 학생")
    st.info("스트림릿 앱으로 수면 건강 데이터를 처음 열어 보는 화면을 만들어 줘. 행 개수, 열 개수, 데이터 앞부분 10행을 보여줘.")
    
    st.markdown("### 🤖 AI")
    st.code("""
import pandas as pd
import streamlit as st

df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
st.write("행 개수:", df.shape[0], "/ 열 개수:", df.shape[1])
st.dataframe(df.head(10))
    """, language="python")
    
    st.markdown("#### ▶️ 이 코드를 실행한 결과")
    st.write(f"**행 개수**: `{df_raw.shape[0]}` / **열 개수**: `{df_raw.shape[1]}`")
    st.dataframe(df_raw.head(10), use_container_width=True)

# -----------------------------------------------------------------------------
# 3. 전처리 페이지
# -----------------------------------------------------------------------------
elif menu == "전처리":
    st.title("🛠️ 전처리")
    st.write("데이터에서 발견된 결측치와 변수 정규화 전처리를 수행합니다.")
    
    st.markdown("### 🧑‍🎓 학생")
    st.info("Sleep Disorder 열의 빈 값(NaN) 219개를 어떻게 처리하면 좋을지 알려줘.")
    
    st.markdown("### 🤖 AI")
    st.markdown("`Sleep Disorder`의 빈 값 219개는 삭제하지 않고 **'None(수면 장애 없음)'**으로 보존 처리하는 것이 바람직합니다.")
    st.code("""
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
st.write("전처리 후 데이터 개수:", len(df))
    """, language="python")
    
    st.markdown("#### ▶️ 이 코드를 실행한 결과")
    df_prep = df_raw.copy()
    if 'Sleep Disorder' in df_prep.columns:
        df_prep['Sleep Disorder'] = df_prep['Sleep Disorder'].fillna('None')
    st.write(f"전처리 전 행 수: `{len(df_raw)}` ➔ 전처리 후 행 수: `{len(df_prep)}` (데이터 보존 완료)")

# -----------------------------------------------------------------------------
# 4. 모델 후보 페이지
# -----------------------------------------------------------------------------
elif menu == "모델 후보":
    st.title("📌 모델 후보·베이스라인")
    st.write("수면 분석에 적합한 군집화 모델 후보 3가지를 비교합니다.")
    
    st.markdown("### 🧑‍🎓 학생")
    st.info("수면 데이터를 군집화하기에 적합한 대표 모델 3가지의 특징과 장단점을 표로 정리해 줘.")
    
    st.markdown("### 🤖 AI")
    model_table = pd.DataFrame({
        "모델": ["K-Means (선택)", "DBSCAN", "계층적 군집화"],
        "원리": ["중심점 기반 그룹화", "밀도 기반 그룹화", "나무 구조 순차 결합"],
        "장점": ["경계와 기준 구분이 명확함", "노이즈 검출에 용이함", "계층 시각화 가능"],
        "우리 데이터 결과": ["실루엣 계수 0.383 ~ 0.672", "대부분 거대 군집 1개로 뭉침", "군집 경계가 다소 모호함"]
    })
    st.table(model_table)

# -----------------------------------------------------------------------------
# 5. 평가지표 페이지
# -----------------------------------------------------------------------------
elif menu == "평가지표":
    st.title("📈 평가지표·결과")
    st.write("K-Means 군집 분석 결과(실루엣 계수 및 박스플롯 분포)를 해석합니다.")
    
    df_model = df_raw.copy()
    features = ['Sleep Duration', 'Quality of Sleep']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_model[features])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_model['Cluster'] = kmeans.fit_predict(scaled)
    sil_score = silhouette_score(scaled, df_model['Cluster'])
    
    centers = df_model.groupby('Cluster')['Sleep Duration'].mean().sort_values()
    mapping = {centers.index[0]: '단수면 군집', centers.index[1]: '중수면 군집', centers.index[2]: '장수면 군집'}
    df_model['Cluster_Name'] = df_model['Cluster'].map(mapping)
    
    st.markdown("### 🧑‍🎓 학생")
    st.info("K-Means 군집화 결과와 군집별 인지 피로도(스트레스 지수) 분포를 그래프로 그려 줘.")
    
    st.markdown("### 🤖 AI")
    st.write(f"**최종 실루엣 계수(Silhouette Score)**: `{sil_score:.3f}`")
    
    fig1 = px.scatter(df_model, x='Sleep Duration', y='Quality of Sleep', color='Cluster_Name', title="K-Means 산점도 분포")
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = px.box(df_model, x='Cluster_Name', y='Stress Level', color='Cluster_Name', title="군집별 인지 피로도(스트레스) 박스플롯")
    st.plotly_chart(fig_stress if 'fig_stress' in locals() else fig2, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. 모델 원리 페이지
# -----------------------------------------------------------------------------
elif menu == "모델 원리":
    st.title("🔍 모델 원리")
    st.write("K-Means 알고리즘이 군집 중심점(Centroid)을 업데이트하는 작동 원리를 설명합니다.")
    
    st.markdown("### 🧑‍🎓 학생")
    st.info("K-Means 알고리즘이 3개의 중심점을 찾아가는 수학적 원리를 간단히 설명해 줘.")
    
    st.markdown("### 🤖 AI")
    st.markdown("""
    1. **중심점 임의 설정**: $K=3$개의 중심점($C_1, C_2, C_3$)을 공간에 무작위 배치합니다.
    2. **유클리드 거리 계산**: 각 데이터 point와 중심점 간의 유클리드 거리를 계산하여 가장 가까운 중심점 군집에 할당합니다.
    3. **중심점 재계산**: 각 군집 소속 데이터의 평균 위치로 중심점 좌표를 이동합니다.
    4. **수렴**: 중심 위치가 더 이상 변하지 않을 때까지 위 과정을 반복합니다.
    """)

# -----------------------------------------------------------------------------
# 7. 해석 한계 페이지
# -----------------------------------------------------------------------------
elif menu == "해석 한계":
    st.title("⚠️ 해석·한계")
    st.write("본 수면 데이터 분석 결과가 지니는 한계점 및 주의사항을 명시합니다.")
    
    st.markdown("### 🧑‍🎓 학생")
    st.info("이 분석 결과를 개인의 건강 상태 진단에 바로 사용할 수 없는 한계점을 수치로 정리해 줘.")
    
    st.markdown("### 🤖 AI")
    st.markdown("""
    * **동일 군집 내 편차**: 단수면 군집 내에서도 스트레스 지수가 4점에서 9점까지 넓게 분포함.
    * **특정 연구 집단 편향**: 374명의 데이터로 특정 직종 비율이 높아 전체 대표성에 한계가 있음.
    * **오남용 주의**: 수면 시간 분류 결과만으로 타인의 정신 건강 및 피로도 수준을 단정해서는 안 됨.
    """)

# -----------------------------------------------------------------------------
# 8. 구술 연습 페이지
# -----------------------------------------------------------------------------
elif menu == "구술 연습":
    st.title("🗣️ 구술 연습")
    st.write("구술 평가 체크리스트 6항목 대본 연습 공간입니다.")
    
    st.markdown("1. **행·열 수**: 총 374행 13열 데이터셋")
    st.markdown("2. **결측치**: Sleep Disorder 열 219개 비어있음 ➔ 'None(정상)'으로 채워 보존")
    st.markdown("3. **이상치**: 수면 시간 4시간 미만 등 ➔ 삭제 없이 Min-Max Scale 정규화 적용")
