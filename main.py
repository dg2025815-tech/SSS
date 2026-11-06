import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import plotly.express as px

# 1. 페이지 설정 및 기본 레이아웃
st.set_page_config(page_title="수면 패턴과 인지 피로도 분석", page_icon="💤", layout="wide")

# 2. 사이드바 - 페이지 Navigation
st.sidebar.title("메뉴 Navigation")
menu = st.sidebar.radio(
    "페이지 선택",
    ["main", "데이터 살펴보기", "전처리", "모델 후보", "평가지표", "모델 원리", "해석 한계", "구술 연습"]
)

# 3. 사이드바 - 진행(Progress) 및 토글 컨트롤 (선생님 예시 동일 구현)
st.sidebar.markdown("---")
st.sidebar.subheader("진행")
show_all = st.sidebar.toggle("대화를 한 번에 모두 보기", value=False)

# Session state 초기화 함수
if "step" not in st.session_state:
    st.session_state.step = 1

def reset_step():
    st.session_state.step = 1

def next_step():
    st.session_state.step += 1

# 페이지가 바뀔 때 대화 단계 초기화
if "current_page" not in st.session_state or st.session_state.current_page != menu:
    st.session_state.current_page = menu
    st.session_state.step = 1

# 데이터 로드 함수
@st.cache_data
def load_data():
    file_path = "human_cognitive_performance.csv"
    if not os.path.exists(file_path):
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                file_path = f
                break
    df = pd.read_csv(file_path)
    
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

# Helper 함수: 대화 단계별 출력 제어
def render_step_content(total_steps, dialogue_blocks):
    current_s = total_steps if show_all else st.session_state.step
    
    # 사이드바 진행 상태 표시
    st.sidebar.caption(f"{min(current_s, total_steps)} / {total_steps} 단계")
    st.sidebar.progress(min(current_s / total_steps, 1.0))
    if st.sidebar.button("처음부터 다시", on_click=reset_step, use_container_width=True):
        st.rerun()

    # 대화 단계 출력
    for idx, block in enumerate(dialogue_blocks):
        step_num = idx + 1
        if current_s >= step_num:
            block()
            
            # 마지막 단계가 아니고 "한 번에 보기"가 꺼져 있을 때 파란색 버튼 표시
            if not show_all and current_s == step_num and step_num < total_steps:
                button_text = "다음: AI가 답하기 ▶" if step_num % 2 == 1 else "다음: 학생이 묻기 ▶"
                st.write("")
                st.button(button_text, on_click=next_step, type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# 1. MAIN 페이지
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
# 2. 데이터 살펴보기 페이지 (대화 4단계)
# -----------------------------------------------------------------------------
elif menu == "데이터 살펴보기":
    st.title("📊 데이터 살펴보기")
    st.write("이 페이지에서는 데이터를 처음 열어 크기, 빈 값, 분포, 자료형을 차례로 살펴보겠습니다.")
    
    def step1():
        st.markdown("### 🧑‍🎓 학생")
        st.info("스트림릿 앱으로 수면 건강 데이터를 처음 열어 보는 화면을 만들어 줘. 행 개수와 열 개수, 앞의 10행, 열마다 자료형을 보여 줘.")
        
    def step2():
        st.markdown("### 🤖 AI")
        st.write("데이터를 읽어 크기, 앞부분 10행, 열마다 자료형을 보여 주는 코드입니다.")
        st.code("""
import pandas as pd
import streamlit as st

df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
st.write("행 개수:", df.shape[0], "/ 열 개수:", df.shape[1])
st.dataframe(df.head(10))
st.write(df.dtypes.astype(str).rename("자료형"))
        """, language="python")
        st.markdown("#### ▶️ 이 코드를 실행한 결과")
        st.write(f"행 개수: `{df_raw.shape[0]}` / 열 개수: `{df_raw.shape[1]}`")
        st.dataframe(df_raw.head(10), use_container_width=True)

    def step3():
        st.markdown("### 🧑‍🎓 학생·직접 확인하기")
        st.info("주요 수치형 변수들(수면 시간, 수면의 질, 스트레스 지수)의 요약 통계량과 빈 값(결측치) 개수를 보여 줘.")

    def step4():
        st.markdown("### 🤖 AI")
        st.write("주요 변수의 통계량과 결측치 현황을 출력하는 코드입니다.")
        st.code("""
st.write("요약 통계량:")
st.dataframe(df.describe())
st.write("결측치 개수:")
st.write(df.isnull().sum())
        """, language="python")
        st.markdown("#### ▶️ 이 코드를 실행한 결과")
        st.dataframe(df_raw.describe(), use_container_width=True)
        st.write("결측치 현황:", df_raw.isnull().sum().to_dict())

    render_step_content(4, [step1, step2, step3, step4])

# -----------------------------------------------------------------------------
# 3. 전처리 페이지 (대화 4단계)
# -----------------------------------------------------------------------------
elif menu == "전처리":
    st.title("🛠️ 전처리")
    st.write("이 페이지에서는 살펴보기에서 찾은 문제를 하나씩 처리합니다.")
    
    def step1():
        st.markdown("### 🧑‍🎓 학생")
        st.info("Sleep Disorder 열의 빈 값 219개를 어떻게 처리하면 좋을지 알려 줘. 코드도 같이 줘.")

    def step2():
        st.markdown("### 🤖 AI")
        st.write("Sleep Disorder의 빈 값은 단순 누락이 아니라 '수면 장애 없음(정상)'을 의미하므로 삭제하지 않고 'None'으로 보존합니다.")
        st.code("""
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
st.write("지우기 전:", len(df_raw), "행 ➔ 지운 뒤:", len(df), "행")
        """, language="python")
        df_prep = df_raw.copy()
        if 'Sleep Disorder' in df_prep.columns:
            df_prep['Sleep Disorder'] = df_prep['Sleep Disorder'].fillna('None')
        st.markdown("#### ▶️ 이 코드를 실행한 결과")
        st.write(f"지우기 전: `{len(df_raw)}` 행 ➔ 지운 뒤: `{len(df_prep)}` 행 (데이터 삭제 없이 374행 유지)")

    def step3():
        st.markdown("### 🧑‍🎓 학생·직접 확인하기")
        st.info("K-Means 군집화를 위해 수면 시간과 수면의 질 변수를 정규화(Min-Max Scaling)하는 코드도 보여 줘.")

    def step4():
        st.markdown("### 🤖 AI")
        st.write("수치 범위를 0~1 사이로 맞추는 Min-Max Scale 적용 코드입니다.")
        st.code("""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Sleep Duration', 'Quality of Sleep']])
        """, language="python")

    render_step_content(4, [step1, step2, step3, step4])

# -----------------------------------------------------------------------------
# 4. 모델 후보 페이지 (대화 2단계)
# -----------------------------------------------------------------------------
elif menu == "모델 후보":
    st.title("📌 모델 후보·베이스라인")
    st.write("이 페이지에서는 군집화 알고리즘 후보 3개를 추천받고 베이스라인을 비교합니다.")
    
    def step1():
        st.markdown("### 🧑‍🎓 학생")
        st.info("수면 데이터를 군집화하기에 적합한 모델 2~3개를 추천해 줘. 어떤 원리로 군집화하는지, 장점과 단점, 우리 데이터 결과를 표로 정리해 줘.")

    def step2():
        st.markdown("### 🤖 AI")
        st.write("군집 분석에 많이 쓰는 모델 세 가지(K-Means, DBSCAN, 계층적 군집화)를 추천합니다.")
        model_table = pd.DataFrame({
            "모델": ["K-Means (선택)", "DBSCAN", "계층적 군집화"],
            "원리": ["중심점 기반 그룹화", "밀도 기반 그룹화", "나무 구조 순차 결합"],
            "장점": ["경계선 구분이 명확하고 해석하기 쉽움", "노이즈 및 이상치 구분에 용이함", "계층 시각화 가능"],
            "우리 데이터 결과": ["실루엣 계수 0.383 ~ 0.672 (최우수)", "대부분 거대 군집 1개로 뭉침", "군집 경계가 다소 모호함"]
        })
        st.table(model_table)

    render_step_content(2, [step1, step2])

# -----------------------------------------------------------------------------
# 5. 평가지표 페이지 (대화 2단계)
# -----------------------------------------------------------------------------
elif menu == "평가지표":
    st.title("📈 평가지표·결과")
    st.write("이 페이지에서는 K-Means 군집 분석 결과와 실루엣 계수, 스트레스 지수 분포를 확인합니다.")
    
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
    
    def step1():
        st.markdown("### 🧑‍🎓 학생")
        st.info("최종 K-Means 군집화 결과 산점도와 군집별 스트레스 지수 박스플롯 그림으로 보여 줘.")

    def step2():
        st.markdown("### 🤖 AI")
        st.write(f"K-Means (K=3) 군집화 결과 실루엣 계수는 **`{sil_score:.3f}`** 입니다.")
        fig1 = px.scatter(df_model, x='Sleep Duration', y='Quality of Sleep', color='Cluster_Name', title="K-Means 군집 분포")
        st.plotly_chart(fig1, use_container_width=True)
        fig2 = px.box(df_model, x='Cluster_Name', y='Stress Level', color='Cluster_Name', title="군집별 인지 피로도(스트레스) 박스플롯")
        st.plotly_chart(fig2, use_container_width=True)

    render_step_content(2, [step1, step2])

# -----------------------------------------------------------------------------
# 6. 모델 원리 페이지 (대화 2단계)
# -----------------------------------------------------------------------------
elif menu == "모델 원리":
    st.title("🔍 모델 원리")
    st.write("이 페이지에서는 K-Means 군집화 모델이 군집 중심을 구하는 원리를 설명합니다.")
    
    def step1():
        st.markdown("### 🧑‍🎓 학생")
        st.info("K-Means 군집 분석 모델이 3개의 군집 중심을 어떤 단계로 업데이트하는지 설명해 줘.")

    def step2():
        st.markdown("### 🤖 AI")
        st.markdown("""
        1. **초기 중심점 설정**: 정규화된 2차원 공간에 3개의 무작위 중심점을 잡습니다.
        2. **군집 할당**: 각 데이터마다 가장 가까운 중심점까지의 유클리드 거리를 계산해 군집으로 묶습니다.
        3. **중심점 이동**: 군집에 속한 데이터들의 좌표 평균값으로 중심 위치를 다시 계산합니다.
        4. **수렴**: 중심 위치에 변화가 없을 때까지 반복하여 최종 수면 군집을 완성합니다.
        """)

    render_step_content(2, [step1, step2])

# -----------------------------------------------------------------------------
# 7. 해석 한계 페이지 (대화 2단계)
# -----------------------------------------------------------------------------
elif menu == "해석 한계":
    st.title("⚠️ 해석·한계")
    st.write("이 페이지에서는 모델 분석 결과의 해석과 한계점을 수치로 정리합니다.")
    
    def step1():
        st.markdown("### 🧑‍🎓 학생")
        st.info("이 분석 결과를 개인의 상태 판단이나 건강 진단에 그대로 쓰면 안 되는 이유를 수치로 알려 줘.")

    def step2():
        st.markdown("### 🤖 AI")
        st.markdown("""
        * **동일 군집 내 개인차**: 단수면 군집 내부에서도 스트레스 지수가 4점에서 9점까지 다양하게 분포함.
        * **특정 집단 편향**: 374명의 데이터 중 특정 직종 비율이 높아 모든 사람에게 그대로 적용하기 어려움.
        * **오남용 위험**: 수면 군집 분류만으로 개인의 정신 피로 수준을 함부로 단정하거나 진단해서는 안 됨.
        """)

    render_step_content(2, [step1, step2])

# -----------------------------------------------------------------------------
# 8. 구술 연습 페이지 (대화 2단계)
# -----------------------------------------------------------------------------
elif menu == "구술 연습":
    st.title("🗣️ 구술 연습")
    st.write("구술 평가 체크리스트 답변 연습 공간입니다.")
    
    def step1():
        st.markdown("### 🧑‍🎓 학생")
        st.info("구술 평가 2번 항목(데이터 살펴보기와 전처리)에 대한 6가지 질문의 핵심 답안을 정리해 줘.")

    def step2():
        st.markdown("### 🤖 AI")
        st.markdown("""
        1. **행·열 수**: 374행 13열 (결측치 219개)
        2. **결측치 vs 이상치**: 장애 없음(None) 결측치 vs 4시간 미만 수면 등의 이상치
        3. **처리 기준**: 219개 결측치는 '정상 수면' 의미로 보존, 이상치는 정규화 적용
        4. **편향 요소**: 특정 전문 직종에 데이터가 치우친 편향 존재
        5. **전처리 전후**: 행 수 374행 유지, 데이터 왜곡 방지
        6. **커밋/로그**: `code/preprocess.py` 커밋 완료 및 결정 로그 기록
        """)

    render_step_content(2, [step1, step2])
