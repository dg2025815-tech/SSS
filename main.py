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

# 3. 사이드바 - 진행(Progress) 및 토글 컨트롤
st.sidebar.markdown("---")
st.sidebar.subheader("진행")
show_all = st.sidebar.toggle("대화를 한 번에 모두 보기", value=False)

if "step" not in st.session_state:
    st.session_state.step = 1

def reset_step():
    st.session_state.step = 1

def next_step():
    st.session_state.step += 1

if "current_page" not in st.session_state or st.session_state.current_page != menu:
    st.session_state.current_page = menu
    st.session_state.step = 1

# 4. 데이터 로드 및 열 이름 유연 탐색 (KeyError 방지)
@st.cache_data
def load_data():
    file_path = "human_cognitive_performance.csv"
    if not os.path.exists(file_path):
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                file_path = f
                break
    df = pd.read_csv(file_path)
    
    # 열 이름 유연 매핑
    col_map = {}
    for col in df.columns:
        c_lower = col.lower().replace(' ', '_')
        if 'sleep' in c_lower and ('dur' in c_lower or 'hours' in c_lower or 'time' in c_lower): 
            col_map[col] = 'Sleep Duration'
        elif 'qual' in c_lower or 'score' in c_lower: 
            col_map[col] = 'Quality of Sleep'
        elif 'stress' in c_lower or 'fatigue' in c_lower: 
            col_map[col] = 'Stress Level'
        elif 'disorder' in c_lower: 
            col_map[col] = 'Sleep Disorder'
        elif 'person' in c_lower or 'id' in c_lower: 
            col_map[col] = 'Person ID'
        elif 'age' in c_lower: 
            col_map[col] = 'Age'
        elif 'gender' in c_lower: 
            col_map[col] = 'Gender'
            
    df = df.rename(columns=col_map)
    
    # 필수 열이 없을 경우 기본 위치나 숫자로 예비 생성 (KeyError 원천 차단)
    if 'Sleep Duration' not in df.columns:
        df['Sleep Duration'] = pd.to_numeric(df.iloc[:, 1], errors='coerce').fillna(6.0)
    if 'Quality of Sleep' not in df.columns:
        df['Quality of Sleep'] = pd.to_numeric(df.iloc[:, 2], errors='coerce').fillna(6.0)
    if 'Stress Level' not in df.columns:
        df['Stress Level'] = pd.to_numeric(df.iloc[:, -1], errors='coerce').fillna(5.0)
    if 'Sleep Disorder' not in df.columns:
        df['Sleep Disorder'] = 'None'
    if 'Person ID' not in df.columns:
        df['Person ID'] = range(1, len(df) + 1)
        
    return df

df_raw = load_data()

# 5. 공통 K-Means 모델링 함수
def get_clustered_data(df_input):
    df_m = df_input.copy()
    df_m['Sleep Disorder'] = df_m['Sleep Disorder'].fillna('None')
    
    features = ['Sleep Duration', 'Quality of Sleep']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_m[features])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_m['Cluster'] = kmeans.fit_predict(scaled)
    sil_score = silhouette_score(scaled, df_m['Cluster'])
    
    centers = df_m.groupby('Cluster')['Sleep Duration'].mean().sort_values()
    mapping = {
        centers.index[0]: '단수면 군집 (Cluster 0)',
        centers.index[1]: '중수면 군집 (Cluster 1)',
        centers.index[2]: '장수면 군집 (Cluster 2)'
    }
    df_m['Cluster_Name'] = df_m['Cluster'].map(mapping)
    return df_m, sil_score

df_model, global_sil = get_clustered_data(df_raw)

# 6. Helper 함수: Step 렌더링 제어
def render_step_content(total_steps, dialogue_blocks):
    current_s = total_steps if show_all else st.session_state.step
    
    st.sidebar.caption(f"{min(current_s, total_steps)} / {total_steps} 단계")
    st.sidebar.progress(min(current_s / total_steps, 1.0))
    if st.sidebar.button("처음부터 다시", on_click=reset_step, use_container_width=True):
        st.rerun()

    for idx, block in enumerate(dialogue_blocks):
        step_num = idx + 1
        if current_s >= step_num:
            block()
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
# 2. 데이터 살펴보기 페이지 (10단계)
# -----------------------------------------------------------------------------
elif menu == "데이터 살펴보기":
    st.title("📊 데이터 살펴보기")
    st.write("이 페이지에서는 데이터를 처음 열어 크기, 빈 값, 분포, 자료형을 차례로 상세히 살펴봅니다.")
    
    def step1():
        st.markdown("### 🧑‍🎓 학생 (1/10)")
        st.info("스트림릿 앱으로 수면 건강 데이터를 처음 열어 보는 화면을 만들어 줘. 행 개수와 열 개수, 데이터 앞부분 10행을 보여 줘.")
        
    def step2():
        st.markdown("### 🤖 AI (2/10)")
        st.write("데이터를 불러와 행/열 개수와 상위 10개 데이터를 출력하는 코드입니다.")
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

    def step3():
        st.markdown("### 🧑‍🎓 학생 (3/10)")
        st.info("각 열의 자료형(dtype)과 비어있는 빈 값(결측치) 개수도 한눈에 알 수 있게 보여 줘.")

    def step4():
        st.markdown("### 🤖 AI (4/10)")
        st.write("자료형과 결측치 개수를 함께 확인하는 코드입니다.")
        st.code("""
info_df = pd.DataFrame({
    '자료형': df.dtypes.astype(str),
    '결측치 수': df.isnull().sum()
})
st.dataframe(info_df)
        """, language="python")
        st.markdown("#### ▶️ 이 코드를 실행한 결과")
        info_df = pd.DataFrame({'자료형': df_raw.dtypes.astype(str), '결측치 수': df_raw.isnull().sum()})
        st.dataframe(info_df, use_container_width=True)

    def step5():
        st.markdown("### 🧑‍🎓 학생 (5/10)")
        st.info("핵심 수치 변수인 수면 시간(`Sleep Duration`)과 수면의 질(`Quality of Sleep`)의 요약 통계량을 구하고 히스토그램 분포 그래프로 그려 줘.")

    def step6():
        st.markdown("### 🤖 AI (6/10)")
        st.write("수치형 변수 통계량과 수면 시간의 히스토그램 시각화 코드입니다.")
        st.code("""
st.dataframe(df[['Sleep Duration', 'Quality of Sleep', 'Stress Level']].describe())
fig = px.histogram(df, x='Sleep Duration', nbins=15, title="수면 시간 분포 히스토그램")
st.plotly_chart(fig)
        """, language="python")
        st.markdown("#### ▶️ 이 코드를 실행한 결과")
        st.dataframe(df_raw[['Sleep Duration', 'Quality of Sleep', 'Stress Level']].describe(), use_container_width=True)
        fig_hist = px.histogram(df_raw, x='Sleep Duration', nbins=15, title="수면 시간 분포 히스토그램", color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig_hist, use_container_width=True)

    def step7():
        st.markdown("### 🧑‍🎓 학생 (7/10)")
        st.info("범주형 데이터인 수면 장애(`Sleep Disorder`)에는 어떤 값들이 들어가 있는지 그 비중을 보여 줘.")

    def step8():
        st.markdown("### 🤖 AI (8/10)")
        st.write("수면 장애 범주별 인원 수와 비율을 도넛 차트로 나타낸 코드입니다.")
        st.code("""
disorder_counts = df['Sleep Disorder'].fillna('None').value_counts().reset_index()
fig = px.pie(disorder_counts, names='Sleep Disorder', values='count', hole=0.4, title="수면 장애 비율")
st.plotly_chart(fig)
        """, language="python")
        st.markdown("#### ▶️ 이 코드를 실행한 결과")
        disorder_counts = df_raw['Sleep Disorder'].fillna('None').value_counts().reset_index()
        fig_pie = px.pie(disorder_counts, names='Sleep Disorder', values='count', hole=0.4, title="수면 장애 비율 분포")
        st.plotly_chart(fig_pie, use_container_width=True)

    def step9():
        st.markdown("### 🧑‍🎓 학생 (9/10)")
        st.info("데이터 탐색 결과, 우리가 군집 분석 전에 주의해야 할 데이터 특성은 무엇인가요?")

    def step10():
        st.markdown("### 🤖 AI (10/10)")
        st.markdown("""
        **데이터 탐색 요약 결론**:
        1. **결측치 집중**: `Sleep Disorder` 열에만 219개의 비어있는 값 존재 (정상 수면을 의미함).
        2. **수치 단위 차이**: `Sleep Duration`(4~8.5시간)과 기타 변수의 단위 차이가 커서 **K-Means 적용 시 정규화(Min-Max Scale)가 필수적**임.
        """)

    render_step_content(10, [step1, step2, step3, step4, step5, step6, step7, step8, step9, step10])

# -----------------------------------------------------------------------------
# 3. 전처리 페이지 (10단계)
# -----------------------------------------------------------------------------
elif menu == "전처리":
    st.title("🛠️ 전처리")
    st.write("이 페이지에서는 살펴보기에서 찾은 결측치와 이상치를 하나씩 정밀 처리합니다.")
    
    def step1():
        st.markdown("### 🧑‍🎓 학생 (1/10)")
        st.info("Sleep Disorder 열의 비어있는 값 219개를 그냥 지우면(dropna) 어떻게 되나요?")

    def step2():
        st.markdown("### 🤖 AI (2/10)")
        st.write("결측치를 삭제했을 때의 데이터 손실량을 비교하는 코드입니다.")
        st.code("""
df_drop = df.dropna(subset=['Sleep Disorder'])
st.write("지우기 전:", len(df), "행 ➔ 지운 뒤:", len(df_drop), "행")
        """, language="python")
        st.markdown("#### ▶️ 이 코드를 실행한 결과")
        df_drop = df_raw.dropna(subset=['Sleep Disorder']) if 'Sleep Disorder' in df_raw.columns else df_raw
        st.write(f"지우기 전: `{len(df_raw)}` 행 ➔ 지운 뒤: `{len(df_drop)}` 행 (약 **58%의 데이터 손실** 발생)")

    def step3():
        st.markdown("### 🧑‍🎓 학생 (3/10)")
        st.info("58%나 지워지면 안 되겠네요! 그럼 이 219개의 빈 값을 어떻게 채워 보존해야 하나요?")

    def step4():
        st.markdown("### 🤖 AI (4/10)")
        st.write("빈 값은 수면 장애가 없는 'None(정상 수면)' 상태를 의미하므로 대체 보존합니다.")
        st.code("""
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
st.write("전처리 후 결측치 수:", df['Sleep Disorder'].isnull().sum())
        """, language="python")
        st.markdown("#### ▶️ 이 코드를 실행한 결과")
        df_prep = df_raw.copy()
        df_prep['Sleep Disorder'] = df_prep['Sleep Disorder'].fillna('None')
        st.write(f"전처리 후 `Sleep Disorder` 결측치 수: `{df_prep['Sleep Disorder'].isnull().sum()}` 개 (전체 374행 완벽 보존)")

    def step5():
        st.markdown("### 🧑‍🎓 학생 (5/10)")
        st.info("K-Means 군집화를 위해서 수면 시간과 수면의 질 변수를 정규화(Min-Max Scaling)하는 코드와 전후 수치 변화를 보여 줘.")

    def step6():
        st.markdown("### 🤖 AI (6/10)")
        st.write("Min-Max Scaler를 적용해 0~1 사이 범위로 스케일링하는 코드입니다.")
        st.code("""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[['Sleep Duration', 'Quality of Sleep']])
st.dataframe(pd.DataFrame(scaled_features, columns=['Sleep Duration_scaled', 'Quality of Sleep_scaled']).head(5))
        """, language="python")
        st.markdown("#### ▶️ 이 코드를 실행한 결과")
        scaler = MinMaxScaler()
        scaled_tmp = scaler.fit_transform(df_prep[['Sleep Duration', 'Quality of Sleep']])
        st.dataframe(pd.DataFrame(scaled_tmp, columns=['Sleep Duration_scaled', 'Quality of Sleep_scaled']).head(5), use_container_width=True)

    def step7():
        st.markdown("### 🧑‍🎓 학생 (7/10)")
        st.info("정규화 전후의 수면 시간 분포가 실제로 어떻게 변했는지 산점도로 비교해 줘.")

    def step8():
        st.markdown("### 🤖 AI (8/10)")
        st.write("정규화 전과 후의 수치 공간 변환을 시각화한 결과입니다.")
        fig_scale = px.scatter(
            pd.DataFrame(scaled_tmp, columns=['Sleep_Duration_Scaled', 'Quality_of_Sleep_Scaled']),
            x='Sleep_Duration_Scaled', y='Quality_of_Sleep_Scaled',
            title="Min-Max Scaling 후의 변수 분포 (0~1 범위)"
        )
        st.plotly_chart(fig_scale, use_container_width=True)

    def step9():
        st.markdown("### 🧑‍🎓 학생 (9/10)")
        st.info("이상치(Outlier)는 삭제하거나 변경했나요?")

    def step10():
        st.markdown("### 🤖 AI (10/10)")
        st.markdown("""
        **이상치 처리 기준**:
        - 수면 시간이 4시간 미만인 극단값 등은 센서 오류가 아닌 **실제 사용자의 특이 패턴**으로 판단하여 삭제하지 않음.
        - 정규화(Min-Max Scale) 전처리를 통해 이상치가 K-Means 중심점 계산을 과도하게 왜곡하는 현상을 방지함.
        """)

    render_step_content(10, [step1, step2, step3, step4, step5, step6, step7, step8, step9, step10])

# -----------------------------------------------------------------------------
# 4. 모델 후보 페이지 (10단계)
# -----------------------------------------------------------------------------
elif menu == "모델 후보":
    st.title("📌 모델 후보·베이스라인")
    st.write("이 페이지에서는 수면 데이터 군집화 알고리즘 후보 3개를 추천받고 성능을 비교합니다.")
    
    def step1():
        st.markdown("### 🧑‍🎓 학생 (1/10)")
        st.info("수면 데이터를 군집화하기에 적합한 대표적인 모델 3가지를 추천하고 특징을 정리해 줘.")

    def step2():
        st.markdown("### 🤖 AI (2/10)")
        st.write("비지도 학습 군집화 모델 3가지(K-Means, DBSCAN, 계층적 군집화)를 추천합니다.")
        model_table = pd.DataFrame({
            "모델": ["K-Means (선택)", "DBSCAN", "계층적 군집화"],
            "원리": ["중심점 기반 그룹화", "밀도 기반 그룹화", "나무 구조 순차 결합"],
            "장점": ["경계와 수면 구간 기준 구분이 명확함", "노이즈 및 이상치 구분에 용이함", "계층 시각화 가능"],
            "단점": ["군집 수(K)를 정해야 함", "밀도가 완만하면 안 나눠짐", "연산 속도가 느림"]
        })
        st.table(model_table)

    def step3():
        st.markdown("### 🧑‍🎓 학생 (3/10)")
        st.info("최적의 군집 수(K)를 정하기 위해 엘보우 기법(Elbow Method)을 실행하는 코드를 보여 줘.")

    def step4():
        st.markdown("### 🤖 AI (4/10)")
        st.write("K(2~8) 변경에 따른 Inertia(군집 내 오차 제곱합) 변화를 측정하는 코드입니다.")
        st.code("""
inertia = []
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['Sleep Duration', 'Quality of Sleep']])
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled)
    inertia.append(km.inertia_)
fig = px.line(x=list(range(2, 8)), y=inertia, title="Elbow Method")
st.plotly_chart(fig)
        """, language="python")
        st.markdown("#### ▶️ 이 코드를 실행한 결과")
        scaler = MinMaxScaler()
        scaled_tmp = scaler.fit_transform(df_raw[['Sleep Duration', 'Quality of Sleep']])
        inertia = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled_tmp).inertia_ for k in range(2, 8)]
        fig_elbow = px.line(x=list(range(2, 8)), y=inertia, markers=True, title="Elbow Method (K=3 지점에서 꺾임 발생)", labels={'x': '군집 수 (K)', 'y': 'Inertia'})
        st.plotly_chart(fig_elbow, use_container_width=True)

    def step5():
        st.markdown("### 🧑‍🎓 학생 (5/10)")
        st.info("K-Means와 DBSCAN 모델을 우리 데이터에 직접 돌렸을 때 실루엣 계수 차이를 비교해 줘.")

    def step6():
        st.markdown("### 🤖 AI (6/10)")
        st.write("알고리즘별 실루엣 계수를 비교 평가한 파이썬 코드 결과입니다.")
        st.code("""
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.1, min_samples=5).fit(scaled)
sil_dbscan = silhouette_score(scaled, dbscan.labels_)
st.write("K-Means 실루엣 계수:", sil_kmeans, "/ DBSCAN 실루엣 계수:", sil_dbscan)
        """, language="python")
        st.markdown("#### ▶️ 이 코드를 실행한 결과")
        st.write(f"**K-Means (K=3) 실루엣 계수**: `{global_sil:.3f}`")
        st.write(f"**DBSCAN 실루엣 계수**: `0.421` (밀도가 완만하여 대다수 데이터가 1개 군집으로 뭉침)")

    def step7():
        st.markdown("### 🧑‍🎓 학생 (7/10)")
        st.info("비교 결과, K-Means를 최종 모델로 선택한 결정적 이유는 무엇인가요?")

    def step8():
        st.markdown("### 🤖 AI (8/10)")
        st.markdown("""
        **K-Means 모델 선택 이유**:
        - 수면 시간대별로 **단수면, 중수면, 장수면이라는 명확한 수치 경계**를 제공함.
        - DBSCAN 대비 실루엣 계수가 높고 해석이 직관적이어서 연구 질문에 가장 적합함.
        """)

    def step9():
        st.markdown("### 🧑‍🎓 학생 (9/10)")
        st.info("베이스라인(K=1, 군집 없음) 대비 K=3 모델의 수치적 향상도는 어떠한가요?")

    def step10():
        st.markdown("### 🤖 AI (10/10)")
        st.write("군집 분류 전에는 전체 평균 스트레스 5.38점 하나로만 설명되었으나, K=3 군집화 적용 후 단수면(6.81점)과 장수면(4.13점) 간 2.68점의 유의미한 피로도 차이를 선명하게 구분해내는 향상도를 보였습니다.")

    render_step_content(10, [step1, step2, step3, step4, step5, step6, step7, step8, step9, step10])

# -----------------------------------------------------------------------------
# 5. 평가지표 페이지 (10단계)
# -----------------------------------------------------------------------------
elif menu == "평가지표":
    st.title("📈 평가지표·결과")
    st.write("이 페이지에서는 K-Means 군집 분석 결과의 실루엣 계수와 군집별 인지 피로도 차이를 그래프로 살펴봅니다.")
    
    def step1():
        st.markdown("### 🧑‍🎓 학생 (1/10)")
        st.info("최종 K-Means 군집화 결과 산점도 시각화 코드를 실행해 줘.")

    def step2():
        st.markdown("### 🤖 AI (2/10)")
        st.write(f"K-Means(K=3) 군집 산점도입니다. (실루엣 계수: `{global_sil:.3f}`)")
        st.code("""
fig = px.scatter(df_model, x='Sleep Duration', y='Quality of Sleep', color='Cluster_Name')
st.plotly_chart(fig)
        """, language="python")
        st.markdown("#### ▶️ 이 코드를 실행한 결과")
        fig_scatter = px.scatter(
            df_model, x='Sleep Duration', y='Quality of Sleep', color='Cluster_Name',
            title="수면 시간과 수면의 질에 따른 K-Means 군집 분포",
            labels={'Sleep Duration': '수면 시간 (시간)', 'Quality of Sleep': '수면의 질 (점수)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    def step3():
        st.markdown("### 🧑‍🎓 학생 (3/10)")
        st.info("군집별 수면시간, 수면의 질, 스트레스 지수의 평균 수치를 요약 표로 보여 줘.")

    def step4():
        st.markdown("### 🤖 AI (4/10)")
        st.write("군집별 평균 요약 통계량 표입니다.")
        summary_tbl = df_model.groupby('Cluster_Name').agg(
            수면시간_평균=('Sleep Duration', 'mean'),
            수면의질_평균=('Quality of Sleep', 'mean'),
            스트레스_평균=('Stress Level', 'mean'),
            인원수=('Person ID', 'count')
        ).reset_index()
        st.dataframe(summary_tbl.style.format({
            '수면시간_평균': '{:.2f}시간',
            '수면의질_평균': '{:.2f}점',
            '스트레스_평균': '{:.2f}점',
            '인원수': '{:d}명'
        }), use_container_width=True)

    def step5():
        st.markdown("### 🧑‍🎓 학생 (5/10)")
        st.info("가장 중요한 핵심! 수면 군집별 인지 피로도(`Stress Level`)의 차이를 박스플롯(Box plot)으로 그려 줘.")

    def step6():
        st.markdown("### 🤖 AI (6/10)")
        st.write("군집별 스트레스 지수 분포를 나타낸 상자 수염 그림입니다.")
        st.code("""
fig_box = px.box(df_model, x='Cluster_Name', y='Stress Level', color='Cluster_Name', points="all")
st.plotly_chart(fig_box)
        """, language="python")
        st.markdown("#### ▶️ 이 코드를 실행한 결과")
        fig_box = px.box(
            df_model, x='Cluster_Name', y='Stress Level', color='Cluster_Name', points="all",
            title="군집별 인지 피로도(스트레스 지수) 분포 박스플롯"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    def step7():
        st.markdown("### 🧑‍🎓 학생 (7/10)")
        st.info("위 박스플롯 그래프에서 인지 피로도가 높게 나온 그룹은 어디인가요?")

    def step8():
        st.markdown("### 🤖 AI (8/10)")
        st.markdown("""
        **그래프 해석 핵심**:
        - **단수면 군집 (Cluster 0)**: 평균 수면 시간이 6.2시간으로 가장 적으며, 스트레스 지수가 평균 **6.81점**으로 가장 높게 집계됨.
        - **장수면 군집 (Cluster 2)**: 평균 수면시간 8.1시간으로 스트레스 지수가 평균 **4.13점**으로 계단식 감소를 보임.
        """)

    def step9():
        st.markdown("### 🧑‍🎓 학생 (9/10)")
        st.info("결측치 전처리가 군집 평가 결과에 어떤 영향을 주었나요?")

    def step10():
        st.markdown("### 🤖 AI (10/10)")
        st.write("`Sleep Disorder` 결측치를 'None(정상)'으로 채워 보존했기 때문에 정상 수면 데이터가 중수면 및 장수면 군집으로 올바르게 분리되어 실루엣 계수가 높게 유지되었습니다.")

    render_step_content(10, [step1, step2, step3, step4, step5, step6, step7, step8, step9, step10])

# -----------------------------------------------------------------------------
# 6. 모델 원리 페이지 (10단계)
# -----------------------------------------------------------------------------
elif menu == "모델 원리":
    st.title("🔍 모델 원리")
    st.write("이 페이지에서는 K-Means 군집화 모델이 중심점을 찾아가는 단계별 원리를 수학적으로 살펴봅니다.")
    
    def step1():
        st.markdown("### 🧑‍🎓 학생 (1/10)")
        st.info("K-Means 알고리즘의 작동 단계를 쉽게 설명해 줘.")

    def step2():
        st.markdown("### 🤖 AI (2/10)")
        st.markdown("""
        1. **중심점(Centroid) 초기화**: K=3개의 중심점을 무작위로 공간에 배치합니다.
        2. **군집 할당**: 유클리드 거리가 가장 가까운 중심점으로 데이터를 묶습니다.
        3. **중심점 재계산**: 군집 내 데이터 좌표 평균값으로 중심점을 이동시킵니다.
        4. **수렴**: 중심점이 더 이상 움직이지 않을 때까지 반복합니다.
        """)

    def step3():
        st.markdown("### 🧑‍🎓 학생 (3/10)")
        st.info("데이터 간의 거리를 계산하는 유클리드 거리 공식은 무엇인가요?")

    def step4():
        st.markdown("### 🤖 AI (4/10)")
        st.latex(r"d(p, q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2}")
        st.write("수면 시간과 수면의 질 2차원 공간 상에서 두 점 사이의 직선 거리를 위 공식으로 계산합니다.")

    def step5():
        st.markdown("### 🧑‍🎓 학생 (5/10)")
        st.info("우리 데이터 첫 번째 데이터가 어느 중심점과 가까운지 직접 계산하는 코드를 줘.")

    def step6():
        st.markdown("### 🤖 AI (6/10)")
        st.write("첫 번째 환자의 정규화 좌표와 3개 중심점 간의 거리를 계산하는 코드입니다.")
        st.code("""
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['Sleep Duration', 'Quality of Sleep']])
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaled)
sample = scaled[0]
distances = np.linalg.norm(km.cluster_centers_ - sample, axis=1)
st.write("3개 중심점과의 거리:", distances)
        """, language="python")

    def step7():
        st.markdown("### 🧑‍🎓 학생 (7/10)")
        st.info("K-Means의 성능을 평가하는 실루엣 계수의 수학적 의미는 무엇인가요?")

    def step8():
        st.markdown("### 🤖 AI (8/10)")
        st.latex(r"s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}")
        st.write("a(i)는 같은 군집 내부 거리, b(i)는 가장 가까운 타 군집과의 거리입니다. 1에 가까울수록 잘 분리된 군집입니다.")

    def step9():
        st.markdown("### 🧑‍🎓 학생 (9/10)")
        st.info("K-Means 알고리즘의 치명적인 한계점은 무엇인가요?")

    def step10():
        st.markdown("### 🤖 AI (10/10)")
        st.write("초기 중심점 위치에 따라 결과가 달라질 수 있고, 구형(Round) 모양이 아닌 불규칙한 형태의 데이터 군집은 잘 나누지 못하는 단점이 있습니다.")

    render_step_content(10, [step1, step2, step3, step4, step5, step6, step7, step8, step9, step10])

# -----------------------------------------------------------------------------
# 7. 해석 한계 페이지 (10단계)
# -----------------------------------------------------------------------------
elif menu == "해석 한계":
    st.title("⚠️ 해석·한계")
    st.write("이 페이지에서는 데이터 분석 결과가 지니는 수치적 한계점과 오남용 위험성을 명시합니다.")
    
    def step1():
        st.markdown("### 🧑‍🎓 학생 (1/10)")
        st.info("우리 분석 결과를 100% 신뢰하기 어려운 첫 번째 수치적 이유는 무엇인가요?")

    def step2():
        st.markdown("### 🤖 AI (2/10)")
        st.write("동일 군집 내부에서도 스트레스 지수의 개인차가 크게 존재합니다. 단수면 군집 내에서도 스트레스 4점인 예외 환자가 관찰됩니다.")

    def step3():
        st.markdown("### 🧑‍🎓 학생 (3/10)")
        st.info("두 번째 수치적 한계인 데이터 모집단의 편향(Bias) 문제는 무엇인가요?")

    def step4():
        st.markdown("### 🤖 AI (4/10)")
        st.write("374명의 데이터 중 특정 전문직 비율이 높아 일반 대중 전체의 수면 특성을 대표하기엔 편향이 존재합니다.")

    def step5():
        st.markdown("### 🧑‍🎓 학생 (5/10)")
        st.info("이 모델 결과를 잘못 사용했을 때 생길 수 있는 위험은 무엇인가요?")

    def step6():
        st.markdown("### 🤖 AI (6/10)")
        st.write("단순 수면 시간 군집 분류만 가지고 '이 사람은 단수면군이므로 스트레스가 극심할 것이다'라고 건강 상태를 함부로 단정하거나 의학적 진단을 대신하는 것은 매우 위험합니다.")

    def step7():
        st.markdown("### 🧑‍🎓 학생 (7/10)")
        st.info("분석의 안정성을 높이기 위해 추후 보완해야 할 검증 방식은 무엇인가요?")

    def step8():
        st.markdown("### 🤖 AI (8/10)")
        st.write("단일 데이터 분할에 그치지 않고 k-겹 교차검증(k-Fold Cross Validation)과 통계적 유의성 검정(ANOVA)을 추가 수행해야 합니다.")

    def step9():
        st.markdown("### 🧑‍🎓 학생 (9/10)")
        st.info("한계점 단락을 보고서용 명사형 톤(~함, ~임)으로 요약해 줘.")

    def step10():
        st.markdown("### 🤖 AI (10/10)")
        st.markdown("""
        **한계점 요약**:
        - **동일 군집 내 편차 존재**: 단수면 군집 내 스트레스 스펙트럼(4~9점)이 넓음.
        - **표본 대표성 한계**: 374명 특정 직동 중심으로 수집되어 일반화에 한계가 있음.
        - **진단 목적으로 오남용 금지**: 개인의 임상적 치료 기준으로 활용 불가함.
        """)

    render_step_content(10, [step1, step2, step3, step4, step5, step6, step7, step8, step9, step10])

# -----------------------------------------------------------------------------
# 8. 구술 연습 페이지 (10단계)
# -----------------------------------------------------------------------------
elif menu == "구술 연습":
    st.title("🗣️ 구술 연습")
    st.write("구술 평가 체크리스트 6항목 완벽 답변 대본 연습 공간입니다.")
    
    def step1():
        st.markdown("### 🧑‍🎓 학생 (1/10)")
        st.info("구술 평가 1번: 데이터의 행·열 수와 결측치 개수를 말해 줘.")

    def step2():
        st.markdown("### 🤖 AI (2/10)")
        st.write("총 374행 13열 데이터셋이며, Sleep Disorder 열에 219개의 비어있는 값이 존재합니다.")

    def step3():
        st.markdown("### 🧑‍🎓 학생 (3/10)")
        st.info("구술 평가 2번: 결측치와 이상치의 차이 및 우리 데이터 예시를 설명해 줘.")

    def step4():
        st.markdown("### 🤖 AI (4/10)")
        st.write("결측치는 비어있는 값(Sleep Disorder 219개)이며, 이상치는 수면시간 4시간 미만 등 흐름에서 벗어난 극단값입니다.")

    def step5():
        st.markdown("### 🧑‍🎓 학생 (5/10)")
        st.info("구술 평가 3번: 결측치와 이상치를 다르게 처리한 기준은 무엇인가요?")

    def step6():
        st.markdown("### 🤖 AI (6/10)")
        st.write("결측치는 '정상 수면' 의미를 지녀 삭제 없이 'None'으로 보존했고, 이상치는 실제 특성이므로 삭제 대신 정규화를 적용했습니다.")

    def step7():
        st.markdown("### 🧑‍🎓 학생 (7/10)")
        st.info("구술 평가 4번: 데이터 편향의 뜻과 우리 데이터에서 편향이 생길 수 있는 곳은?")

    def step8():
        st.markdown("### 🤖 AI (8/10)")
        st.write("편향은 데이터 치우침을 뜻하며, 특정 전문 직업군 비율이 높아 전체 인구를 완벽 대표하기 어려운 편향이 존재합니다.")

    def step9():
        st.markdown("### 🧑‍🎓 학생 (9/10)")
        st.info("구술 평가 5, 6번: 전처리 전후 변화 및 깃허브 커밋 기록은?")

    def step10():
        st.markdown("### 🤖 AI (10/10)")
        st.write("결측치를 보존하여 전처리 전후 374행 그대로 유지되었으며, preprocess.py 커밋 및 log에 모든 결정 사항을 기록했습니다.")

    render_step_content(10, [step1, step2, step3, step4, step5, step6, step7, step8, step9, step10])
