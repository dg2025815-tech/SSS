import streamlit as st

st.set_page_config(page_title="수면 패턴과 인지 피로도 분석", page_icon="💤", layout="wide")

st.title("🧪 데이터 과학 실습실: 수면 분석 프로젝트")
st.write("수면 패턴(수면 시간, 수면의 질)과 인지 피로도(`Stress Level`)의 관계를 K-Means 군집 분석을 통해 탐구하는 실습실입니다.")
st.markdown("---")

st.subheader("① 주제 선정·데이터 수집")
st.info("수면 패턴에 따라 데이터 기반의 군집을 형성하고, 그룹 간 스트레스 지수의 차이가 나타나는지 분석합니다.")

st.subheader("② 데이터 시각화·인사이트 & 전처리")
st.info("결측치(`Sleep Disorder` 219개)를 'None(정상 수면)'으로 대체 보존하고 Min-Max Scale 정규화를 수행합니다.")

st.subheader("③ 모델 후보·베이스라인")
st.info("군집화 모델 후보(K-Means, DBSCAN, 계층적 군집화)를 비교하고 K-Means(K=3)를 최종 채택했습니다.")
