import pandas as pd
import os

def main():
    # 1. 파일 경로 설정
    input_file = "Sleep_health_and_lifestyle_dataset.csv"
    output_file = "sleep_health_processed.csv"

    # 파일 존재 여부 확인
    if not os.path.exists(input_file):
        print(f"[오류] '{input_file}' 파일이 존재하지 않습니다. Kaggle에서 다운로드한 CSV 파일을 같은 폴더에 넣어주세요.")
        return

    # 2. Kaggle 데이터셋 불러오기
    print(" 데이터를 불러오는 중...")
    df = pd.read_csv(input_file)

    # 3. 연구 질문에 필요한 주요 열(Column) 선택
    # - 군집화 입력 변수: Sleep Duration, Quality of Sleep
    # - 비교/결과 변수: Stress Level
    # - 추가 관련 변수: Physical Activity Level, Heart Rate
    selected_columns = [
        'Person ID',
        'Sleep Duration',
        'Quality of Sleep',
        'Stress Level',
        'Physical Activity Level',
        'Heart Rate'
    ]

    df_selected = df[selected_columns]

    # 4. 데이터 요약 정보 확인
    print("\n=== 데이터 정보 ===")
    print(df_selected.info())

    print("\n=== 상위 5개 데이터 ===")
    print(df_selected.head())

    # 5. 깃허브에 올릴 정제된 CSV 파일 저장
    df_selected.to_csv(output_file, index=False)
    print(f"\n 정제된 데이터가 '{output_file}' 파일로 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    main()
