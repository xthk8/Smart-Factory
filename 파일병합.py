import pandas as pd
from glob import glob

file_names = glob("C:/Users/USER/Desktop/설비학회/Dataset_장비이상 조기탐지 AI 데이터셋/data/5공정_180sec/*.csv") # 모든 csv 파일을 대상으로 합니다
total = pd.DataFrame() # 빈 데이터프레임 하나를 생성합니다

for file_name in file_names:
    try:
        temp = pd.read_csv(file_name, sep=',', encoding='utf-8') # csv 파일을 하나씩 열어 임시 데이터프레임으로 생성합니다
        total = pd.concat([total, temp]) # 전체 데이터프레임에 추가하여 넣습니다
    except Exception as e:
        print(f"Error with file {file_name}: {str(e)}") # 오류 발생 시, 해당 파일 이름과 오류 메시지를 출력합니다

total.to_csv("C:/Users/USER/Desktop/설비학회/Dataset_장비이상 조기탐지 AI 데이터셋/data/5공정_180sec/total.csv")