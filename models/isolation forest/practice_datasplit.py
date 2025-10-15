import pandas as pd
import numpy as np

# 🔹 1. 데이터 불러오기
df = pd.read_excel('./data2/train_final.xlsx')
df.info()

df.columns

# 🔹 2. date + time 컬럼을 합쳐 datetime 생성
df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))

# 🔹 3. 금형코드(mold_code) + 시간(datetime) 순으로 정렬
df = df.sort_values(['mold_code', 'datetime']).reset_index(drop=True)

# 🔹 4. 전체 행 수 확인
n = len(df)

# 🔹 5. 비율 기준 분할 (시계열 순서 유지)
train_df = df.iloc[:int(n * 0.7)]
valid_df = df.iloc[int(n * 0.7):int(n * 0.85)]
test_df  = df.iloc[int(n * 0.85):]


# 🔹 6. 파일로 저장 (csv)
train_df.to_csv("train_split.csv", index=False)
valid_df.to_csv("valid_split.csv", index=False)
test_df.to_csv("test_split.csv", index=False)

print("✅ 파일 생성 완료!")
print("train_split.csv, valid_split.csv, test_split.csv 저장 완료.")


