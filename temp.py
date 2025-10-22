import pandas as pd

# 원본 엑셀 파일 읽기
df = pd.read_excel("./data/residual_analysis_results.xlsx")

# model_path 수정
df["model_path"] = (
    df["model_path"]
    .str.replace("./saved_models", "./models/ArimaModel", regex=False)
    .str.replace("\\", "/", regex=False)  # 역슬래시 → 슬래시
)

# CSV로 저장
df.to_csv("./data/residual_analysis_results.csv", index=False, encoding="utf-8-sig")

print("✅ 변환 완료: ./data/residual_analysis_results.csv")
