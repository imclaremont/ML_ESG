import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

def read_co2_data(filename):
    """
    ESG 관련 CSV 파일을 읽는 함수
    - UTF-8 인코딩으로 시도하고 실패하면 CP949로 시도한다.
    """
    encodings = ['utf-8', 'cp949']
    for encoding in encodings:
        try:
            df = pd.read_csv(filename, encoding=encoding)
            print(f"Successfully read file using {encoding} encoding.")
            return df
        except Exception as e:
            print(f"Failed to read with {encoding}: {e}")
    
    raise ValueError("Failed to read the file with any known encoding.")

# 파일 경로 설정
data_folder = "data"
figure_folder = "figure"
os.makedirs(data_folder, exist_ok=True)  # data 폴더가 없으면 생성
os.makedirs(figure_folder, exist_ok=True)  # figure 폴더가 없으면 생성

# 분석할 파일 선택
filename = "ESG_2022.csv"  # 사용할 파일을 변경 가능
raw_data_file = os.path.join(data_folder, filename)
processed_data_file = os.path.join(data_folder, f"processed_{filename}")

# 파일 읽기
df = read_co2_data(raw_data_file)

# 필수 컬럼 확인
df_columns = df.columns
if "year" not in df_columns:
    raise ValueError("Required column 'year' is missing in the dataset.")

# 'Net Emissions' 컬럼 확인 및 표준화
if "Net Emissions" in df_columns:
    df.rename(columns={"Net Emissions": "net emission"}, inplace=True)

# 숫자형 데이터 선택 (year 제외) 및 값이 전부 0인 컬럼 제거
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_columns.remove("year")
df = df.loc[:, (df != 0).any(axis=0)]  # 모든 값이 0인 컬럼 제거
numeric_columns = [col for col in numeric_columns if col in df.columns]

# 결측치 확인 및 처리
print("\nMissing values before handling:\n", df.isnull().sum())
df = df.dropna()  # 결측치가 있는 행 제거
print("\nMissing values after handling:\n", df.isnull().sum())

# 데이터 타입 확인 및 변환
df['year'] = df['year'].astype(int)  # 연도를 정수형으로 변환
print("\nUpdated Data Types:\n", df.dtypes)

# 전처리된 데이터 저장
df.to_csv(processed_data_file, index=False, encoding='utf-8')
print(f"Processed data saved to {processed_data_file}")

# 전처리된 데이터 재로드
df = read_co2_data(processed_data_file)

# 다중 선형 회귀 모델 학습
X = df[['year']].values  # 입력 변수 (년도)
models = {}
predictions = {}

for col in numeric_columns:
    y = df[col].values  # 출력 변수 (각 배출량 컬럼)
    model = LinearRegression()
    model.fit(X, y)
    models[col] = model

    # 향후 10년 예측
    years_future = np.arange(df['year'].max() + 1, df['year'].max() + 11).reshape(-1, 1)
    predictions[col] = model.predict(years_future)

# 예측 결과 추가
df_future = pd.DataFrame({"year": years_future.flatten()})
for col in numeric_columns:
    df_future[col] = predictions[col]

print("\nPredicted Emissions for Next 10 Years:\n", df_future)

# 데이터 시각화 (값이 높은 것과 낮은 것을 나누어 하나의 파일에 두 개의 그래프 표시)
median_value = df[numeric_columns].median().median()  # 중앙값을 기준으로 분류
high_value_columns = [col for col in numeric_columns if df[col].median() >= median_value and col not in ['chemical industry', 'mineral industry']]
low_value_columns = [col for col in numeric_columns if df[col].median() < median_value or col in ['chemical industry', 'mineral industry']]

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
unique_colors = plt.cm.get_cmap("tab20", len(numeric_columns))  # 더 다양한 색상을 활용
color_map = {col: unique_colors(i) for i, col in enumerate(numeric_columns)}  # 컬럼별 색상 매핑

# 높은 값들의 그래프
for col in high_value_columns:
    axes[0].plot(df['year'], df[col], marker='o', linestyle='-', color=color_map[col], label=f'Actual {col}')
    axes[0].plot(df_future['year'], df_future[col], marker='s', linestyle='--', color=color_map[col], label=f'Predicted {col}')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('High Emission Values')
axes[0].set_title(f'High Yearly Emissions with Predictions ({filename})')
axes[0].legend()
axes[0].grid(True)

# 낮은 값들의 그래프
for col in low_value_columns:
    axes[1].plot(df['year'], df[col], marker='o', linestyle='-', color=color_map[col], label=f'Actual {col}')
    axes[1].plot(df_future['year'], df_future[col], marker='s', linestyle='--', color=color_map[col], label=f'Predicted {col}')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Low Emission Values')
axes[1].set_title(f'Low Yearly Emissions with Predictions ({filename})')
axes[1].legend()
axes[1].grid(True)

# 그래프 이미지 저장
image_file_combined = os.path.join(figure_folder, f"combined_emissions_prediction_{filename}.png")
plt.savefig(image_file_combined)
print(f"Graph saved as {image_file_combined}")
plt.show()