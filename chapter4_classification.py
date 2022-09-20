### logistic Regression
## 데이터 준비하기
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()
print(pd.unique(fish['Species'])) # 생선 종류가 무엇이 있는지 확인

# input, target 분할
fish_input = fish[['Weight', 'Length', 'Diagonal','Height','Width']].to_numpy() # 생선 종을 제외한 나머지 열을 입력 데이터로 추출
fish_target = fish['Species'].to_numpy() # 2차원배열은 입력만 만족, target은 2차원 아니어도 됨

# train, test 분할
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state = 42)

# StandardScaler를 이용한 scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

## k-nearest 분류기의 확률 예측
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors = 3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled,train_target)) #0.8907563025210085
print(kn.score(test_scaled, test_target)) #0.85

# target 데이터에 2개 이상의 클래스가 포함된 문제를 다중분류(multi-class classification) / 타깃값이 숫자가 아닌 문자열이라면 알파벳 순으로 자동 정렬됨
print(kn.classes_) # 알파벳 순 정렬

print(kn.predict(test_scaled[:5])) # 5개 sample에 대한 분류 예측
# sklearn 분류모델은 predict_proba() 메소드로 클래스별 확률값을 반환
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals = 4)) #decimals 매개변수는 자리수(네번째 자리까지 표기)

# 계산한 확률이 가장 가까운 이웃의 비율이 맞는지 확인
distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes]) #[['Roach' 'Perch' 'Perch']] --> 값은 정확하게 나왔으나 3개의 최근접 이웃을 사용하기에 값 범위가 0, 1/3, 2/3, 1 뿐이다 --> 다양하게 표현할 필요가 있음

## 로지스틱 회귀 -> 이름은 회귀지만 분류모델 / 선형 방정식 학습 / 단, 확률의 값을 가지려면 0~1 사이의 값을 가져야 함 --> 시그모이드 함수 이용 (z가 무한하게 큰 음수이면 0, z가 무한하게 큰 양수이면 1)

# 시그모이드 함수
import numpy as np
import matplotlib.pyplot as plot
z  = np.arange(-5,5,0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show(block=True)

# 간단한 이진분류에서는 확률이 0.5 이상이면 양성, 작으면 음성으로 판단(정확하게 0.5이면 음성으로 판단, 라이브러리마다 다름)
# boolean indexing
char_arr = np.array(['A',',B','C','D','E'])
print(char_arr[[True, False, True, False, False]]) # True에 해당하는 a, c만 출력