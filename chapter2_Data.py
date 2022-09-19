'''
지도학습 : 데이터(입력)와 정답(타깃)을 부여하고 이 둘을 합쳐 훈련 데이터라고 함
비지도학습 : 데이터의 정답을 부여하지 않고 입력 데이터만을 사용

훈련 데이터로 fitting한 이후 같은 데이터로 테스트를 진행한다면 모두 맞추는 것이 당연한 결과 -> 과적합 / 때문에 머신러닝 성능을 제대로 평가하려면 훈련 데이터와 평가에 사용할 데이터가 달라야 함
'''

지도학습 : 데이터(입력)와 정답(타깃)을 부여하고 이 둘을 합쳐 훈련 데이터라고 함
비지도학습 : 데이터의 정답을 부여하지 않고 입력 데이터만을 사용

훈련 데이터로 fitting한 이후 같은 데이터로 테스트를 진행한다면 모두 맞추는 것이 당연한 결과 -> 과적합 / 때문에 머신러닝 성능을 제대로 평가하려면 훈련 데이터와 평가에 사용할 데이터가 달라야 함

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

fish_data = [[l,w] for l, w in zip(fish_length,fish_weight)]
fish_target = [1]*35 + [0] * 14

# 처음 35개를 훈련 세트, 14개를 테스트 셋으로 활용
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()

train_input = fish_data[:35]
train_target = fish_target[:35]
test_input = fish_data[35:]
test_target = fish_target[35:]

kn = kn.fit(train_input,train_target)
kn.score(test_input,test_target) # 0.0 -> 샘플링 편향으로 인한 정확도 0 / 데이터끼리 골고루 섞이게 만들어야 함 --> Numpy를 이용

# Numpy : 고차원 배열을 손쉽게 만들고 조작할 수 있는 간편한 도구 제공
# 배열의 시작점이 왼쪽 위

# 생선 데이터의 2차원 넘파이 배열 변환
import numpy as np

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
print(input_arr)
# 랜덤하게 샘플을 선택해 훈련셋과 테스트셋을 만들 차례, 하지만 각 배열에서 같은 위치는 함께 선택되어야함

np.random.seed(42)
index = np.arange(49) # 0~48까지 1씩 증가하는 배열을 만들어줌
np.random.shuffle(index) # 셔플

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

import matplotlib.pyplot as plt
plt.scatter(train_input[:,0],  train_input[:,1])
plt.scatter(test_input[:, 0], test_input[: , 1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show(block=True) # 양쪽에 도미와 빙어가 골고루 섞여있음 -> 다시 fitting

kn = kn.fit(train_input, train_target)
kn.score(test_input, test_target) # 1.0
kn.predict(test_input) # array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0]) --> 반환값은 파이썬 리스트가 아닌 넘파이 배열 / sklearn의 모든 결과는 numpy 배열
test_target # array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0])

### 데이터 전처리

import numpy as np

np.column_stack(([1,2,3], [4,5,6])) # 전달받은 리스트를 일렬로 세운 후 차례대로 나란히 연결(컬럼방향)

fish_data = np.column_stack((fish_length,fish_weight))
print(fish_data[:5])

# np.ones와 np.zeros를 이용한 타겟 생성

print(np.ones(5))

# np.column_stack() = column 방향으로 연결(1차원 + 1차원 -> 2차원) / np.concatenate() = 첫번째 차원에 따라 배열을 연결(1차원 + 1차원 -> 1차원)
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
print(fish_target) # numpy는 계산 부분이 c언어, c++로 개발되어 속도가 빠르므로 데이터 크기가 크다면 numpy를 이용

# sklearn을 이용한 train / test 분리 -> 앞에서는 인덱싱을 이용해서 분할하였으나 귀찮은 작업이므로 train_test_split()을 이용

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state = 42) # 기본적으로 25%의 값을 test set으로 분리
print(train_input.shape, test_input.shape) #numpy 배열의 shape 속성으로 입력 데이터 크기 출력
print(train_target.shape, test_target.shape)

print(test_target) # shuffle 테스트 -> 본래 비율은 2.5:1 이지만 shuffle 결과 3.3:1로 약간의 편향이 존재 / stratify 매개변수를 이용해 target 데이터를 전달하면 클래스 비율에 맞게 분할

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify = fish_target, random_state = 42)
print(test_target) # 정확하게 분류

# 준비한 데이터를 이용한 k-Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(train_input,train_target)
kn.score(test_input,test_target) # 1.0

print(kn.predict([[25,150]])) # [25,150]인 데이터를 집어넣었더니 결과가 도미가 아닌 빙어로 분류됨

import matplotlib.pyplot as plt

plt.scatter(train_input[:,0],  train_input[:,1])
plt.scatter(25,150,marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show(block=True) # 샘플이 분명 도미데이터에 가깝지만 빙어로 분류 / kn은 이웃 샘플 중 다수인 클래스를 예측으로 사용

# 해당 포인트에서 가장 가까운 이웃을 찾아주는 함수 = kneighbors()
distances, indexes = kn.kneighbors([[25,150]])

plt.scatter(train_input[:,0],  train_input[:,1])
plt.scatter(25,150,marker = '^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker = 'D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show(block=True) # 산점도를 통해 보면 빙어쪽의 자료가 더 많이 포함되어 있음을 알 수 있다

# 산점도를 보면 도미로 직관적인 분류가 가능하지만 왜 빙어로 분류되어 있을까? -> x축과 y축의 간격이 다르기에 y축으로 조금만 멀어져도 매우 큰 값을 가지는 거리로 계산됨

plt.scatter(train_input[:,0],  train_input[:,1])
plt.scatter(25,150,marker = '^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker = 'D')
plt.xlim((0,1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show(block=True) # x와 y의 간격을 같게 설정하면 확실하게 알 수 있음 -> 두 특성의 스케일이 다르기에 이러한 문제가 발생

# 데이터를 표현하는 기준이 다르다면 알고리즘이 올바르게 예측할 수 없음(거리기반 알고리즘일 경우 특히 그렇다) -> 일정한 기준으로 맞춰주어야 한다

# 1. 표준점수(standard score)(z-score)

mean = np.mean(train_input, axis = 0)
std = np.std(train_input, axis = 0) # 특성별로 각 값을 계산해야 하므로 axis = 0 (행을 따라 각 열의 통계 값을 계산해줌)(이 경우에는 세로방향) / axis = 1 이면 가로방향
print(mean,std)

train_scaled = (train_input-mean) / std # numpy의 브로드캐스팅 기능을 이용한 계산 / 계산하는 배열끼리의 차원이 일치한다면 각 차원에 대응되도록 계산을 실행

plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(25,150,marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show(block=True) # train 데이터는 scaling을 했지만 (25, 150)은 하지 않았기에 큰 차이가 발생

new = ([25,150] - mean) / std
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0],new[1],marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show(block=True)

# scaling이 끝난 데이터를 이용한 fitting
kn.fit(train_scaled, train_target) # 훈련을 마친 이후 test set으로 평가할 경우 test set 또한 scaling을 해야 한다

test_scaled = (test_input - mean) / std
kn.score(test_scaled, test_target) # 1.0

print(kn.predict([new])) # 추가된 point에 대한 예측도 도미로 분류

distances, indexes = kn.kneighbors([new]) # new point 주변에 존재하는 점들에 대한 거리와 index
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0],new[1],marker = '^')
plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker = 'D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show(block=True) # 가장 가까운 점이 도미로 변경되었음

# 대부분의 머신러닝 알고리즘은 scale이 다르다면 잘 작동하지 않음