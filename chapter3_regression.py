### 예측(Prediction)
# classification <---> prediction

'''
지도학습 알고리즘은 크게 분류(classification)와 회귀(regression)로 나누어짐
k-nearest neighbors 을 이용한 회귀를 할 경우 분류와 비슷하지만 회귀에서는 주변에 가장 가까운 점들 값의 평균을 이용
'''

import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 준비
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# 데이터 파악을 위한 scatter plot

plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show(block=True) # 농어의 길이가 길어짐에 따라 무게도 늘어지는 모양

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(perch_length,perch_weight,random_state = 42)

# sklearn에 사용해야할 train set은 2차원 배열이어야 하므로 1차원 배열을 2차원으로 변경 --> reshape() 메소드를 이용

test_array = np.array([1,2,3,4])
print(test_array.shape) # (4,)

test_array = test_array.reshape(2,2) # numpy 배열은 배열 크기를 바꿀 수 있는 reshape() 메소드를 제공 / 바꾸려면 바꾸기 전의 원소 개수와 바꾼 후의 원소 개수가 같아야 함
print(test_array.shape)

train_input = train_input.reshape(-1,1) # 크기에 -1이 들어가면 나머지 원소 개수로 채우라는 의미
test_input = test_input.reshape(-1,1)
print(train_input.shape, test_input.shape) # 2차원 배열로의 변환

# 결정계수(R^2)
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()
knr.fit(train_input,  train_target)

print(knr.score(test_input, test_target)) # 0.992809406101064 / 분류의 경우는 정확하게 분류한 개수의 비율이지만 회귀에서는 결정계수로 평가(R^2)

# 좀 더 명확한 평가를 위해 다른 평가척도를 이용
from sklearn.metrics import mean_absolute_error

test_prediction = knr.predict(test_input)

mae = mean_absolute_error(test_target,test_prediction)
print(mae) # 19.157142857142862 -> 예측이 평균적으로 19g 다르다는 것을 알 수 있음

print(knr.score(train_input, train_target)) #0.9698823289099254 -> score 값으로 train set을 넣었을 때 보통은 train을 넣은 것이 test보다 값이 좋게 나옴 = 과소적합

# 만약 train에서 값이 좋았지만 test는 나쁘다면 과대적합 / 반대로 훈련 세트보다 테스트 세트의 점수가 더 높다면 과소적합 -> 모델이 단순하여 훈련세트에 적절히 훈련되지 않은 경우
# 과소적합을 해결하기 위해서는 모델을 좀 더 복잡하게 만들어야 한다 -> k-nearest에서 모델을 복잡하게 만들려면 k의 값을 줄이면 됨 (default는 5 이므로 3으로 낮춤)

knr.n_neighbors = 3

knr.fit(train_input, train_target)
print(knr.score(test_input, test_target)) #0.9746459963987609
print(knr.score(train_input, train_target)) # 0.9804899950518966 -> train을 넣은 경우가 더 크고 둘 사이의 차이가 크지 않으므로 과소적합도 아니고 과대적합 또한 아니다

### 선형 회귀
import numpy as np
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state = 42)
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors = 3)
knr.fit(train_input, train_target)

print(knr.predict([[50]])) # 1033g으로 무게 예측 --> 예측값과 실제값의 큰 차이가 발생 why?

# 무게 오차의 원인을 알기 위해 산점도 작성
import matplotlib.pyplot as plt

distances, indexes = knr.kneighbors([[50]])
plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker = 'D')
plt.scatter(50, 1033, marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show(block=True) # 산점도 결과 길이 50인 도미와 가장 가까운 점들이 45cm 근방이기에 k-nearest 알고리즘은 이 부근의 평균을 출력할 수 밖에 없음 --> 때문에 결과가 실제값과 달라짐
print(np.mean(train_target[indexes])) # k-nearest의 특성 상 새로운 값이 기존의 범위를 벗어나면 엉뚱한 값을 예측할 수 밖에 없다 --> 현재 데이터셋에서는 도미의 길이가 아무리 늘어나도 무게 예측값이 항상 1033g으로 나올 수 밖에 없음 --> 다른 알고리즘이 필요

# 선형 회귀
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(train_input, train_target)
print(lr.predict([[50]]))
print(lr.coef_, lr.intercept_) # lr 모형을 이용해 fitting한 이후 모델의 계수값과 절편값 보기(parameter)

plt.scatter(train_input, train_target)
plt.plot([15,50], [15*lr.coef_ + lr.intercept_, 50*lr.coef_ + lr.intercept_]) # 15~50까지의 1차방정식 그래프 --> 이 직선이 선형회귀에서 찾은 데이터셋의 최적 직선
plt.scatter(50, 1241.8, marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show(block=True)

print(lr.score(train_input, train_target)) #0.939846333997604
print(lr.score(test_input, test_target)) #0.8247503123313558

# 다항회귀

