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

# 좀 더 명확한 평가를 위해 다른 평가 척도를 이용
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

## 데이터 분할 (train + test)
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state = 42)
train_input = train_input.reshape(-1,1) # input 데이터 2차원 리스트 변환
test_input = test_input.reshape(-1,1)

## KNeighborsRegressor fitting
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors = 3)
knr.fit(train_input, train_target)

# predict
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

# 다항 회귀 : 선형회귀로 모델을 만들었지만 그래프를 보면 약간 구부러진 곡선의 형태를 지니고 있기에 최적의 곡선을 찾아야 한다 -> 제곱항의 추가

train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))
print(train_poly.shape, test_poly.shape)

lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.predict([[50**2, 50]]))
print(lr.coef_, lr.intercept_)

# 곡선
point = np.arange(15,50)
plt.scatter(train_input, train_target)
plt.plot(point, 1.01 * point ** 2 - 21.6 * point + 116.05)
plt.scatter(50, 1574, marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show(block=True)

print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target)) # 점수가 각각 높아졌으나 여전히 테스트에 대한 점수가 더 높음 --> 과소적합이 존재 -> 더 복잡한 모델이 필요

# k-nearest 의 가장 큰 단점은 훈련 세트 범위 밖의 샘플을 예측할 수 없다는 것 -> 해결을 위해 선형회귀 사용 -> 선형 회귀로도 최적 곡선이 나오지않아 다항 회귀 사용

### 특성 공학과 규제
# 어느정도 무게 예측이 가능해졌지만 여전히 훈련 세트보다 테스트 세트의 점수가 더 높은 편 ---> 과소적합 -> 하나가 아닌 여러 개의 특성을 이용 / 여러 개의 특성을 이용한 회귀 = 다중 회귀
# 항이 두개라면 직선이 아닌 평면의 형태로 방정식을 그려낼 수 있음
# 특성 간의 결합을 새로운 특성으로 것 = 특성 공학(feature engineering)

# 데이터 준비 -> csv 파일을 pandas를 이용해 import한 후 numpy를 이용해 numpy 배열로 변환
import pandas as pd
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full) # 입력데이터(input)

import numpy as np
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0]) # target 데이터

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state = 42) # 데이터 분할

# sklearn의 변환기 --> 특성을 만들거나 전처리하기 위한 다양한 클래스 / fit() , transform()
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures()
poly.fit([[2,3]]) # fit 에서 새롭게 만들 수 있는 특성 조합을 찾음 / transform 에서는 실제 데이터로 변환 --> 변환기에서는 target데이터가 필요없음
print(poly.transform([[2,3]])) # fit과 transform을 한번에 할 수 있는 fit_transform() 메소드도 있음

poly = PolynomialFeatures(include_bias = False) # 특성 목록에 항상 1이 포함되어 있기에 이것을 제거하기 위한 옵션 (include_bias = False) / 굳이 이것을 지정하지 않아도 모델에서는 제외됨
poly.fit([[2,3]])
print(poly.transform([[2,3]]))  #[[2. 3. 4. 6. 9.]]

poly = PolynomialFeatures(include_bias = False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape) #(42, 9)

poly.get_feature_names_out() # array(['x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2', 'x1^2', 'x1 x2', 'x2^2'], dtype=object)

test_poly = poly.transform(test_input) # 훈련 셋을 변환할 떄와 같은 변환기를 사용하지 않아도 되지만 대부분 같은 변환을 하는 것이 좋으므로 습관을 가지자

# 다중 회귀 모델 훈련하기 -> 다중회귀모델의 훈련은 선형회귀모델 훈련과 같으며 다만 특성이 여러개일 뿐이다
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target)) #0.9903183436982125
print(lr.score(test_poly, test_target)) #0.9714559911594155 --- > 테스트 셋의 점수가 더 높고 각각의 점수가 높은 편 / 과소적합 해결

# 특성을 여기서 더 추가? --> 제곱항에 이어 세제곱, 네제곱까지 넣을 수 있음
# PolynomialFeatures 의 degree 매개변수 사용 --> 5제곱까지 가능
poly = PolynomialFeatures(degree = 5, include_bias = False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape) #(42, 55) --> 특성 개수가 55

lr.fit(train_poly, train_target)
print(lr.score(train_poly,train_target)) #0.9999999999938143
print(lr.score(test_poly, test_target)) #-144.40744532797535 ---> 특성의 개수를 엄청나게 늘리면 테스트셋에 대한 평가척도는 매우 잘 나오지만 테스트 셋에서는 형편없는 결과 = 과대적합

# 규제 -> 과대적합 해결을 위한 방법
# 규제를 하기 전에 미리 scaling 해야함 --> 표준점수 or  StandardScaler
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly) # 반드시 훈련세트로 학습한 변환기로 테스트셋을 변환해야함

# 선형 회귀에 규제를 추가한 모델 = Ridge(릿지) , lasso(라쏘)
# Ridge = 계수를 제곱한 값을 기준으로 적용 / lasso = 계수의 절댓값을 이용 --> 일반적으로 릿지를 더 선호 / 두 알고리즘 모두 계수를 감소시키지만 라쏘는 아예 0으로 만들 수 있음

## Lidge 회귀
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled,train_target)) #0.9896101671037343
print(ridge.score(test_scaled,test_target)) #0.979069397761539 --> 규제가 없을 떄의 다중회귀에서 정상적으로 돌아옴

# 릿지와 라쏘를 사용할 떄 규제의 정도를 조절 가능 --> 매개변수 alpha ( alpha가 커지면 규제강도가 상승해 과소적합, alpha가 작아지면 선형회귀 모델과 유사해지므로 과대적합)
# 이러한 매개변수는 분석자가 직접 지정해야함 = 하이퍼파라미터(hyperparameter)
# 적절한 alpha값을 찾는 방법은 alpha에 대한 R^2를 그려보는 것 --> train과 test의 점수가 가장 가까운 지점이 최적 alpha

import matplotlib.pyplot as plt
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
       # Ridge model
       ridge = Ridge(alpha = alpha)
       # fitting
       ridge.fit(train_scaled, train_target)
       # score
       train_score.append(ridge.score(train_scaled, train_target))
       test_score.append(ridge.score(test_scaled, test_target))

# alpha값을 0.001부터 10배씩 늘렸기에 그래프 왼쪽이 너무 촘촘 --> 지수변환으로 해결
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list),test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show(block=True) # 전형적인 과대적합에서 과소적합으로 가는 그래프 --> 두 차이가 가장 적은 부분은 alpha = -1 (log변환했으므로 0.1)

# alpha=0.1 fitting
ridge = Ridge(alpha = 0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target)) #0.9903815817570368
print(ridge.score(test_scaled, test_target)) #0.9827976465386954

## Lasso 회귀
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target)) #0.989789897208096
print(lasso.score(test_scaled, test_target)) #0.9800593698421884 --> lasso 회귀 모델 또한 어느정도 과대적합을 억제

# 최적 alpha찾기
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
       # Lasso model
       lasso = Lasso(alpha=alpha, max_iter=50000000)
       # fitting
       lasso.fit(train_scaled, train_target)
       # score
       train_score.append(lasso.score(train_scaled, train_target))
       test_score.append(lasso.score(test_scaled, test_target))

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list),test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show(block=True) # alpha = 1 (log이므로 10에서 최적)

lasso = Lasso(alpha = 10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target)

# lasso 모델은 계수 0이 존재
print(np.sum(lasso.coef_ == 0))