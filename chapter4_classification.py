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
import matplotlib.pyplot as plt
z  = np.arange(-5,5,0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show(block=True)

"""
간단한 이진분류에서는 확률이 0.5 이상이면 양성, 작으면 음성으로 판단(정확하게 0.5이면 음성으로 판단, 라이브러리마다 다름)
"""
# boolean indexing
char_arr = np.array(['A',',B','C','D','E'])
print(char_arr[[True, False, True, False, False]]) # True에 해당하는 a, c만 출력

# boolean indexing 을 이용해 도미와 빙어에 관한 행만 검색
bream_smelt_indexes = (train_target == 'Bream') | (train_target =='Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

# fitting
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:5])) #['Bream' 'Smelt' 'Bream' 'Bream' 'Bream'] / 두 번째를 제외하고는 모두 도미로 예측
print(lr.predict_proba(train_bream_smelt[:5]))
print(lr.classes_)
"""
sklearn은 타깃값을 알파벳순으로 자동정렬하여 사용하기 때문에 predict_proba 의 결과는 [도미, 빙어] 의 순서이다
"""

print(lr.coef_, lr.intercept_) # 5개 특성에 대한 계수 + 절편값

# z값 출력
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions) # 이 값들을 시그모이드에 통과시키면 확률이 계산됨 --> scipy 라이브러리의 expit() 함수

# z값 -> 확률 변환
from scipy.special import expit
print(expit(decisions)) # decision_function은 양성 클래스에 대한 z값을 반환(이진분류에서 값을 1로 분류한 것)(이 경우에서는 빙어)

## 로지스틱 회귀로 다중 분류 수행
"""
- 이진 분류와 다중 분류는 크게 다르지 않음
- 로지스틱회귀는 반복적인 알고리즘을 사용하고 max_iter 매개변수에서 반복횟수 지정(default = 100)
- 로지스틱회귀는 기본적으로 Ridge와 같이 계수의 제곱 규제(L2 규제) --> 규제 제어 매개변수는 'C'이지만 ridge의 매개변수인 alpha와는 다르게 작아지면 규제가 커짐 (C의 default = 1)
"""

lr = LogisticRegression(C=20, max_iter = 1000) # 규제 완화를 위해 C = 20, 반복횟수 충당을 위해 max_iter = 1000
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled,train_target)) #0.9327731092436975
print(lr.score(test_scaled,test_target)) #0.925 --> 테스트셋에 대한 점수가 더 높고 과대적합이나 과소적합으로 치우치지 않음

print(lr.predict(test_scaled[:5])) # 분류 예측

proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals = 3)) # 반올림
print(lr.classes_) # 각 열이 어떤 물고기를 뜻하는지 파악

# 다중분류의 선형 방정식
print(lr.coef_, lr.intercept_) # 행이 7개 출력됨

"""
- 다중 분류는 클래스마다 z값을 하나씩 계산하고 가장 높은 z값을 출력하는 클래스가 예측 클래스가 됨 
- 이진분류에서는 시그모이드 함수를 이용해서 z값을 0~1로 변환했지만 다중분류에서는 softmax function을 이용
- softmax 는 여러 개의 선형 방정식 출력값을 0~1 사이로 압축하고 전체 합이 1이 되도록 만듦
"""

# z 값 계산
decision = lr.decision_function(test_scaled[:5]) # z값 계산
print(np.round(decision, decimals = 2))

# softmax
from scipy.special import softmax
proba = softmax(decision, axis = 1) # axis = 1 로 지정하여 각 행에 대해 계산
print(np.round(proba, decimals = 3))

## 확률적 경사 하강법
"""
- 훈련 데이터가 한번에 주어지는 것이 아닌 점진적으로 조금씩 전달 --> 데이터가 추가될 때마다 훈련을 반복하면 해결이지만 이는 시간이 지날수록 많은 시간을 소비
- 다른 방법은 새로운 데이터를 추가할 때 이전 데이터를 버려 훈련데이터 크기를 일정하게 유지 --> 예측력이 떨어짐
- 앞서 훈련한 모델을 버리지 않고 새로운 데이터에 대해서만 조금씩 더 훈련 --> 점진적 학습 (대표적으로 확률적 경사 하강법(stochastic Gradient Descent))
- 확률적 경사 하강법 : 훈련 셋에서 랜덤하게 하나의 샘플을 선택해 조금씩 하강하고 다 사용할 때까지 반복 -> 사용 후에도 하강하지 못했으면? 다시 처음부터 시작
- 하강법에서 훈련셋을 모두 한번 사용하는 것 = 에포크(epoch) --> 일반적으로 수십, 수백번 진행
- 샘플을 1개가 아닌 몇 개의 샘플을 무작위 선택해 하강하는 방법 = minibatch gradient descent
- 한번 이동하는 데 모든 샘플을 사용 = batch gradient descent --> 가장 안정적인 방법일 수 있지만 자원 소비가 큼
- gradient descent가 반드시 사용되는 곳 = 신경망 알고리즘
- loss function(cost function) = 알고리즘이 얼마나 쓸만한지 측정하는 기준 --> 작을 수록 좋음 / gradient descent 로 손실함수에서 가장 작은 값을 가지는 point를 찾아야 함
- 분류에서의 손실을 정답을 맞추지 못하는 것
- 손실함수는 gradient descent를 활용하기 위해서 연속적이어야 함(미분가능)
- 로지스틱 손실 함수 (이진 크로스엔트로피 손실 함수)
"""

# SGDClassifier --> 확률적 경사하강법을 실행 (미니배치나 배치는 사용 불가능)
# data import
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')

# target / input
fish_target = fish['Species'].to_numpy()
fish_input = fish[['Weight', 'Length', 'Diagonal','Height','Width']].to_numpy()

# train / test split
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state = 42)

# 전처리
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier # 매개변수 loss = 손실함수의 종류 정의 / 매개변수 max_iter = 수행할 에포크 횟수
sc = SGDClassifier(loss = 'log', max_iter = 1000, random_state = 42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target)) #0.8403361344537815
print(sc.score(test_scaled, test_target)) #0.8

# 모델을 이어서 훈련 -> partial_fit()
sc.partial_fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target)) #0.907563025210084
print(sc.score(test_scaled, test_target)) #0.925

# 과대적합 / 과소적합 --> 에포크 횟수에 따라 과대적합, 과소적합이 나누어짐 / 때문에 테스트 셋 점수가 감소하는 지점에서 훈련을 종료 --> 조기 종료
import numpy as np
sc = SGDClassifier(loss = 'log', random_state = 42)
train_score = []
test_score = []
classes = np.unique(train_target)

for _ in range(0, 300) :
    sc.partial_fit(train_scaled, train_target, classes = classes) # partial_fit() 만 사용하려면 전체 레이블을 전달해야함 (classes = 전체 레이블 )
    train_score.append(sc.score(train_scaled,train_target))
    test_score.append(sc.score(test_scaled, test_target))

import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('score')
plt.show(block=True) # 초기에는 에포크 횟수가 적어 과소적합이다가 100번째 에포크 이후로 과대적합(차이가 커짐)

# 최적 에포크를 100으로 설정
sc = SGDClassifier(max_iter = 100, loss = 'log', tol = None, random_state = 42) # 본래 SGDClassifier는 일정 에포크동안 성능이 향상되지 않으면 훈련을 멈추지만 tol 매개변수로 이를 정지시킬 수 있음 (tol =None)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target)) #0.957983193277311
print(sc.score(test_scaled, test_target)) #0.925

"""
- SGDClassifier의 loss 매개변수 default = 'hinge' : SVM 알고리즘을 위한 손실함수
"""