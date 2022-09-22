### DecisionTree

# logistic regression을 이용한 와인 분류
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')
wine.head() # class 특성이 와인 종류(0 = 레드, 1 = 화이트)
wine.info() #df의 각 열 데이터 타입과 누락 데이터 확인 / 누락값이 있다면 그 값을 버리거나 평균값으로 대체, 뭐가 좋은지는 모름
wine.describe() # mean, std, min, max 값을 볼 수 있음 + median, IQR --> 각 특성 간 스케일이 다르다는 것을 알 수 있음

# Data Split
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size = 0.2, random_state = 42) # test_size 의 default = 0.25
print(train_input.shape, test_input.shape)

# Data Scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# LogisticRegression Fitting
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target)) # 0.7808350971714451
print(lr.score(test_scaled, test_target)) # 0.7776923076923077 --> 두 셋의 점수가 모두 낮은 것을 보아 과소적합이 의심됨

# result
print(lr.coef_, lr.intercept_) #[[ 0.51270274  1.6733911  -0.68767781]] [1.81777902]

"""
이러한 계수값을 결과로 뽑아도 다른 사람에게 그 결과를 설명하는 것이 직관적이지 못함 (숫자가 가지는 의미를 설명하기 어려움) --> 이는 다른 머신러닝 기법들에게도 적용되는 문제
"""

## DecisionTree --> 설명하기 쉬움
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target)) # 0.996921300750433
print(dt.score(test_scaled, test_target)) # 0.8592307692307692 --> 훈련 셋의 점수는 매우 높은 반면 테스트 셋 점수가 낮은 것으로 보아 과대적합이 의심됨

# plot_tree()
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize = (10,7))
plot_tree(dt, max_depth = 1, filled = True, feature_names = ['alcohol','sugar','pH'])
plt.show(block=True)  # 너무 많은 node가 출력되므로 depth를 조절

"""
max_depth = 최대 depth 설정
filled = 노드 색 지정
feature_names = 특성 이름 전달
Node에서 가장 많은 클래스가 예측 클래스가 됨
"""

## 불순도
"""
gini : DecisionTreeClassifier의 criterion 매개변수의 default / criterion은 노드에서 데이터를 분할할 때 그 기준을 정하는 것
gini = 1 - ('음성 클래스 비율' ^ 2 + '양성 클래스 비율' ^ 2) --> 0.5인 경우가 최악, 0이면 순수 노드
DecisionTree 는 부모 노드와 자식 노드의 불순도 차이(정보 이득)가 크도록 트리를 성장시킴
정보 이득 = 부모와 자식 노드 사이의 불순도 차이

criterion = 'entropy' 라면 엔트로피 불순도를 사용 가능 / 제곱이 아닌 log를 사용 / 결과 자체는 gini와 별 차이 없음

* 정리 : 불순도 기준을 gini 나 entropy 중 선택한 후 정보 이득이 최대가 되도록 노드를 분할, 노드를 순수하게 나눌수록 정보 이득이 커짐
"""

## 가지치기
"""
이전 결과에서는 제한 없이 계속 트리가 성장했기 때문에 테스트 점수가 낮게 나왔음(일반화가 잘 돼지 않았음) --> 가지치기 필요
가지치기의 가장 간단한 방법 = max_depth의 지정
"""

# max_depth = 3
dt = DecisionTreeClassifier(max_depth = 3, random_state = 42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target)) # 0.8454877814123533
print(dt.score(test_scaled, test_target)) # 0.8415384615384616

# 쉬운 이해를 위한 그래프 작성
plt.figure(figsize = (20,15))
plot_tree(dt, filled = True, feature_names = ['alcohol', 'sugar','pH'])
plt.show(block=True)

"""
특성값의 scale은 분석에 전혀 영향을 주지 않음 --> 전처리 필요 없음
"""
dt = DecisionTreeClassifier(max_depth = 3, random_state = 42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target)) # 0.8454877814123533
print(dt.score(test_input, test_target)) # 0.8415384615384616 --> scaling을 한 것과 정확하게 동일한 값

# 가장 유용한 특성을 나타내는 특성 중요도 계산
print(dt.feature_importances_) #[0.12345626 0.86862934 0.0079144 ] --> 두번쨰 특성이 가장 중요 (당도) / 특성 중요도를 특성 선택에 활용할 수 있음


### 교차 검증과 그리드 서치
"""
max_depth = 3이었지만 매개변수에 여러 값을 넣어서 계산하면 더 정확한 모델을 만들 수 있을 것 / 물론 DecisionTree는 max_depth 이외에 더 많은 매개변수가 있음
테스트 셋을 사용하지 않고 과대적합 여부를 판정하려면 validation set을 이용 (보통 20%정도) --> 60 / 20 / 20
훈련 셋을 통해 모델을 훈련한 후 검증 셋으로 모델 평가
"""

## Validation Set
# 데이터 준비
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')

# X / Y 분할
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# train / test 분할
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size = 0.2, random_state = 42) # test_size 의 default = 0.25

# train / validation 분할
sub_input, valid_input, sub_target, valid_target = train_test_split(train_input, train_target, test_size = 0.2, random_state = 42)
print(sub_input.shape, valid_input.shape)

# model fitting
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target)) # 0.9971133028626413
print(dt.score(valid_input, valid_target)) #0.864423076923077 --> 과대적합

## 교차검증 --> 보통 5-fold or 10-fold 교차검증을 주로 사용
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target) # cross_validate(model, train_input, train_target)
print(scores) # fit_time, score_time, test_score 키를 가진 딕셔너리 반환 --> 교차검증 최종 점수는 test_score의 값 평균

import numpy as np
print(np.mean(scores['test_score'])) # 0.855300214703487


"""
cross_validate() 를 진행할 떄 이미 섞여진 데이터를 넣었기에 splitter를 지정할 필요가 없지만 만약 섞이지 않았다면 필요)
sklearn의 splitter 는 교차검증에서 폴드를 어떻게 나눌지 결정 --> 기본적으로 회귀인 경우 KFold 분할기, 분류인 경우 StratifiedKFold 사용
"""
from sklearn.model_selection import StratifiedKFold
scores = cross_validate(dt, train_input, train_target, cv =StratifiedKFold()) # cv default = StratifiedKFold()
print(np.mean(scores['test_score'])) #0.855300214703487 --> 위의 교차검증과 동일

# 훈련 셋 섞기 + 10-fold
splitter = StratifiedKFold(n_splits = 10, shuffle=True, random_state = 42) # n_splits 매개변수로 k-fold의 k를 결정 / KFold 클래스도 동일하게 사용 가능
scores = cross_validate(dt,train_input, train_target, cv = splitter)
print(np.mean(scores['test_score'])) #0.8574181117533719

## HyperParameter 튜닝
"""
모델이 학습하는 파라미터 = 모델 파라미터
모델이 학습할 수 없어 사용자가 지정해줘야만 하는 파라미터 = 하이퍼파라미터 --> 메소드의 매개변수로 들어감
하이퍼 파라미터는 여러 개라면 파라미터끼리 상호작용이 있기에 모든 파라미터를 최적화할 필요가 있음 --> for로 모든 반복을 구할 수 있지만 기존 만들어진 메소드가 존재 = GridSearchCV()
"""
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease' : [0.0001,0.0002,0.0003,0.0004,0.0005]} # 노드를 분할하기 위한 불순도 감소 최소량
gs = GridSearchCV(DecisionTreeClassifier(random_state = 42), params, n_jobs = -1)
"""
min_impurity_decrease 값을 바꿔가며 5번 실행 / GridSearchCV의 매개변수 기본값 = 5 --> 총 25번 실행
n_jobs = 사용하는 cpu코어 숫자 (기본값은 1)
"""
gs.fit(train_input, train_target) # 최적 hyperparameter를 찾으면 전체 훈련셋으로 다시 모델 fitting --> gridsearchCV는 자동으로 수행해줌 (best_estimator_ 속성에 저장)
dt = gs.best_estimator_
print(dt.score(train_input, train_target)) # 0.9615162593804117
print(gs.best_params_) # {'min_impurity_decrease': 0.0001}
print(gs.cv_results_['mean_test_score']) # [0.86819297 0.86453617 0.86492226 0.86780891 0.86761605]

best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index]) # {'min_impurity_decrease': 0.0001}

"""
<순서>
1. 탐색 매개변수 지정
2. 훈련 세트에서 그리드 서치 -> 최상 점수가 나오는 매개변수 조합 찾기 -> 그리드 서치 객체에 저장
3. 그리드 서치는 최상 매개변수에서 전체 훈련 세트를 사용해 최종 모델 훈련 -> 그리드 서치 객체에 저장
"""

# min_impurity_decreas + max_depth + min_samples_split
params = {'min_impurity_decrease' : np.arange(0.0001, 0.001, 0.0001),
                'max_depth' : range(5,20,1),
                'min_samples_split' : range(2,100,10)} # 총 만들어지는 모델의 수 = 9 * 15 * 10 = 1350 * 5-fold = 6750

gs = GridSearchCV(DecisionTreeClassifier(random_state = 42), params, n_jobs = -1)
gs.fit(train_input, train_target)

print(gs.best_params_) # {'max_depth': 14, 'min_impurity_decrease': 0.0004, 'min_samples_split': 12}
print(np.max(gs.cv_results_['mean_test_score'])) # 0.8683865773302731

## 랜덤 서치
"""
매개변수 값이 수치일 때 값 범위나 간격을 정하기 어려울 수 있음 + 변수 조건이 많아 시간 소요 ==> 랜덤 서치
랜덤 서치는 매개변수 값의 목록이 아닌 확률 분포를 전달
"""

from scipy.stats import uniform, randint
rgen = randint(0,10) # 0~10 사이의 랜덤한 숫자
rgen.rvs(10) # 0~10개 숫자 중에서 10개를 뽑음
np.unique(rgen.rvs(1000), return_counts = True) # 각 숫자 개수도 함께 출력

# 균등분포에 적용
ugen = uniform(0,1)
ugen.rvs(10)

params = {'min_impurity_decrease' : uniform(0.0001,0.001),
                'max_depth' : randint(20,50),
                'min_samples_split' : randint(2,25)
                'min_samples_leaf' : randint(1,25)}  #   'min_samples_leaf' = 리프 노드가 되기 위한 최소 샘플 개수로 어떤 노드가 분할해서 만들어질 자식 노드 샘플 수가 이보다 작으면 분할 안함

from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state = 42), params, n_iter = 100, n_jobs = -1, random_state = 42) # 설정한 param 범위에서 100번의 반복을 실행
gs.fit(train_input, train_target)
print(gs.best_params_) # {'min_samples_split': 12, 'min_impurity_decrease': 0.0005, 'max_depth': 11}
print(np.max(gs.cv_results_['mean_test_score'])) # 0.8681935292811135

dt = gs.best_estimator_
print(dt.score(test_input, test_target)) # 0.8615384615384616

# 수동으로 하이퍼 파라미터를 바꾸는 것보다 gridsearch OR RandomizedSearchCV 를 사용하는 것이 좋을 것