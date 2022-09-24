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

# 수동으로 하이퍼 파라미터를 바꾸는 것보다 gridsearchCV OR RandomizedSearchCV 를 사용하는 것이 좋을 것

## 트리의 앙상블
"""
*정형 데이터와 비정형 데이터
- 정형 데이터 = 어떠한 구조로 되어있는 데이터 / CSV, DB, EXCEL 에 저장하기 쉬움
- 비정형 데이터 = DB나 EXCEL 
- 보통의 머신러닝 알고리즘은 정형 데이터에 잘 맞음 --> 그 중 가장 뛰어난 성과를 내는 알고리즘 = 앙상블 학습
- 앙상블은 결정 트리 기반의 머신러닝 모델
- 비정형 데이터에 대해서는 신경망 알고리즘을 사용 / 비정형 데이터는 규칙성을 찾기 어려워 전통적 머신러닝으로는 모델 만들기 어려움
"""

## RandomForest (앙상블 학습의 대표)
"""
RandomForest 는 각 트리를 훈련하기 위한 데이터를 랜덤하게 생성 -> 훈련 데이터 중 랜덤하게 샘플을 선택해 훈련데이터를 만듦 (중복 추출도 가능) / 복원추출 ==> 부트스트랩 샘플
BootStrapSample 은 보통 훈련세트와 크기가 같음
각 노드 분할 시 전체 특성 중에서 일부 특성을 무작위로 선택한 후 이 중에서 최선의 분할을 찾음 --> RandomForestClassifier 는 기본적으로 전체 특성 수의 제곱근만큼 특성 선택
RandomForestRegressor 는 전체 특성을 전부 사용
Sklearn의 RandomForest는 기본적으로 100개의 결정 트리를 훈련 -> 분류라면 각 트리의 클래스 별 확률을 평균해 가장 높은 확률을 가진 클래스를 예측으로 하고 회귀라면 단순히 각 트리의 예측을 평균
RandomForest는 랜덤한 샘플과 특성을 사용하기에 train set 과대적합을 막을 수 있음
"""

# Data Import + Data Split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar','pH']].to_numpy()
target = wine['class'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(data, target, random_state = 42, test_size = 0.2)

# CrossValidate를 이용한 교차 검증
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs = -1, random_state = 42)
scores = cross_validate(rf, train_input, train_target, return_train_score = True, n_jobs = -1)
print(np.mean(scores['train_score']), np.mean(scores['test_score'])) # 0.9973541965122431 0.8905151032797809 --> 과대적합 (HyperParameter에 대한 조정이 필요하지만 데이터 특성 수가 별로 없어 효과 미미)
"""
기본적으로 많은 계산량을 필요로 하므로 n_jobs = -1 / return_train_score = True 라면 검증 점수뿐만 아니라 훈련 세트에 대한 점수도 같이 반환(과대적합 파악에 용이)
RandomForest는 DecisionTree의 앙상블이기에 DecisionTree가 필요로 하는 매개변수를 모두 제공 + 특성 중요도를 계산 / RandomForest의 특성 중요도는 각 DecisionTree의 특성 중요도를 취합한 것
"""
rf.fit(train_input, train_target)
print(rf.feature_importances_)
# [0.23167441 0.50039841 0.26792718] --> DeicisonTree의 결과보다 두번째 특성의 중요도가 감소하고 나머지 특성들의 중요도가 늘었음 = RandomForest가 특성 일부를 랜덤하게 선택해 훈련 --> 과대적합 감소 + 일반화
"""
RandomForestClassifier 는 자체적으로 모델 평가 점수를 계산 
BootStrap Sample을 만들고 남은 샘플을 OOB라고 하는 데, 이를 이용해서 만들어진 결정 트리를 평가 (Validation Set의 역할을 수행)
모델 평가 점수를 얻으려면 oob_score = True로 지정해야 함 (Default = False)
"""

rf = RandomForestClassifier(oob_score = True, n_jobs = -1, random_state = 42)
rf.fit(train_input, train_target)
print(rf.oob_score_) # 0.8934000384837406 / OOB 점수를 이용하면 교차 검증을 대신할 수 있어 결과적으로 많은 훈련 세트 샘플을 사용할 수 있음

"""
< oob score 에 관하여>
oob_score는 강력한 검증 기술 중 하나로 특히 RandomForest 알고리즘에서 최소 분산 결과를 도출하는 데 좋다.
cross-validation 기법을 사용하면 매 검증마다 데이터 누출이 일어나기에 분산이 늘어날 수 밖에 없다

- 장점 
1. 데이터 누출 없음 : 데이터가 oob sample에서 검증되었으므로 모델 훈련하는 동안 데이터가 사용되지 않아 데이터 누출이 없음
2. Less Variance : 데이터가 과도하게 적합되지 않아 분산 최소화
3. 더 나은 예측 모델 : 분산이 낮으므로 다른 검증 기술을 사용하는 모델보다 더 나은 예측 모델을 만들 수 있음
4. 적은 계산 : 학습되는 데이터를 테스트할 수 있으므로 계산량 감소

- 단점
1. 시간 소요 많음 : 다른 검증 기법에 비해 시간이 많이 소요됨
2. 대용량 데이터 세트에는 적합하지 않음 : 대용량일 경우 시간이 매우 오래 걸릴 수 있음
3. 중소 규모 데이터 세트에 적합 : 이 경우에는 oob_score를 사용할만하다
"""

## 엑스트라 트리 (Extra Tree)
"""
RandomForest와 기본적으로 비슷하게 동작 -> 100개의 DecisionTree를 훈련 / DecisionTree가 제공하는 대부분의 매개변수 지원
전체 특성 중 일부 특성을 선택해서 노드 분할에 사용
- RandomForest와 ExtraTree의 차이점은 BootStrap Sample을 사용하지 않는다는 것 + 노드 분할 시 가장 좋은 분할을 찾는 것이 아닌 무작위 분할 (이전 DecisionTreeClassifier에서 splitter가 Random 인 경우)
- splitter를 random으로 한다면 성능은 낮아지지만 많은 트리를 앙상블하기에 과대적합을 막고 검증세트의 점수를 높이는 효과 있음
"""
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_jobs = -1, random_state = 42)
scores = cross_validate(et, train_input, train_target, n_jobs = -1, return_train_score = True)
print(np.mean(scores['train_score']), np.mean(scores['test_score'])) # 0.9974503966084433 0.8887848893166506
# RandomForestClassifier 와 비슷한 결과지만 RandomForest보다 더 많은 결정 트리 훈련이 필요하지만 계산 속도는 더 빠름 (DecisionTree는 최적 분할을 찾는 데 시간을 많이 소모하기 때문)
# ExtraTree 또한 특성 중요도를 반환

et.fit(train_input, train_target)
print(et.feature_importances_) # [0.20183568 0.52242907 0.27573525]

## Gradient Boosting
"""
얕은 깊이의 결정 트리를 사용해 이전 트리의 오차를 보완하는 방식으로 앙상블
sklearn의 GradientBoostingClassifier 는 기본적으로 Depth 3의 DecisionTree 100개를 사용 --> 과대적합에 강하고 일반적으로 높은 일반화 성능을 기대할 수 있음
GradientDescent를 이용해 트리를 앙상블에 추가 --> Classifier에서는 로지스틱 손실 함수를 사용하고, Regression에서는 평균 제곱 오차함수(MSE)를 사용
- GradientBoosting은 결정 트리를 계속 추가하면서 가장 낮은 곳을 찾아 이동 (모델 가중치와 절편을 조금씩 바꾸는 것) --> 조금씩 바꾸기 위해서 Depth가 작은 DecisionTree를 사용 + 학습률 매개변수도 사용
"""

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state = 42)
scores = cross_validate(gb, train_input, train_target, return_train_score = True, n_jobs = -1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))