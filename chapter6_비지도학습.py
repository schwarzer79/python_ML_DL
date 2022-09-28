### 비지도학습 (unsupervised learning)
"""
지도학습은 타깃이 주어졌을 때 사용하는 머신러닝 알고리즘이었음
비지도학습은 타깃이 없을 때 사용하는 머신러닝 알고리즘으로 사람이 가르쳐 주지 않아도 데이터에 있는 것을 학습
"""
## 군집 알고리즘

# 과일 사진 데이터 준비
import wget
url = 'https://bit.ly/fruits_300_data'
file = wget.download(url)

# 다운받은 파일에서 데이터 로드
import numpy as np
import matplotlib.pyplot as plt
fruits = np.load(file)
print(fruits.shape) #(300, 100, 100) / (샘플 개수, 이미지 높이, 이미지 너비) --> 배열 크기가 100x100 / 각 픽셀이 넘파이 배열의 원소 하나에 대응
print(fruits[0,0,:]) # 첫번째 이미지의 첫번째 행 출력

# matplotlib의 imshow() 함수를 이용한 이미지 그리기
plt.imshow(fruits[0], cmap = 'gray')
plt.show(block = True) # 배열 값이 0에 가까울수록 검고, 높으면 밝게 표시
"""
보통 흑백 이미지는 바탕이 밝고 사물이 어둡지만 여기서는 numpy 배열로 변환할 때 반전시킨 것
컴퓨터는 255에 가까운 바탕에 집중 (연산에 사용되기 쉽기 떄문)
cmap 매개변수 값을 'gray_r'로 지정하면 보기 좋게 출력 가능
"""
plt.imshow(fruits[0], cmap = 'gray_r')
plt.show(block=True) # 밝은 부분이 0, 짙은 부분이 255

fig, axs = plt.subplots(1,2) # subplots()를 이용해 여러 개의 그래프를 배열처럼 쌓을 수 있게 해줌 / subplots(x,y) 에서 매개변수 x,y는 그래프를 쌓을 행과 열 / axs 에 2개의 서브 그래프를 담고 있는 배열
axs[0].imshow(fruits[100], cmap = 'gray_r')
axs[1].imshow(fruits[200], cmap = 'gray_r')
plt.show(block = True)

# 픽셀 값 분석하기 -> 계산이 용이하도록 2차원 배열을 1차원 배열로 변환
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100) # 각 배열 크기는 (100, 10000)

# mean()을 사용해 샘플의 픽셀 평균값을 계산 / axis = 0라면 행, axis = 1이라면 열을 따라 계산
print(apple.mean(axis = 1))
plt.hist(np.mean(apple, axis = 1), alpha = 0.8)
plt.hist(np.mean(pineapple, axis = 1), alpha = 0.8)
plt.hist(np.mean(banana, axis = 1), alpha = 0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show(block=True) # 바나나 평균값은 40 아래 / 사과 + 파인애플은 90 ~ 100 -> 평균만으로도 바나나는 구분할 수 있음 / 그럼 사과와 파인애플은?

# 샘플 평균값이 아닌 픽셀별 평균값 비교
fig, axs = plt.subplots(1,3,figsize = (20,5))
axs[0].bar(range(10000), np.mean(apple, axis = 0))
axs[1].bar(range(10000), np.mean(pineapple, axis = 0))
axs[2].bar(range(10000), np.mean(banana, axis = 0))
plt.show(block = True)
"""
사과 : 사진 아래쪽으로 갈수록 값이 높아짐
파인애플 : 비교적 고르고 높음
바나나 : 중앙 픽셀값이 높음
"""

apple_mean = np.mean(apple, axis = 0).reshape(100,100)
pineapple_mean = np.mean(pineapple, axis = 0).reshape(100,100)
banana_mean = np.mean(banana, axis = 0).reshape(100,100)
fig, axs = plt.subplots(1,3, figsize = (20,5))
axs[0].imshow(apple_mean, cmap = 'gray_r')
axs[1].imshow(pineapple_mean, cmap = 'gray_r')
axs[2].imshow(banana_mean, cmap = 'gray_r')
plt.show(block= True)

# 평균값과 가까운 사진 고르기 --> fruits 배열의 모든 샘플에서 apple_mean을 뺸 절댓값의 평균

abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis = (1,2)) # axis = (1,2) --> (d, r, c) = (0,1,2) 에서 1,2에 대해 지정된 함수를 실행
print(abs_mean.shape) # (300,)

apple_index = np.argsort(abs_mean)[:100] # 차이가 가장 작은 샘플 100개 추출
fig, axs = plt.subplots(10,10,figsize = (10,10))
for i in range(10) :
    for j in range(10) :
        axs[i,j].imshow(fruits[apple_index[i*10 + j]], cmap = 'gray_r')
        axs[i,j].axis('off')
plt.show(block=True)
""" 
~ 코드 해석
subplots 함수로 10*10, 총 100개의 서브 그래프를 만들고 그래프가 많기에 전체 그래프 크기를 figsize 로 크게 지정
2중 for 반복문으로 각 샘플이 그래프의 어느 i * j 지점에 들어갈지 지정
깔끔한 이미지 출력만을 위해서 axis('off') 로 좌표축을 그리지 않음
--> 이렇게 비슷한 샘플끼리 그룹으로 모으는 작업을 군집(clustering) 이라고 한다 / 여기서 만들어진 그룹을 클러스터(cluster)라고 함
하지만 예시에서는 타깃을 알고 있었기에 평균값을 이용한 clustering을 할 수 있었지만 실전에서는 그렇지 않음 --> K-means 알고리즘을 이용해 문제 해결
"""

### K-Means
"""
이전 챕터에서는 타깃값이 주어졌기에 해당 타깃에 대한 평균값을 구해서 clustering을 할 수 있었음 --> 타깃이 없다면?
K-Means 알고리즘을 이용해 평균값을 찾을 수 있음 / 평균값은 클러스터의 중심에 위치하기에 cluster center OR centroid 라고 부름

K-Means 알고리즘 소개
1. 무작위로 k개의 클러스터 중심을 결정
2. 각 샘플에서 가장 가까운 클러스터 중심을 찾아 해당 클러스터의 샘플로 지정
3. 클러스터에 속한 샘플의 평균값으로 클러스터 중심 변경
4. 클러스터 중심에 변화가 없을 때까지 2번으로 돌아가 반복
"""

## K-Means 클래스
import numpy as np
fruits = np.load(file)
fruits_2d = fruits.reshape(-1, 100*100) # k-means 모델 훈련을 위해 (샘플개수, 너비, 높이) 의 3차원 배열을 (샘플 개수, 너비 * 높이) 의 2차원 배열로 변경

from sklearn.cluster import KMeans # 매개변수 n_clusters = 클러스터 개수 지정
km = KMeans(n_clusters = 3, random_state = 42)
km.fit(fruits_2d) # 군집 결과는 KMeans 객체의 labels_ 속성에 저장 / labels_ 배열 길이는 샘플 개수와 동일 / n_clusters = 3이기에 labels 배열 값은 0,1,2 중 하나
print(km.labels_)
print(np.unique(km.labels_, return_counts = True)) # (array([0, 1, 2]), array([111,  98,  91], dtype=int64))

# 각 클러스터가 어떤 이미지를 나타냈는지 그림으로 출력 --> draw_fruits() 함수 만들기
import matplotlib.pyplot as plt
def draw_fruits(arr, ratio = 1) :
    n = len(arr) # 샘플 개수
    rows = int(np.ceil(n/10)) # 한줄에 10개씩 이미지 그리기
    cols = n if rows < 2 else 10 # 행이 1개이면 열 개수는 샘플 개수, 아니면 10개
    fig, axs = plt.subplots(rows, cols, figsize = (cols * ratio, rows * ratio), squeeze = False)
    for i in range(rows) :
        for j in range(cols) :
            if i *10 + j < n :
                axs[i,j].imshow(arr[i*10 + j], cmap = 'gray_r')
            axs[i,j].axis('off')
    plt.show(block=True)


"""
draw_fruits()
- (샘플 개수, 너비, 높이) 의 3차원 배열을 입력받아 가로로 10개씩 이미지 출력 / 샘플 개수에 따라 행과 열의 개수를 계산하고 figsize 지정
- 2중 for 반복문을 이용해 이미지 출력
- boolean indexing 을 이용해서 0, 1, 2에 해당하는 이미지들을 출력
"""
draw_fruits(fruits[km.labels_ == 0])
draw_fruits(fruits[km.labels_ == 1])
draw_fruits(fruits[km.labels_ == 2]) # 완벽하게 구분은 하지 못했지만 유사한 샘플들끼리 잘 모았음

## 클러스터 중심 --> KMeans 클래스에서 찾은 최종 centroid는 clusters_ centers_ 속성에 있음, 단 이것은 fruits_2d(2차원배열)의 중심이므로 이미지 출력을 위해 100 * 100으로 바꿔야 함
draw_fruits(km.cluster_centers_.reshape(-1,100,100), ratio = 3) # 이전 절에서 각 과일의 평균값을 출력했던 것과 아주 유사하게 나왔음

# KMeans 클래스는 훈련 데이터 샘플에서 centroid까지 거리로 변환해 주는 transform() 메소드가 있음
print(km.transform(fruits_2d[100:101]))
"""
fruits_2d[100] 으로 전달하면 (10000,1) 이 아닌 (10000,)로 전달되기에 오류 발생
각 cluster의 centroid 거리를 나타낸 것
"""
print(km.predict(fruits_2d[100:101])) # predict()로 해당 샘플이 어느 cluster에 속할 것인지 예측
draw_fruits(fruits[100:101])
print(km.n_iter_) # 알고리즘이 반복한 횟수는 km.n_iter_ 에 저장됨

# 이번 실습에서도 약간의 편법이 있었음 --> 실전에서는 샘플들이 몇 개의 타깃으로 구성되어있는지 알 수 없음 --> n_clusters 개수를 정확하게 지정할 수 없음

## 최적 k 찾기
"""
K-Means 알고리즘의 단점 중 하나는 클러스터 개수를 사전에 지정해야 한다는 것 --> 적절한 k값을 찾기 위한 방법? / 완벽한 방법은 없음
<k값 찾기>
1. elbow : k-means 는 centroid와 클러스터에 속한 샘플 사이의 거리를 잴 수 있는 데 이 거리의 제곱합을 이너셔(inertia)라고 한다
inertia를 통해 클러스터에 속한 샘플이 얼마나 가깝게 모여있는지 알 수 있음 / 일반적으로 클러스터의 개수가 늘어나면 각 클러스터의 크기는 줄어들어 inertia도 줄어듬
elbow는 클러스터 개수를 늘려갈 떄 inertia의 변화를 관찰해 최적 k를 찾는 방법 / k를 늘리다보면 inertia가 감소하는 속도가 꺾이는 지점이 있음, 이 지점 이후로는 클러스터 개수를 늘려도 개선이 잘 안됨
KMeans 클래스는 자동으로 inertia_ 속성으로 제공
"""
inertia = []
for k in range(2,7) :
    km = KMeans(n_clusters = k, random_state = 42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)
plt.plot(range(2,7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show(block =True) # k=3에서 그래프의 기울기가 조금 바뀐 것을 알 수 있음, 하지만 명확하지는 않음

### 주성분 분석
"""
<차원과 차원 축소>
데이터가 가진 속성 = 특성 / 사진의 경우 10000개의 픽셀이 있었기에 10000개의 특성을 가지고 있는 것 , 머신러닝에서는 이러한 특성을 차원(dimension)이라고도 함

2차원 배열과 1차원 배열의 차원
- 다차원 배열에서의 차원은 축 개수, 1차원 배열에서는 원소의 개수를 의미

차원 축소는 데이터를 가장 잘 나타내는 일부 특성을 선택해 데이터 크기를 줄이고 지도학습모델의 성능을 향상시킬 수 있는 방법 / 손실을 최대한 줄이면서 줄어든 차원에서 원본 차원으로 복구도 가능
대표적인 차원 축소 알고리즘이 주성분 분석(PCA)

<주성분 분석 소개>
- 데이터에 있는 분산이 큰 방향을 찾는 것으로 이해 가능 / 분산은 데이터가 널리 퍼져있는 정도를 의미하는 데 분산이 큰 방향이라는 것은 데이터를 잘 표현하는 어떤 벡터라고도 말할 수 있음 --> 이 벡터를 주성분(Principal component)이라고 함
- 주성분 벡터의 원소 개수는 원본 데이터셋에 있는 특성 개수와 같다
- 주성분을 찾은 다음 해당 벡터에 수직이고 분산이 가장 큰 다음 방향을 찾음, 이것이 두번째 주성분
- 주성분은 원본 특성의 개수만큼 찾을 수 있다
- 기술적인 이유로 주성분은 원본 특성 개수와 샘플 개수 중 작은 값만큼만 찾을 수 있음
"""

## PCA 클래스
import numpy as np
fruits = np.load(file)
fruits_2d = fruits.reshape(-1,100*100)

from sklearn.decomposition import PCA # PCA 클래스의 객체를 만들 때 n_components 매개변수에 주성분 개수를 지정해야 함 / 비지도학습이기에 target 값 제공하지 않음
pca = PCA(n_components = 50)
pca.fit(fruits_2d)
print(pca.components_.shape) # n_components = 50 , 원본 데이터 특성 개수 10000 개이므로 (50, 10000)이 나옴
draw_fruits(pca.components_.reshape(-1,100,100)) # 원본 데이터에서 분산이 가장 큰 방향을 순서대로 나타낸 것 --> 찾아낸 주성분 50개로 특성 개수를 줄일 수 있음

print(fruits_2d.shape)
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape) # 성공적으로 차원 축소 (10000 -> 50)

## 원본 데이터 재구성 -> 10000개에서 50개로 줄이는 과정에서 손실이 발생하기는 했지만 분산이 가장 큰 방향으로 데이터를 투영했기에 상당 부분 재구성 가능
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape) # (300, 10000)개로 특성 복원
fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0,100,200] :
    draw_fruits(fruits_reconstruct[start:start + 100])
    print("\n") # 대부분의 파일이 제대로 복원되었음

## 설명된 분산 --> 주성분이 원본 데이터 분산을 얼마나 잘 나타내는지 기록한 값(explained variance) / PCA 클래스의 explained_variance_ratio_에 기록
print(np.sum(pca.explained_variance_ratio_)) # 0.9215215257927912 / 92%의 분산 유지

plt.plot(pca.explained_variance_ratio_)
plt.show(block = True) # 처음 10개의 주성분이 대부분의 분산을 표현

## PCA로 축소된 데이터를 사용한 지도학습
# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression() # 지도학습을 하려면 타깃값이 있어야 하므로 사과 : 0, 파인애플 : 1, 바나나 : 2로 지정
target = np.array([0]*100+ [1] * 100+ [2] * 100)

from sklearn.model_selection import cross_validate
scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score'])) # 0.9966666666666667
print(np.mean(scores['fit_time'])) # 0.2841970443725586 --> 각 교차 검증 폴드의 훈련 시간

scores = cross_validate(lr,fruits_pca, target)
print(np.mean(scores['test_score'])) # 1.0
print(np.mean(scores['fit_time'])) # .018738698959350587 --> 차원 축소로 시간이 대폭 감소

pca = PCA(n_components = 0.5) # n_components에 주성분 개수를 지정할 수도 있고 원하는 설명된 분산의 비율을 지정할 수도 있음
pca.fit(fruits_2d)
print(pca.n_components_) # 2
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape) # (300, 2) --> 주성분이 2개

scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score'])) # 0.9933333333333334 --> 2개의 주성분만 사용했는 데 99%의 정확도
print(np.mean(scores['fit_time'])) # 0.034353208541870114

# k-means
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3, random_state = 42)
km.fit(fruits_pca)
print(np.unique(km.labels_, return_counts =True)) # (array([0, 1, 2]), array([110,  99,  91], dtype=int64)) --> 이전 원본 데이터를 사용했을 때와 결과 비슷

for label in range(0,3) :
    draw_fruits(fruits[km.labels_ == label])
    print('\n')

# 훈련 데이터 차원을 줄이는 것의 장점 = 시각화 (3개 이하로 차원을 줄이면 화면 출력이 쉬움)
for label in range(0,3) :
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:, 0], data[:, 1])
plt.legend(['apple','banana','pineapple'])
plt.show(block=True) # 차원 축소를 통한 시각화로 또다른 통찰을 얻을 수 있음