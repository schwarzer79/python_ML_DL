bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]  # 도미 길이
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]  # 도미 무게

# 생선을 구분할 수 있는 특징 = 특성(feature) -> 여기서는 생선의 길이와 무게를 의미
# 특성을 잘 이해하기 위해서 그래프로 표현 -> 산점도(scatter plot)
# matplotlib 패키지를 이용한 산점도 그리기

import matplotlib.pyplot as plt

plt.scatter(bream_length,bream_weight, c = 'b', marker = 'o') # plt.scatter(x,y, c = '색', marker = '마커 스타일')
plt.xlabel('length') # x 이름
plt.ylabel('weight') # y 이름
plt.show(block=True) # block=True 옵션을 넣어야 렉 안걸리고 플롯이 정상적으로 출력

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]  # 빙어 길이
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]  # 빙어 무게

plt.scatter(bream_length,bream_weight)
plt.scatter(smelt_length,smelt_weight)
plt.xlabel('length') # x 이름
plt.ylabel('weight') # y 이름
plt.show(block=True) # scatter 두 개를 넣어서 작성하면 최종 결과에 각 색깔 별로 구분되어 나타남

### k-최근접 이웃(k-Nearest Neighbors)을 이용한 머신러닝 프로그램 작성

# 1. 데이터 병합(빙어+방어)
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# 책에서 사용하는 머신러닝 패키지는 scikit-learn / 해당 패키지를 사용하려면 리스트를 2차원 리스트로 만들어야 함 --> zip() 함수와 리스트 내포 구문 활용

fish_data = [[l,w] for l,w in zip(length,weight)] # 2차원 리스트 생성

# 추가적으로 정답을 학습시켜줄 정답 데이터를 준비 -> 도미 = 1, 빙어 = 0
fish_target = [1] * 35 + [0] * 14
print(fish_target)

# scikit-learn에서 k-Nearest Neighbors 알고리즘 클래스를 import
from sklearn.neighbors import KNeighborsClassifier # 매개변수 p : 거리 측정 방법 지정, 1 = 맨해튼 거리, 2=유클리디안 거리(기본값은 2) / n_jobs : 사용 cpu 코어 개수 지정(-1은 모든 코어, 기본값은 1) / n_neighbors = 분류에 사용되는 주변 데이터 개수

kn = KNeighborsClassifier() # import한 알고리즘 클래스를 이용해 객체 생성 -> 생성된 객체에 방어와 빙어의 데이터를 넣어 훈련(fitting) -> fit()함수를 이용
kn.fit(fish_data,fish_target) # fitting
kn.score(fish_data,fish_target) #score()를 이용한 평가 -> 결과값으로 accuracy를 반환

# 위 머신러닝 알고리즘으로 사용한 k-nearsest neighbors 는 어떠한 데이터의 분류 답을 구할 때 주변에 있는 데이터를 보고 결정
# if) 어떠한 데이터를 넣었을 때 이것이 어느쪽으로 분류될지 예측
kn.predict([[30,600]]) # fitting 시에 데이터를 [[ ]] 의 2차원 리스트로 전달했기에 여기서도 같은 형식을 유지

# 단 이러한 알고리즘 방식 때문에 데이터가 엄청나게 많다면 사용하기 어려움 -> 메모리가 많이 필요하고 계산 시간 소요

print(kn._fit_X) # 분석자가 전달한 fish_data를 호출
print(kn._y) # fish_target

# 해당 알고리즘에서 가까운 데이터를 참고해 분류하는 데, 이 값의 default는 5 / n_neighbors 변수로 변경 가능

kn49 = KNeighborsClassifier(n_neighbors=49)
kn49.fit(fish_data,fish_target)
kn49.score(fish_data,fish_target) # 49개의 데이터를 참조한다면 모든 데이터에 대해 참조하는 것이므로 숫자가 많은 도미가 무조건 결과로 나오게 됨


# n_neighbors 를 5~50까지 늘리면서 score < 1인 case 찾기

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)

for n in range(5,50) :
    kn.n_neighbors = n
    score = kn.score(fish_data, fish_target)
    if score<1:
        print(n, score)
        break
