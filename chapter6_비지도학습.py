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