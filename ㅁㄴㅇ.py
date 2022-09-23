"""
클래스 과제 : kNN 알고리즘 구현
"""

import random  # 난수 생성 모듈
from statistics import sqrt  # 제곱근 함수

random.seed(100)  # 난수 seed값 지정(난수 고정)

### 단계1. data 생성

# 1) 적용데이터셋 10개  : 소숫점 5자리 반올림
dataset = [[round(random.random(), 5), round(random.random(), 5)] for i in range(10)]

# 2)기준데이터셋 1개 : 소숫점 5자리 반올림
base = [round(random.random(), 5), round(random.random(), 5)]

print('dataset :', dataset)
print('base : ', base)
print('-' * 60)

# # 단계3. kNN 분류기 객체 생성
# knn = kNN() # (적용데이터셋, 기준데이터셋)

#  class kNN :

#     q = dataset
#     p = base

#     def __init__(self,q,p):
#         self.q = q
#         self.p = p

#     def distance(self):
#         for i in dataset:
#           distance = [round(sqrt((self.p[0] - i[0]) ** 2
#                      + (self.p[1] - i[1]) ** 2), 5) for i in self.q]

#
#           return distance


distance = [round(sqrt((base[0] - i[0]) ** 2 + (base[1] - i[1]) ** 2), 5) for i in dataset]

# dictionary = dict{'distance[i]':q[i]}
# dictionary = {i : distance[i] for i in range(len(q))}
# print(dictionary)

sorted_distance = sorted(distance)

print(sorted_distance)

print('k1 ->', sorted_distance[0], '\n''real data :', dataset[0])
print('k3 ->', sorted_distance[0:3], '\n''real data :', dataset[0:3])
print('k5 ->', sorted_distance[0:5], '\n''real data :', dataset[0:5])
