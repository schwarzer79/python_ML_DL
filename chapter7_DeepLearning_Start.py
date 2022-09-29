### 인공신경망

## 패선 MNIST --> 딥러닝에서 자주 사용되는 데이터

from tensorflow import keras
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
# load_data()를 이용하면 훈련 데이터와 테스트 데이터를 나누어서 반환해줌 / (입력, 타깃) 의 쌍으로 구성
print(train_input.shape, train_target.shape) # (60000, 28, 28) (60000,) --> 60000개의 이미지로 훈련 데이터 구성 (28 * 28 사이즈)
print(test_input.shape, test_target.shape) # (10000, 28, 28) (10000,) --> 10000개의 이미지

# sample 10개에 해당하는 이미지 확인 + target number 확인
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1,10,figsize = (10,10))
for i in range(10) :
    axs[i].imshow(train_input[i], cmap = 'gray_r')
    axs[i].axis('off')
plt.show(block = True)
print([train_target[i] for i in range(10)]) # [9, 0, 0, 3, 0, 2, 7, 2, 5, 5] / target은 0~9까지의 숫자 레이블로 구성

# 레이블 당 샘플 수 확인
import numpy as np
print(np.unique(train_target, return_counts = True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64))

## 로지스틱 회귀로 패션 아이템 분류하기
"""
샘플이 60000개이기에 전체 데이터를 한번에 활용해 모델 훈련하는 것보다 하나씩 꺼내서 하는 것이 더 효율적 --> SGDClassifier (확률적 경사 하강법)
SGDClassifier 클래스의 loss 매개변수를 'log'로 하면 로지스틱 손실 함수를 최소로 만드는 확률적 경사 하강법 모델을 만들 수 있음
SGDClassifier 는 scaling을 하지 않으면 기울기가 가장 큰 특성을 따라 내려가기에 올바른 결과를 만들어 낼 수 없음 --> 0~1 사이로 정규화
SGDClassifier 는 2차원 배열을 다루지 못하기에 2차원 배열을 reshape로 1차원 배열 전환
"""
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
print(train_scaled.shape) # (60000, 784)

# SGDClassifier와 cross_validate를 이용한 성능 검증
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss = 'log', max_iter = 5, random_state = 42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs = -1)
print(np.mean(scores['test_score'])) # 0.8192833333333333 / max_iter=5 였지만 이를 늘려도 별 차이는 없음

"""
위에서 로지스틱 회귀분석으로 분석을 했는 데 이 떄 첫번쨰 레이블인 티셔츠에 대한 회귀식을 살펴보면 특성에 픽셀이 들어가기에 특성 수가 총 784개이다
두번쨰 레이블인 바지 또한 특성이 784개인 것은 동일하나 가중치와 절편값은 다를 것이다
이런 방식으로 모든 클래스에 대한 선형 방정식이 존재할 것이고 SGDClassifier는 클래스를 잘 구분할 수 있도록 10개 방정식에 대한 가중치와 절편을 찾은 것임
10개 방정식을 모두 계산한 이후 softmax 함수를 통과해 각 클래스에 대한 확률을 얻을 수 있음
"""

## 인공 신경망
"""
가장 기본적인 인공신경망은 확률적 경사 하강법을 이용하는 로지스틱 회귀와 동일
"""

import tensorflow as tf
from tensorflow import keras

## 인공 신경망으로 모델 만들기
"""
로지스틱 회귀에서는 교차 검증을 이용해 모델 평가를 했지만, 인공 신경망에서는 교차 검증을 잘 사용하지 않고 검증 세트를 별도로 덜어내어 사용
- 이유 - 
1. 딥러닝 데이터셋은 충분히 크기에 검증 점수가 안정적
2. 교차 검증 수행하기에는 훈련 시간이 너무 오래 걸림
"""

from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size = 0.2, random_state = 42)
print(train_scaled.shape, train_target.shape) # (48000, 784) (48000,)
print(val_scaled.shape, val_target.shape) # (12000, 784) (12000,)

# 밀집층 만들기
dense = keras.layers.Dense(10, activation = 'softmax', input_shape = (784,))
"""
입력층, 밀집층, 출력층
- 밀집층(dense layer) : 밀집층 중에서도 양쪽 뉴런이 모두 연결하고 있기 때문에 완전 연결층(fully connected layer)
keras.layers.Dense(뉴런 개수, 뉴런 출력에 적용할 함수, 입력 크기) 의 형식
- 분류하고 싶은 뉴런의 개수를 지정하고 10개 뉴런에서 출력되는 값을 확률로 바꾸기 위해서 softmax 함수를 사용 / 이진 분류라면 sigmoid를 사용
- input_shape 는 입력값의 크기로 10개 뉴런이 각각 몇 개의 입력을 받는지 튜플로 지정
- activation function = softmax나 sigmoid와 같이 뉴런 선형 방정식 계산 결과에 적용되는 함수 (앞으로는 a로 표시)
"""

# 밀집층을 가진 신경망 모델 만들기
model = keras.Sequential(dense)

## 인공 신경망으로 패션 아이템 분류하기
"""
keras 모델은 훈련하기전 설정 단계가 있음 --> model의 compile() 메소드에서 수행 / 반드시 지정해야할 것은 손실함수의 종류 / 훈련 과정에서 계산하고 싶은 측정값을 지정
~ 손실함수
이진 분류 : loss = 'binary_crossentropy' = 이진 크로스 엔트로피 손실 함수
다중 분류 : loss = 'categorical_crossentropy' = 크로스 엔트로피 손실 함수

sparse는 왜 붙는가 ? --> 이진 크로스 엔트로피 함수는 -log(예측 확률) * 타깃값(정답) 을 수행
binary인 경우 출력층의 뉴런이 하나이고 뉴런이 출력하는 확률값 a(sigmoid 출력값)을 사용해 양성, 음성 클래스에 대한 크로스 엔트로피를 계산
binary 의 출력 뉴런은 양성 클래스에 대한 a만 계산하기에 음성 클래스는 1-a로 계산 가능 / 타깃값 또한 양성 = 1, 음성 = 0 --> 이런식으로 뉴런이 하나라도 양성, 음성 클래스에 대한 크로스 엔트로피
 손실을 모두 계산할 수 있음 --> binary가 아닌 categorical 이라면?
 
catergorical 일 떄는 10개의 출력층에서 10개의 클래스에 대한 확률을 모두 출력하고 타깃에 해당하는 확률만 남겨놓기 위해서 나머지 확률에 모두 0을 곱함
[a1, a2, a3, a4, a5, ... , a10] * [1, 0, 0, 0, 0, 0 ... 0] 의 형식 = a1만 남게됨
-> 신경망은 손실을 낮추려면 a1의 값을 가능한 1에 가깝게 만들어야 함
마찬가지로 두번째 뉴런의 활성화 출력인 a2만 남기려면 [0,1,0,0...,0]을 곱해 a2만 남길 수 있고 샘플을 정확하게 분류하려면 신경망이 a2의 출력을 가능한 한 높여주어야 한다
---> 위와 같이 타깃값을 해당 클래스만 1, 나머지는 0인 배열로 만드는 것 = one-hot encoding / 다중 분류에서는 크로스 엔트로피 손실 함수를 사용하기 위해서는 0,1,2,...식의 타깃값을 원-핫 인코딩으로 변경

metrics 매개변수
-> keras는 모델 훈련 시 기본적으로 에포크마다 손실 값을 출력, 손실이 줄어드는 것을 보고 훈련이 잘 되었다는 것을 알 수도 있지만 정확도를 함께 출력하는 것이 더 좋음 => metrics = 'accuracy'
"""
model.compile(loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
print(train_target[:10]) # [7 3 5 8 6 9 3 3 9 9] --> 모두 정수로 구성 / 하지만 tensorflow에서는 정수로 된 타깃값을 원-핫 인코딩으로 변경하지 않고 그냥 사용할 수 있음 = sparse_catergorical_crossentropy

# model fitting
model.fit(train_scaled, train_target, epochs = 5) # tensorflow는 인공신경망을 만들고 훈련할 떄 랜덤하게 움직이는 특성이 있어 결과값이 매번 다름
# 점점 loss는 감소하고 정확도는 늘어가는 모습 / 최종적으로 5번째 epoch에서는 accuracy = 85%

# validation set에서의 성능 테스트 -> evaluate()
model.evaluate(val_scaled, val_target) # [0.44676724076271057, 0.8527500033378601] = [loss, accuracy] -> 검증 set의 점수는 훈련 셋보다 조금 낮은 것이 특징

### 심층 신경망

from tensorflow import keras
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

from sklearn.model_selection import train_test_split
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1,28 *28)
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size = 0.2, random_state = 42)

"""
입력층과 출력층 사이에 밀집층이 추가될 수 있는 데 이러한 밀집층을 은닉층(hidden layer)라고 함
은닉층에는 활성화 함수가 표시되어 있음 / 출력층에 적용되는 활성화 함수는 sigmoid, softmax 등으로 제한적이지만 은닉층에 적용되는 활성화 함수는 비교적 자유로움 (sigmoid OR ReLU)

회귀 문제에서는 출력층 활성화 함수를 사용하지 않음 -> 확률을 그대로 출력하면 됨

<은닉층에 활성화 함수를 적용하는 이유?>
만약 두 개의 방정식이 있고 하나의 방정식에서 나온 결과값을 그대로 다음 방정식에 넣기만 해서 계산한다면 굳이 두개의 방정식으로 나누어질 필요가 없고 합쳐버릴 수 있음
하지만, 사이에서 (은닉층) 어떠한 산술계산을 선형적이 아닌 비선형적으로 수행한다면 나름의 역할을 가질 수 있음 / 은닉층에는 항상 활성화 함수가 존재(그림에는 은닉층에 통합되어 없을 수 있음)
"""

# 은닉층 활성화 함수 = sigmoid / 출력층 활성화 함수 = softmax 인 인공 신경망
dense1 = keras.layers.Dense(100, activation = 'sigmoid', input_shape = (784,))
dense2 = keras.layers.Dense(10, activation = 'softmax')

"""
은닉층의 뉴런 개수를 지정하는 것은 별다른 기준이 없고 분석자의 경험에 의존해야 함 / 적어도 출력층의 뉴런보다는 많아야 한다는 제약은 있음
출력층의 뉴런 개수는 10개의 클래스 분류에 관한 문제이므로 10개 지정
"""

## 심층 신경망 만들기
model = keras.Sequential([dense1, dense2]) # 각 dense 들을 list로 만들어서 전달, 단 순서는 가장 처음 등장하는 은닉층에서 마지막 출력층의 순서를 지켜야 함
"""
인공신경망의 강점은 이렇게 여러 개의 층을 추가해서 입력 데이터에 대해 연속적인 학습을 진행하는 능력에서 나옴
"""

model.summary() # 층에 대한 유용한 정보 호출
"""
Model: "sequential_2" --> 모델 이름
_________________________________________________________________  
 Layer (type)                Output Shape              Param #   
================================================================= # 모델에 들어있는 층 나열 + 층마다 층 이름, 클래스, 출력 크기, 모델 파라미터 개수 출력
 dense_3 (Dense)             (None, 100)               78500                      # 층을 만들 때 name 매개변수로 이름 지정 가능
                                                                                        # 출력 크기는 (None, 100) 의 형식 / 첫번쨰 차원은 샘플 개수인 데, 이것이 none인 이유는 keras 모델의 fit()는 훈련 데이터
 dense_4 (Dense)             (None, 10)                1010                        # 주입 시 한번에 모든 데이터를 사용하지 않고 잘게 나누어 여러 번에 걸쳐 사용하는 minibatch Gradient Descent를 사용
                                                                                        # 기본 minibatch 크기는 32개 / fit() 메소드에서 batch_size로 변경 가능                                                                 
================================================================= # 모델 파라미터 개수 출력 
Total params: 79,510                                                                # 79,510 = 785 * 100 + 101 * 10
Trainable params: 79,510                                                            # 간혹 훈련되지 않는 파라미터(Non - trainable params)가 있을 수 있음
Non-trainable params: 0
_________________________________________________________________
"""

## 층을 추가하는 다른 방법
"""
앞에서는 Sequential([dense1,dense2]) 의 형식으로 층을 추가했지만 일반적으로는 따로 층을 만들어서 넣지 않고 Sequential() 메소드에서 생성
"""
model = keras.Sequential([keras.layers.Dense(100, activation='sigmoid', input_shape = (784,), name = 'hidden'),
                          keras.layers.Dense(10, activation = 'softmax', name = 'output')],
                         name = '패션 MNIST 모델') # NAME으로 model의 이름을 지정할 수도 있음 / 모델 이름과 달리 층 이름은 반드시 영어

model.summary()
"""
Model: "패션 MNIST 모델"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 hidden (Dense)              (None, 100)               78500     
                                                                 
 output (Dense)              (None, 10)                1010      
                                                                 
=================================================================
Total params: 79,510
Trainable params: 79,510
Non-trainable params: 0
_________________________________________________________________
"""
# 편하기는 하지만 Sequential 클래스 생성자가 길어지는 단점과 조건부 층 추가가 불가능 --> 층 추가 메소드인 add()를 가장 많이 사용

model = keras.Sequential()
model.add(keras.layers.Dense(100, activation = 'sigmoid', input_shape = (784,)))
model.add(keras.layers.Dense(10, activation = 'softmax'))
model.summary()

# model fitting
model.compile(loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
model.fit(train_scaled, train_target, epochs = 5)

# 몇 개의 층을 추가하더라도 compile(), fit() 사용법은 동일

## 렐루 함수 (ReLU)
"""
초창기 은닉층 활성화 함수로는 sigmoid를 많이 사용했지만 sigmoid는 양 끝으로 갈수록 그래프가 누워있기에 올바른 출력을 만드는 데 신속한 대응을 하지 못함 --> ReLu 함수 제안
ReLU는 입력이 양수인 경우에는 활성화 함수가 없는 것처럼 입력을 통과시키고 음수인 경우에는 0으로 만듬
max(0,z) : z가 0보다 크면 출력, 아니면 0 / 이미지 처리에서 성능이 좋음

keras 제공 기능 중 차원에 관한 것
이미지 데이터를 인공 신경망에 넣기 위해 numpy의 reshape() 메소드를 활용해서 1차원으로 펼쳤는 데 이를 keras의 Flatten 층을 이용할 수 있음
Flatten은 배치 차원을 제외하고 나머지 입력 차원을 모두 일렬로 펼치는 역할만 수행하기에 인공 신경망 성능 향상에는 영향 없지만 이 클래스를 층처럼 입력층과 출력층 사이에 넣기에 층이라고 부름
"""
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape = (28,28)))
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))
model.summary()
""" # flatten을 추가하면 입력층의 차원을 짐작할 수 있다는 것이 장점
Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten (Flatten)           (None, 784)               0         
                                                                 
 dense_7 (Dense)             (None, 100)               78500     
                                                                 
 dense_8 (Dense)             (None, 10)                1010      
                                                                 
=================================================================
Total params: 79,510
Trainable params: 79,510
Non-trainable params: 0
_________________________________________________________________
"""

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size = 0.2, random_state = 42)

model.compile(loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
model.fit(train_scaled, train_target, epochs = 5)

model.evaluate(val_scaled, val_target) #  [0.3570577800273895, 0.8773333430290222] --> sigmoid 함수를 사용했을 때보다 더 올랐음

## Optimizer
"""
신경망에는 HyperParameter가 매우 많음 (지금까지 나온 것 = 은닉층 개수, 뉴런 개수, 활성화 함수, 층 종류, 배치 사이즈 매개변수, epoch 매개변수)
keras 에서는 다양한 종류의 경사 하강법 알고리즘을 제공하는 데 이를 Optimizer 라고 함 -> 다른 Optimizer를 테스트할 필요가 있음 + 각 알고리즘 학습률 매개변수도 있음
때문에 hyperparameter의 최적값을 찾는 것이 매우 어려움
- 가장 기본적인 Optimizer는 확률적 경사 하강법인 SGD(1개씩 뽑아서 하지는 않고 minibatch 방식을 사용)
model.compile(optimizer = 'sgd', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
==
sgd = keras.optimizers.SGD()
model.compile(optimizer = sgd, loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')

만약 sgd의 학습률 매개변수를 변경하고 싶다면 sgd = keras.optimizer.SGD(learning_rate = 0.1) 으로 변경 가능

- 기본 경사 하강법 Optimizer
이 optimizer들은 모두 sgd 클래스에서 제공하고 기본적인 sgd의 경우 momentum 매개변수 값이 0, 이를 0보다 크게 지정하면 이전 gradient를 가속도로 사용하는 momentum optimization을 사용
보통 momentum 매개변수는 0.9 이상을 지정 
SGD 클래스의 nesterov 매개변수를 True로 바꾸면 '네스테로프 모멘텀 최적화'를 사용
    sgd = keras.optimizers.SGD(momentum = 0.9, nesterov = True) / nesterov는 모멘텀 최적화를 2회 반복해 구현 / 대부분 네스테로프 모멘텀 최적화가 기본 sgd보다는 좋은 성능을 제공
모델이 최적점에 가까워질수록 학습률을 낮출 수 있고 이러면 안정적으로 최적점에 수렴할 가능성이 높음, 이 때의 학습률이 적응적 학습률(adaptive learning rate)
 --> adaptive learning rate를 사용하면 학습률 매개변수를 튜닝하는 수고를 덜 수 있는 것이 장점
 
-적응적 학습률 적용 optimizer = Adagrad , RMSprop / optimizer 매개변수의 기본값이 rmsprop    
adagrad = keras.optimizers.Adagrad()
model.compile(optimizer = adagrad, loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
rmsprop = keras.optimizers.RMSprop()
model.compile(optimizer = rmsprop, loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')

- 모멘텀 최적화와 RMSprop의 장점을 접목한 것 = Adam / 가장 처음 시도해볼 수 있는 좋은 알고리즘
"""

# Adam 클래스를 사용한 모델
model =  keras.Sequential()
model.add(keras.layers.Flatten(input_shape = (28,28)))
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics = 'accuracy')
model.fit(train_scaled, train_target, epochs = 5) # RMSprop을 사용했을 때와 거의 유사한 결과

model.evaluate(val_scaled, val_target) # [0.345048189163208, 0.8775833249092102]