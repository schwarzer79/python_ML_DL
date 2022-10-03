### 인공신경망

## 패선 MNIST --> 딥러닝에서 자주 사용되는 데이터

#pip install tensorflow
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

*** One - Hot Encoding 과 label Encoding ***
1. Categorical Encoding 이란?
머신은 텍스트가 아닌 숫자만을 이해하기에 텍스트로 이루어진 자료를 숫자형으로 변환해줄 필요가 있다. 때문에 catergorical encoding 은 필수적인 작업
catergorical encoding을 크게 두 가지 방법으로 분류하면 Label Encoding + One-Hot Encoding 

2. Label Encoding (관련 질의 : https://stackoverflow.com/questions/59914210/why-shouldnt-the-sklearn-labelencoder-be-used-to-encode-input-data)
텍스트를 알파벳 순서대로 정렬한 후 그 순서대로 번호를 매겨 숫자를 할당해준다는 뜻, 하지만 데이터 특성에 순서나 랭크가 존재하지 않는 경우도 있어 분석자가 원하지 않는 방향으로 encoding 될 수 있음
* 주의할점
 1) sklearn의 LabelEncoder 는 1차원 배열만을 입력으로 받음 -> dataframe을 넣을 수 없음 (각 column마다 LabelEncoder를 불러와서 처리해야함)
 2) 웬만해서는 사용하지 않는 것이 좋음 -> 순서가 없는(독립적인) 속성값들을 연속형 수치들로 바꾸게 되면 학습과정에서 의도치 않은 가중치에 차이를 두게 됨 / 떄문에 변수의 cardinality가 높아도 label encoding보다는 
 범주형 변수 자체를 처리할 수 있는 분류기를 사용하는 것이 best(CatBoost, lightGBM 등은 범주형 변수를 encoding 없이 넣어도 처리할 수 있게 되어있음)
 
** cardinality 에 관하여 
- 중복도가 낮으면 cardinality가 높고 중복도가 높으면 cardinality가 낮다고 말할 수 있음 
- 예를 들어 주민등록번호 같은 경우 중복도가 낮기 때문에 cardinality가 높고 성별이나 이름의 경우 주민등록번호와 다르게 중복도가 높으므로 cardinality가 낮다고 할 수 있음

3. One-Hot Encoding
목록값을 이진값으로 변환하는 방법 / 더미변수를 만드는 것과 유사
- sklearn의 OneHotEncoder()를 사용하거나 pandas의 get_dummies()를 사용할 수 있음 / pandas의 get_dummies()를 사용하는 것이 각 column 명에 변수특성을 명시해줘 OneHotEncoder()를 통한 명시보다는 보기 편함

4. 선택 기준
데이터셋에 순서가 없고 고유값의 개수가 많지 않다면? -> One - Hot Encoding
데이터셋에 순서가 존재하고 고유값의 개수가 많다면? -> Label Encoding


*** metrics 매개변수 ***
-> keras는 모델 훈련 시 기본적으로 epoch마다 손실 값을 출력, 손실이 줄어드는 것을 보고 훈련이 잘 되었다는 것을 알 수도 있지만 정확도를 함께 출력하는 것이 더 좋음 => metrics = 'accuracy'
"""
model.compile(loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
print(train_target[:10]) # [7 3 5 8 6 9 3 3 9 9] --> 모두 정수로 구성 / 하지만 tensorflow에서는 정수로 된 타깃값을 원-핫 인코딩으로 변경하지 않고 그냥 사용할 수 있음 = sparse_catergorical_crossentropy

# model fitting
model.fit(train_scaled, train_target, epochs = 5) # tensorflow는 인공신경망을 만들고 훈련할 때 랜덤하게 움직이는 특성이 있어 결과값이 매번 다름
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

*** keras 제공 기능 중 차원에 관한 것 ***
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

### 신경망 모델 훈련
"""
이전까지 tensorflow의 keras API를 이용해 인공 신경망을 만들기 + add(),Flatten() 등을 이용한 은닉층 추가 + Optimizer 들을 적용하는 방법
sklearn에서 배우던 머신러닝 알고리즘은 좋은 성능을 위해 매개변수를 조정하고 훈련 -> 모델 구조가 어느정도 고정되어있음
인공신경망은 모델의 구조를 직접 만드는 느낌이 강함
"""

## 손실 곡선
"""
fit() 으로 모델을 훈련하면 epoch, loss, accuracy 등을 볼 수 있었음 + print() 명령이 없어도 자동으로 마지막 라인의 실행 결과를 출력 = fit() 메소드가 무언가를 반환한다는 의미 -> fit()은 History 클래스 객체를 반환
History 객체에는 loss와 accuracy 값이 저장되어 있음
"""

from tensorflow import keras
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target  = train_test_split(train_scaled, train_target, test_size = 0.2, random_state = 42)

def model_fn(a_layer = None) :
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape = (28,28)))
    model.add(keras.layers.Dense(100, activation = 'relu'))
    if a_layer :
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation = 'softmax'))
    return model # if문을 제외하고는 앞에서 한 모델 생성과 동일 / if문에는 a_layer 변수를 추가하면 은닉층 뒤에 또다른 층을 하나 더 추가해 주는 것

model = model_fn()
model.summary()
model.compile(loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
history = model.fit(train_scaled, train_target, epochs = 5, verbose = 0)
"""
verbose : 훈련과정의 출력을 조절하는 매개변수로 default = 1, 1이면 epoch마다 진행 막대와 손실 등의 지표 출력 / 2이면 진행 막대를 뺴고 출력 / 0이면 훈련 과정 나타내지 않음
"""
print(history.history.keys()) # history 객체에는 history 딕셔너리가 들어있음 (key값으로 loss, accuracy를 가짐)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show(block=True)

plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show(block=True) # epoch가 커질수록 loss는 감소하고 accuracy는 증가하는 그래프

# epoch = 20으로 증가
model = model_fn()
model.compile(loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
history = model.fit(train_scaled, train_target, epochs = 20, verbose = 0)
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show(block=True)

## 검증 손실
"""
stochastic gradient descent 를 사용했을 때 과대/과소적합과 epoch의 관계를 이전에 알아봤음 -> 인공신경망은 일종의 SGD를 사용하기에 동일한 개념이 적용됨
epoch에 따른 과대/과소적합을 판단하려면 훈련셋뿐만 아니라 검증셋에 대한 결과도 있어야 함 / 이전에는 accuracy에 관해 설명했지만 여기서는 loss를 이용
인공 신경망이 최적화하는 것은 loss, accuracy와 loss가 비례하는 경우도 있지만 아닌 경우도 있음, 때문에 모델이 잘 fitting되었는지 확인하는 데는 loss를 사용하는 것이 바람직
"""
model = model_fn()
model.compile(loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
history = model.fit(train_scaled, train_target, epochs = 20, verbose = 0, validation_data = (val_scaled, val_target)) # epoch마다의 검증 손실 계산을 위해 validation_data 매개변수에 (입력, 타깃) 튜플을 만들어 전달
print(history.history.keys()) # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

# 그래프 그리기
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show(block = True) # 훈련 loss는 꾸준히 감소하고 검증 loss는 특정 지점까지는 감소하다 다시 증가하므로 과대적합 model / val_loss가 감소하는 시점을 가능한 한 뒤로 늦추면 검증 세트에 대한 소실이 줄어들고 정확도도 증가할 것

# 과대적합 해결을 위한 Optimizer 설정 - Adam / 기본적으로는 RMSprop을 사용
model = model_fn()
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
history = model.fit(train_scaled, train_target, epochs = 20, verbose = 0, validation_data = (val_scaled, val_target))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show(block = True) # 과대적합이 확실히 감소 / val_loss가 감소하는 추세가 RMSprop을 사용할 때보다 확실하게 뒤로 밀렸음 --> 이 데이터셋에는 Adam Optimizer가 더 잘 어울림 / 학습률을 조정해 더 나은 손실 곡선을 얻을 수도 있음

## 드롭아웃(Dropout)
"""
Dropout은 훈련 과정에서 층에 있는 일부 뉴런을 랜덤하게 꺼서(뉴런 출력을 0로) 과대적합을 막는다 / 얼마나 많은 뉴런을 드롭아웃할지 결정하는 것은 분석자의 선택
- 일부 뉴런이 랜덤하게 꺼지면 특정 뉴런에 과도하게 의지하는 것을 줄일 수 있고, 일부 뉴런 출력이 없을 수 있다는 것을 감안하면 신경망은 더 안정적인 예측을 할 수 있을 것
- 이는 2개의 신경망을 앙상블하는 것과 유사 (과대적합을 막을 수 있음)
- keras.layers.Dropout으로 제공 / 층으로서 입력되긴 하지만 훈련되는 모델 parameter는 없음
"""

model = model_fn(keras.layers.Dropout(0.3))
model.summary()
"""
Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten_4 (Flatten)         (None, 784)               0         
                                                                 
 dense_8 (Dense)             (None, 100)               78500     
                                                                 
 dropout (Dropout)           (None, 100)               0         
                                                                 
 dense_9 (Dense)             (None, 10)                1010      
                                                                 
=================================================================
Total params: 79,510
Trainable params: 79,510
Non-trainable params: 0
_________________________________________________________________
-  dropout 층에서는 훈련되는 모델 parameter 가 없음 / 일부 뉴런 출력을 0으로 하지만 전체 출력 배열의 크기를 바꾸지는 않음
- 단, 훈련이 끝난 후 평가나 예측을 할 때에는 드롭아웃을 적용하면 안됨(훈련된 모든 뉴런을 사용해야 올바른 예측) / keras와 tensorflow에서는 모델이 평가와 예측에 사용될 때에는 자동으로 dropout을 적용하지 않음
"""

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
history = model.fit(train_scaled, train_target, epochs = 20, verbose = 0, validation_data = (val_scaled, val_target))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show(block = True) # 10번쨰 epoch에서 val_loss가 증가하지 않고 추세를 유지하지만 20번의 epoch를 수행했기에 다소 과대적합 되어있음 -> epochs = 10으로 변경

## 모델 저장 & 복원
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
history = model.fit(train_scaled, train_target, epochs = 10, verbose = 0, validation_data = (val_scaled, val_target))

model.save_weights('model-weights.h5') # 모델의 파라미터 저장 / 기본적으로 tensorflow의 체크포인트 format으로 저장하지만 확장자를 '.h5'로 지정하면 HDF5 format으로 저장
model.save('model-whole.h5') # 모델 구조와 파라미터를 저장 / 기본적으로 SavedModel 포맷으로 저장되지만 확장자를 'h5'로 저장하면 HDF5 포맷으로 저장

model = model_fn(keras.layers.Dropout(0.3))
model.load_weights('model-weights.h5') # 새로운 모델을 생성하고 저장했던 모델 파라미터를 적재 / load_weights()를 사용하려면 save_weights()를 사용했던 모델과 정확히 같은 구조여야 함

"""
keras.predict() : 샘플마다 10개의 클래스에 대한 확률을 반환
검증 set의 샘플 개수는 12,000개 이기에 predict()에서는 (12000,10) 배열을 반환 -> 10개 확률 중 가장 큰 것을 골라 target label과 비교해 정확도 계산
"""
import numpy as np
val_labels = np.argmax(model.predict(val_scaled), axis = -1) # 가장 큰 값을 고르기 위해 argmax / argmax 의 axis = -1은 배열 마지막 차원을 따라 최댓값을 선택
print(np.mean(val_labels == val_target)) # 0.8743333333333333

model = keras.models.load_model('model-whole.h5')
model.evaluate(val_scaled, val_target) # [0.3448579013347626, 0.8743333220481873]

## CALLBACK : 훈련 과정 중간에 어떤 작업을 수행할 수 있게 하는 객체
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5', save_best_only = True)
model.fit(train_scaled, train_target, epochs = 20, verbose = 0, validation_data = (val_scaled, val_target), callbacks = [checkpoint_cb])

model = keras.models.load_model('best-model.h5')
model.evaluate(val_scaled, val_target) #  [0.3207039535045624, 0.8863333463668823]
"""
callbacks.ModelCheckpoint 클래스에 객체 checkpoint_cb를 만들어 이를 fit()의 callbacks 매개변수에 리스트로 감싸 전달
ModelCheckpoint 콜백이 검증 점수가 가장 낮은 모델을 자동으로 저장 
검증 점수가 상승하기 시작한다면 그 이상의 epoch부터는 과대적합이기에 훈련을 계속할 필요가 없음 -> 이렇게 과대적합 전에 훈련을 중지하는 것을 early stopping이라고 함
early stopping은 훈련 epoch 횟수를 제한하는 역할이지만 과대적합을 막아주기에 규제 방법의 일종으로 생각 가능

keras.callbacks.EarlyStopping(patience = , restore_best_weights = )
- patience 매개변수 = 검증 점수가 향상되지 않더라도 참을 epoch횟수 / patience = 2라면 2번 연속 검증 점수가 향상되지 않으면 훈련 중지
- restore_best_weights : True라면 가장 낮은 검증 손실을 낸 모델 파라미터로 되돌림
EarlyStooping을 ModelCheckpoint랑 같이 사용하면 가장 낮은 검증 loss의 모델을 파일에 저장하고 검증 손실이 상승할 때 훈련 중지 가능, 훈련 중지 후 현제 모델 파라미터를 최상의 파라미터로 복구
"""

## Callback .ModelCheckpoint + callback.EarlyStopping 사용
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True)
model.fit(train_scaled, train_target, epochs = 20, verbose = 0, validation_data = (val_scaled, val_target), callbacks = [checkpoint_cb,early_stopping_cb])

print(early_stopping_cb.stopped_epoch) # stopped_epoch 속성에서 훈련이 중지된 epoch를 알 수 있음 / patience = 2였으므로 13-2 = 11번째 epoch가 최상의 모델

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show(block = True)

model.evaluate(val_scaled, val_target) # [0.3209066092967987, 0.8818333148956299]