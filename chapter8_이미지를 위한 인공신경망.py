### 합성곱 신경망의 구성 요소
"""
*** 합성곱 ***
- 입력 데이터에 무언가의 작업을 통해 유용한 특성만을 드러나게 하는 것
- 밀집층에서 입력으로 들어온 개수만큼 가중치를 가지고 해당 입력마다 가중치(w)를 곱하고 절편을 더한다 --> 1개의 출력 생성
- 인공신경망은 10개의 입력에서 가중치 w1 ~ w10과 절편 b를 랜덤하게 초기화한 후 다음 epoch를 반복하면서 SGD를 사용해 loss가 낮아지는 방향으로 최적 가중치와 절편을 찾아감 = 모델 fitting
- 합성곱은 입력 전체에 가중치를 적용하지 않고 일부에만 가중치를 곱함
- n (n < 10) 울 선택하고 처음부터 n개 특성을 통해 1개의 출력을 생성 -> 한칸 내려가 새로운 출력 생성 / 단, 이 때 곱해지는 가중치 w는 모든 반복에서 같음
 (Ex) if n == 3, 1~3 입력 * 가징치(w) = 1개 출력 / 2~4 입력 * 가중치(w) = 1개 출력 --> 10개 입력에서 n=3이라면 8개의 출력이 생성 / n = Hyperparameter
- 합성곱 신경망(CNN)에서는 뉴런이라고 부르기 애매해 완전연결신경망과는 달리 뉴런을 필터(filter)라고 부르거나 커널(kernel)이라고 함
    * 완전 연결 신경망 = 완전 연결 층(밀집층)만을 사용해 만든 신경망
- 여기서는 kernel 을 입력에 곱하는 가중치, filter 를 뉴런 개수로 사용
- 합성곱의 장점은 1차원 입력 뿐만 아니라 2차원 입력에서도 사용 가능하다는 것
- input이 2차원 배열이라면 filter 또한 2차원이엉 함 / 오른쪽으로 이동하다가 더이상 이동하지 못할 때, 한칸 내려가 왼쪽부터 다시 시작
 (Ex) 4*4 input을 3*3 filter로 찍으면? -> 4개의 출력 발생 / 발생한 4개의 출력을 2차원으로 배치 -> (4,4) input을 (2,2) output으로 압축한 느낌
- 합성곱 계산을 통해 얻은 출력 = 특성맵(feature map)
- 여기서 filter를 하나만 사용하지 않고 t개를 사용하면 (2,2,t)의 feature map 들이 쌓여진 모습이 만들어짐
- 밀집층에서 각 뉴런에 대한 가중치가 모두 다르듯이 합성곱 층에 있는 필터 가중치(커널) 또한 모두 다름 -> 같은 가중치를 가진 filter를 여러개 사용할 이유는 없음
- 밀집층과는 다르게 2차원 구조를 그대로 유지할 수 있어서 이미지 처리 분야에서 성능이 좋음


*** 케라스 합성곱 층 ***
from tensorflow import keras
keras.layers.Conv2D(10, kernel_size = (3,3), activation = 'relu') -> 왼쪽에서 오른쪽, 위에서 아래로 움직이는 합성곱은 Conv2D 클래스를 이용
- 첫번째 매개변수 = filter의 개수 / 두번째 매개변수(kernel_size) = 커널의 크기 / activation = 활성화함수
- feature map은 활성화 함수까지 모두 통과하고 난 뒤의 결과
- kernel 크기는 hyperparameter이므로 분석자가 최적값을 찾아야함, 일반적으로는 (3,3), (5,5)
- 합성곱 신경망이라고 모든 층을 합성곱 층으로 구성해야 하는 것은 아니고 1개 층이라도 합성곱을 사용하면 합성곱 신경망이라고 부름
    -> 클래스 확률을 계산하려면 마지막 층에 클래스 개수만큼의 뉴런을 가진 밀집층을 두는 것이 일반적

*** 패딩과 스트라이드 ***
합성곱을 이용하면 (4,4) input에 (3,3) kernel을 적용해 (2,2) output을 만들어 낼 수 있었음 -> output을 input 크기와 동일하게 (4,4) 로 유지하려면 어떻게?
- 동일한 출력을 만드려면 마치 더 큰 입력에 합성곱하는 척을 해야함 -> (4,4) input이지만 (6,6)처럼 취급
- (6,6) 크기에 (3,3) 커널로 합성곱 했을 때 출력 크기는? -> (4,4) => 이렇게 가상의 원소로 input배열 주위를 채우는 것 = 패딩(padding)
- 실제 입력값은 아니기에 모두 0으로 채움 -> 단순히 도장 찍을 기회를 늘려주는 역할 (계산에 영향 없음)
 -> input 과 feature map의 크기를 동등하게 만들어 주기 위한 padding = same padding (합성곱 신경망에서 많이 사용)
- padding 없이 순수한 input에서만 합성곱을 해 특성맵을 만드는 경우 = valid padding (feature map 크기가 감소)

* 합성곱에서 padding을 사용하는 이유?
- input을 이미지로 생각하면 valid padding으로는 모서리의 정보가 feature map에 잘 전달되지 않을 수 있고, 중앙에 있는 정보는 두드러지게 표현될 수 있음 (filter에 찍히는 횟수 차이)
- padding을 통해 중앙부와 모서리 픽셀이 합성곱에 참여하는 비율을 크게 줄일 수 있음
- 일반적으로 합성곱 신경망에서는 same padding을 많이 쓰고, Conv2D 클래스의 padding 매개변수에서 패딩 지정 가능 -> padding = valid 가 default, padding = 'same' 이라면 same padding
    keras.layers.Conv2D(10, kernel_size = (3,3), activation = 'relu', padding = 'same')

* Stride
- 지금까지으 합성곱 신경망에서는 한칸씩 좌우, 위아래 이동을 했지만 이동 간격을 변경할 수 있음 -> 이동 간격 = Stride (기본적으로 1)
- strides = (x,y) 로 (오른쪽 이동, 아래 이동 크기)를 튜플로서 지정가능하지만 보통은 x=y 로 지정하고 strides = 1보다 크게 사용하는 경우도 드물다 --> strides 변수를 잘 사용하지 않음

*** 풀링(pooling) ***
- 만들어진 feature map의 크기를 줄이는 역할, 개수는 줄이지 않음 -> (2,2,3) 의 feature map에 pooling을 적용하면 (1,1,3)으로 만들 수 있음
- pooling 또한 합성곱처럼 input 위를 지나가면서 도장을 찍지만 여기에는 가중치가 없음, 도장 영역에서 가장 큰 값을 고르거나 평균값을 계산하는 방식, 각각 max pooling, average pooling 이라 함
- pooling은 합성곱 층과는 확실하게 분리되기 때문에 pooling layer 라고 부름
 (Ex) (4,4) feature map이 있을 때, (2,2)) max pooling을 적용한다면 절반으로 크기가 감소 -> (2,2)가 됨
- 합성곱 층과 pooling 층에서 나오는 모든 값을 feature map이라고 함
- 눈여겨볼 점은 pooling은 합성곱과는 다르게 커널이 한칸씩 이동하지 않고 겹치지 않게 두 칸씩 이동했다는 점 -> pooling 크기가 (2,2)라면 stride = 2, pooling 크기가 (3,3)이라면 stride = 3
    keras.layers.MaxPooling2D(2)
    - 매개변수 1 : pooling의 크기 (대부분 2, 가로세로 방향의 풀링 크기를 다르게 할 수 있지만 드뭄)
    - 매개변수 2 : strides = , 자동설정되기 때문에 따로 지정할 필요 없음
    - 매개변수 3 : padding = , default가 'valid'로 패딩을 하지 않음

    keras.layers.AveragePooling2D
    - 위의 max pooling과 매개변수 동일, 일반적으로 max pooling을 더 많이 사용 -> average pooling은 feature map에 있는 중요 정보를 평균해 희석시킬 수 있기 때문

*** 합성곱 신경망의 전체 구조 ***
합성곱 layer + filter + padding + stride + pooling 개념으로 전체 구조
- 합성곱 신경망은 밀집층을 이용한 신경망처럼 일렬로 늘어선 모양으로는 표현하기 어려움 -> 입체적
 (Ex) 전체 구조
 1) 합성곱 층
 - input : (4,4)
 - filter : (3,3) * 3 -> 각 filter마다 가중치는 서로 다름(절편 포함)
 - padding = 'same'
 - strides = 1 (default)
 - feature map = (4,4,3)

 2) 풀링 층 : 합성곱 층에서 만든 특성 맵의 가로세로 크기를 줄임 (보통 (2,2) 풀링)
 - pooling을 사용하는 이유는 합성곱 층에서 strides를 늘리는 것보다 pooling으로 크기를 줄이는 것이 더 좋은 결과를 가져오기 때문
 - pooling 결과로 (4,4,3) -> (2,2,3)

 3) 밀집층
 - 풀링 층을 통해 나온 (2,2,3) feature map을 밀집층에 전달하려면 이를 1차원배열로 펼쳐야 함, Flatten() 이용-> (12,)
 - 출력층에는 3개 뉴런을 두었으므로 이 문제는 3개의 클래스를 분류하는 다중 분류 문제, softmax activation 함수를 통해 최종 예측 확률이 됨

*** 컬러 이미지를 사용한 합성곱 ***
위의 예시에서는 입력을 2차원 배열로 가정했지만 이미지가 흑백이 아닌 컬러이미지인 경우 3차원 배열로 표현해야 한다
- (4,4)이라면 RGB를 고려해 (4,4,3)이 되는 형태
- 깊이가 존재하는 합성곱에서는 filter 또한 깊이를 가져야 함 -> filter의 kernel 크기가 (3,3,3) / input의 깊이와 커널 배열의 깊이는 항상 같음
- filter를 통해 계산하면 (3,3,3) 영역에 해당하는 27개 원소에 27개의 가중치를 곱하고 절편을 더하는 방식
- 중요한 것은 입력이나 필터의 차원에 관계없이 그 출력은 반드시 하나의 값이라는 점 -> feature map에는 한 원소가 채워짐
- keras의 합성곱 층은 항상 3차원 배열의 입력을 전제로 하고 있음, 때문에 2차원 배열 입력이 들어와도 (28,28,1)과 같이 3차원 배열로 변환한 후 계산
- 비슷한 경우로 합성곱 층 - pooling 층 이후로 또 합성곱 층이 오는 경우가 있음
 (Ex) 첫번쨰 합성곱 층의 filter 개수가 5개 -> pooling 층을 통과한 feature map의 크기가 (4,4,5) ? 이 경우의 input은? (same padding)
 - pooling의 경우 stride = 2이므로 pooling 층 이전에는 (8,8,5), same padding을 했으므로 input = (8,8,5)
 - 첫번째 합성곱 + pooling을 통과한 feature map이 (4,4,5) -> 두번쨰 합성곱 층의 filter 너비와 높이가 (3,3)이라면 이 filter의 크기는 (3,3,5) (filter 깊이와 input 깊이가 같아야 하기 때문)
   45(3*3*5)개의 입력에 가중치를 곱해서 1개의 출력을 생성하고 이를 feature map에 저장
 - 두번째 합성곱 층의 filter가 10개라면 만들어지는 feature map의 크기는 (2,2,10) --> 합성곱 신경망은 진행할 수록 너비와 높이가 감소하고 깊이가 깊어지는 것이 특징
 - 이를 마지막 출력층 전에 feature map을 모두 펼쳐 1차원 배열로 만든 후 밀집층의 input으로 사용

- 합성곱 신경망에서의 filter는 어떤 특징을 찾는다고 생각할 수 있음 / 처음에는 간단하고 기본적인 특징(직선, 곡선 등)을 찾고 depth가 깊어질수록 다양하고 구체적인 특징을 감지할 수 있도록 filter 수를 늘림
 + 어떤 특징이 어디에 위치하더라도 쉽게 감지할 수 있도록 너비와 높이 차원을 점차 줄여나감
"""

### 합성곱 신경망을 이용한 이미지 분류
"""
합성곱에 관한 기본적인 이론과 용어에 대해 학습했는 데, tensorflow를 이용하면 합성곱, 패딩, 풀링 크기를 직접 계산할 필요는 없음
"""
## 패션 MNIST 데이터 불러오기
"""
이전의 완전 연결 신경망과 동일하지만 합성곱 신경망은 1차원 배열로 펼쳐서 입력할 필요가 없기 때문에 해당 과정을 생략할 수 있음
다만, 입력 이미지는 항상 깊이 차원이 존재해야 하므로 흑백의 경우에는 채널 차원이 없는 2차원 배열이지만 Conv2D를 이용하기 위해 마지막에 이 채널 차원을 추가 -> reshpae() 이용
"""

# Data Import + preprocessing
from tensorflow import keras
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size = 0.2, random_state = 42)
# Conv2D의 input으로 사용하기 위해 2차원 배열 (28,28) 에서 3차원 배열 (28,28,1)로 변경

## 합성곱 신경망 만들기 -> 합성곱 층으로 이미지에서 특징을 감지한 후 밀집층으로 클래스에 따른 분류확률을 계산
# 첫번째 합성곱 층 Conv2D 추가
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size = 3, activation = 'relu', padding = 'same', input_shape = (28,28,1))) # 처음 매개변수인 32는 filter의 개수 / (3,3) filter + relu activation / same padding

# pooling 층 추가
model.add(keras.layers.MaxPooling2D(2)) # same padding 후 (2,2) pooling을 했으므로 feature map의 크기는 반으로 줄어듦 -> (14,14,32)

# 첫번쨰 합성곱 - 풀링 층 이후 두번째 합성곱 - 풀링 층 추가
model.add(keras.layers.Conv2D(64, kernel_size = 3, activation = 'relu', padding = 'same')) # filter개수만 64개로 증가
model.add(keras.layers.MaxPooling2D(2)) # same padding + (2,2) pooling이므로 반으로 감소 -> (7,7,64)

# 1차원 배열로 펼치기 -> 10개 뉴런을 가진 출력층에서 확률 계산
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation = 'relu')) # relu 함수를 activation function으로 활용
model.add(keras.layers.Dropout(0.4)) # dropout을 넣어 은닉층의 과대적합을 막고 성능 개선
model.add(keras.layers.Dense(10, activation = 'softmax')) # 다중 분류 문제이므로 softmax를 사용

# 모델 구조 출력
model.summary()
"""
Model: "sequential_13"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 32)        320 # (3,3) * 1 depth인 filter 32개에 filter마다 절편 하나씩 = 3*3*1*32 + 32 = 320       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 14, 14, 32)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 14, 14, 64)        18496     # (3,3) * 32 인 filter 64 + 절편 하나씩 = 3*3*32*64 + 64 = 18,496
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 7, 7, 64)         0         
 2D)                                                             
                                                                 
 flatten_13 (Flatten)        (None, 3136)              0         
                                                                 
 dense_26 (Dense)            (None, 100)               313700    # (7,7,64)를 1차원 배열 = 7*7*64 = (3136,) 이를 100개 뉴런과 연결하고 절편도 있으므로 (3136+1) * 100 = 313,700
                                                                 
 dropout_9 (Dropout)         (None, 100)               0         
                                                                 
 dense_27 (Dense)            (None, 10)                1010    # 10개의 뉴런에 대한 parameter = 1010개
                                                                 
=================================================================
Total params: 333,526
Trainable params: 333,526
Non-trainable params: 0
_________________________________________________________________
"""
keras.utils.plot_model(model, to_file = 'model.png',show_shapes = True) # show_shapes = True 라면 입력과 출력 크기를 표시해줌

## 모델 컴파일 + fitting
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True)
history = model.fit(train_scaled, train_target, epochs = 20, validation_data = (val_scaled, val_target), callbacks = [checkpoint_cb, early_stopping_cb])
print(early_stopping_cb.stopped_epoch) # 9 -> 7 epoch가 최적

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show(block=True)

# 성능 평가 -> EarlyStopping 클래스에서 restore_best_weights 매개변수 = True였으므로 model객체는 최적 파라미터로 복원되어 있음
model.evaluate(val_scaled, val_target) # [0.2211090326309204, 0.922249972820282] -> 성능이 가장 잘 나옴

# 훈련된 모델을 이용한 새로운 데이터에 대한 예측
plt.imshow(val_scaled[0].reshape(28,28), cmap = 'gray_r')
plt.show(block=True)

preds = model.predict(val_scaled[0:1]) # 슬라이싱을 이용해서 (28,28)이 아닌 (28,28,1)의 3차원 배열로 입력
print(preds) # [[1.0249173e-16 1.4278137e-29 2.1198979e-18 2.5219081e-20 2.4499685e-17 3.7072903e-18 9.0936789e-17 2.2627559e-19 1.0000000e+00 3.8139059e-22]] -> 9번쨰 값이 1이고 나머지는 거의 0에 가까움 = 9번째 클래스

plt.bar(range(1,11),preds[0])
plt.xlabel('class')
plt.ylabel('prob.')
plt.show(block=True)

classes = ['티셔츠', '바지', '스웨터','드레스','코트','샌달','셔츠','스니커즈','가방','앵클 부츠']

import numpy as np
print(classes[np.argmax(preds)])

test_scaled = test_input.reshape(-1,28,28,1) / 255.0
model.evaluate(test_scaled, test_target) #  [0.2408510446548462, 0.91839998960495] -> 약 92%의 성능을 기대할 수 있음 / 검증세트보다는 좀 더 작게 나옴

### 합성곱 신경망의 시각화 ###
"""
합성곱 신경망은 이미지에 있는 특징을 찾아 압축하는 데 뛰어난 성능 
이번 챕터에서 합성곱 층이 이미지에서 어떤 것을 학습했는지 알아보기 위해 합성곱 층의 가중치 +  feature map 을 그림으로 시각화

이전까지는 model 생성에 있어 keras.Sequential()을 사용했지만 좀 더 복잡한 모델을 만들 수 있는 함수형 API 또한 keras 에서 제공
"""

## 가중치 시각화
"""
합성곱 층은 여러 개 필터를 사용해 이미지에서 특징을 학습 -> 각 필터는 커널이라고 하는 가중치 + 절편을 가짐 / 절편은 시각적으로 의미가 있지는 앟고 가중치는 어떤 특징을 두드러지게 표현하는 역할
"""

 from tensorflow import keras
 model = keras.models.load_model('best-cnn-model.h5')

# model 의 layer를 알아보기
model.layers # List의 형태로 출력
"""
[<keras.layers.convolutional.conv2d.Conv2D at 0x17de4751d30>, # 합성곱 
 <keras.layers.pooling.max_pooling2d.MaxPooling2D at 0x17de4751d90>, # pooling
 <keras.layers.convolutional.conv2d.Conv2D at 0x17de4785310>, # 합성곱2
 <keras.layers.pooling.max_pooling2d.MaxPooling2D at 0x17de47853d0>, # pooling
 <keras.layers.reshaping.flatten.Flatten at 0x17de47a0970>, # 밀집층에 넣기 위해 Flatten으로 1차원 배열화
 <keras.layers.core.dense.Dense at 0x17de47a8c70>, # 밀집층
 <keras.layers.regularization.dropout.Dropout at 0x17de47a8af0>, # dropout -> 과적합 감소
 <keras.layers.core.dense.Dense at 0x17de47bca60>] # 출력층
"""

# 첫번째 합성곱 층의 weight (층의 가중치와 절편)
conv = model.layers[0] # 첫번째 합성곱 층 선택
print(conv.weights[0].shape, conv.weights[1].shape) # 선택한 합성곱 층의 weight 중 [0] = 가중치, [1] = 절편

"""
model 생성 시 첫번째 합성곱 층의 kernel size = (3.3) 이고 이 층에 전달되는 input의 depth가 1이므로 실제 커널 크기가 (3,3,1)이 됨 + filter 개수가 32개 --> (3,3,1,32)
절편은 각 filter마다 하나씩 존재하므로 (32,)
"""

# 다루기 용이하도록 numpy 배열로 변환 -> weights 속성은 tensorflow의 Tensor 클래스 객체
conv_weights = conv.weights[0].numpy()
print(conv_weights.mean(), conv_weights.std()) # -0.022935035 0.26352292

# 가중치들을 훈련하기 전의 가중치와 비교하기 위해 Histogram 작성
import matplotlib.pyplot as plt
plt.hist(conv_weights.reshape(-1,1)) # histogram을 그르기 위해서는 1차원 배열로 전달
plt.xlabel('weight')
plt.ylabel('count')
plt.show(block=True)

# 32개의 kernel을 16 * 2로 출력
fig, axs = plt.subplots(2,16,figsize = (15,2))
for i in range(2) :
 for j in range(16) :
  axs[i,j].imshow(conv_weights[:, :, 0, i*16+j], vmin=-0.5, vmax = 0.5) # imshow는 배열에 있는 최댓값과 최솟값을 활용해 픽셀 강도를 표현하기에 갑이 낮아도 해당 배열의 최댓값이면 밝은 색이 출력됨
  # 이를 방지하기 위해서 vmin과 vmax로 범위 지정
  axs[i, j].axis('off')
plt.show(block=True)

# 훈련하기 전의 빈 합성곱 신경망
no_training_model = keras.Sequential()
no_training_model.add(keras.layers.Conv2D(32, kernel_size = 3, activation = 'relu', padding = 'same', input_shape = (28,28,1)))
no_training_conv = no_training_model.layers[0]
print(no_training_conv.weights[0].shape) # (3, 3, 1, 32)

no_training_weights = no_training_conv.weights[0].numpy()
print(no_training_weights.mean(), no_training_weights.std()) # -0.004588199 0.084816985 / 평균은 훈련 모델과 유사하지만 표준편차가 매우 작아짐

plt.hist(no_training_weights.reshape(-1,1)) # histogram을 그르기 위해서는 1차원 배열로 전달
plt.xlabel('weight')
plt.ylabel('count')
plt.show(block=True) # 가중치가 대부분 -0.15 ~ 0.15 사이에 존재하고 고른 분포를 보임 -> tensorflow가 신경망 가중치 초기화 시 uniform 분포에서 랜덤하게 값을 선택하기 때문

fig, axs = plt.subplots(2,16,figsize = (15,2))
for i in range(2) :
 for j in range(16) :
  axs[i,j].imshow(no_training_weights[:, :, 0, i*16+j], vmin=-0.5, vmax = 0.5) # imshow는 배열에 있는 최댓값과 최솟값을 활용해 픽셀 강도를 표현하기에 갑이 낮아도 해당 배열의 최댓값이면 밝은 색이 출력됨
  # 이를 방지하기 위해서 vmin과 vmax로 범위 지정
  axs[i, j].axis('off')
plt.show(block=True) # 가중치가 fitting model에 비해 밋밋하게 초기화됨 -> 합성곱 신경망이 fitting을 통해 data 분류 정확도를 높일 수 있는 유용한 패턴을 학습했다는 것을 알 수 있음

## 함수형 API
"""
이전까지는 model 생성을 위해 keras.Sequential()을 사용헀지만 이 class는 layer를 차례대로 쌓은 모델을 만듦

"""