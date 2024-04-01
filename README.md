# CNN
> 이 자료는 MIT 강의 영상을 기반으로 작성되었으며, 공부 기록을 목적으로 두고 있습니다.
> > 강의 주소: <http://introtodeeplearning.com/>
## 1. Introduction
> * Deep learning과 Machine learning을 사용하여 시각적 정보만 보고 어디에 무엇이 있는 지 확인하고 예측할 수 있는 Vision System을 구축하는 것이 목표입니다.
> * 어디에 무엇이 있는 지 이해하는 것을 넘어 이미지의 여러 단서를 통해 추론하여 미래를 예측하는 것이 핵심입니다.
## 2. What computers "see"
> * 컴퓨터에 입력된 사진은 "<b>Pixel</b>"로 이루어져 있으며, Pixel은 컬러를 구성하는 최소 단위입니다.
![스크린샷 2024-03-24 231155](https://github.com/SangHyeokNam/CNN/assets/149642144/b17c29eb-9757-439f-b3b4-385484b462bb)
> * 위 아브라함 링컨의 사진은 Grayscale로 구성된 이미지이므로 단일 숫자를 가지며, 이미지는 2차원 Matrix로 표현할 수 있습니다. 이 2차원 Matrix를 통해 무엇을 할 수 있는 지, 어떤 유형의 Vision 알고리즘을 구축하여 사용할 수 있는 지 등에 대해 배우는 것이 중요합니다.
> * 먼저 알아야 할 두 가지 개념이 있습니다. 바로 <b>분류</b>와 <b>회귀</b>입니다.
> * 회귀 모델은 예측값으로 연속적인 값을 출력하고, 분류 모델은 예측값으로 이산적인 값을 출력합니다. 이산적인 값은 연속적인 값과 반대되는 뜻으로, 연속되지 않고 단절된 값을 의미합니다.

![스크린샷 2024-03-25 033619](https://github.com/SangHyeokNam/CNN/assets/149642144/24d3e7a8-4a12-40ae-9a85-8b5d2bb3a314)
> * Computer Vision Pipeline을 구축하는 경우 두 가지 주요 단계를 거쳐야 하는데, 첫 번째는 데이터에서 특징이나 패턴을 찾아야하고, 두 번째는 해당 패턴이 어떤 Class에 있는 지 추론하는 것입니다.
> * 이 문제를 해결하는 한 가지 방법은 특정 분야에 대한 지식을 활용하는 것입니다. 해당 분야의 지식을 활용해 구성하는 특징을 정의할 수 있습니다. 하지만 이미지는 3차원 배열일 뿐이므로 특징을 정의할 때 동일 유형의 객체를 밝기 조절, 회전 등 변형을 통해 정의합니다.
> * 따라서 Computer Vision 알고리즘을 구축하는 방법은 데이터의 패턴에 맞는 특징을 감지하고 추출하는 것입니다.
## 3. Learning visual features
> * 완전 연결 네트워크(<b>Fully Connected Neural Network</b>, 1차원 배열의 형태로 평탄화된 행렬을 통해 이미지를 분류하는 데 사용되는 네트워크.)를 사용한다고 가정하겠습니다. (평탄화 : 이 전 Layer의 출력을 "평탄화"하여 다음 Layer의 입력이 될 수 있는 단일 벡터로 변환.)
> * 2차원 배열의 이미지를 1차원 배열로 축소하고, 입력 Layer부터 다음 Layer의 모든 뉴런과 연결되며, 모든 픽셀을 공급합니다. 뉴런이 완전히 연결되어 있기 때문에 엄청난 수의 매개변수를 갖게 됩니다. 따라서 Overfitting 발생 가능성이 높아집니다.

![스크린샷 2024-03-28 022506](https://github.com/SangHyeokNam/CNN/assets/149642144/d036fedb-144c-497d-9a09-ea1f5e780a11)
> * 실제 처리해야 하는 이미지의 크기는 이러한 방식이 불가능하기 때문에 2D 이미지를 2차원 숫자 배열로 표현하는 것이 중요합니다. 이 전 Layer의 일부 공간이 다음 Layer의 뉴런 하나에 입력되며, 또 다른 일부 공간이 두 번째 뉴런이 되는 방식으로, 전체 이미지에 걸쳐 픽셀 단위로 적용할 수 있습니다. 이러한 방식으로 입력에 내재된 핵심적인 공간 정보를 모두 보존하며, 패턴을 학습하여 이미지를 감지하고 분류할 수 있습니다.
## 4. Feature Extraction and Convolution
> * 예를 들어 'X' 이미지를 감지하거나 분류하는 <b>Convolution Algorithm</b>을 구축한다고 가정해보겠습니다. 다음 이미지는 모든 픽셀이 0 또는 1로 표현된다고 가정하며, 예시를 들기 위해 단순하게 표현된 것입니다.

![스크린샷 2024-03-28 030245](https://github.com/SangHyeokNam/CNN/assets/149642144/a6435ef9-3496-4739-8c0d-bc60507719d9)
> * 위 이미지에서 좌,우의 'X'는 같은 문자이지만 우측은 변형(회전)이 된 이미지입니다. 우리는 두 이미지 모두 감지하는 것이 목표입니다.
> * 모델이 동일한 위치에서 X를 정의하는 기능의 패치(Kernel에 대한 입력)를 찾을 수 있다면 이미지를 감지할 수 있습니다. 이는 단순히 두 이미지 간의 유사성을 측정하는 것보다 훨씬 나은 결과를 얻을 수 있습니다.

![스크린샷 2024-03-28 040112](https://github.com/SangHyeokNam/CNN/assets/149642144/d2cc4bd3-1700-4eb0-906c-3add5bfc78fb)
> * 필터를 사용하여 찾고자하는 패치를 포착할 수 있으며, 위 3가지 필터는 X의 경우 패치를 찾는데 적합한 필터입니다. 따라서 위 필터를 통해 X 이미지가 변환이 되어도 특징을 찾을 수 있습니다.

![스크린샷 2024-03-28 041316](https://github.com/SangHyeokNam/CNN/assets/149642144/a2b2f5fb-856d-4176-840b-8772e143e689)
> * 패치를 찾으면 필터와 요소별 곱셈을 수행한 후, 나온 행렬의 요소들을 모두 더합니다. 더하여 나온 값은 <b>feature map</b>의 요소가 되며, 필터를 슬라이딩하여 적용하면 원래 이미지의 정보를 보존한 새로운 feature map이 만들어집니다.
## 5. Convolution neural networks
CNN의 단계별 세 가지 주요 작업
![스크린샷 2024-03-28 044905](https://github.com/SangHyeokNam/CNN/assets/149642144/8a7b0169-f599-4c88-8638-8d285e2ae4ef)
> * 첫 번째 단계는 <b>Convolution</b>입니다. Convolution을 사용하여 이전 이미지와 필터를 모두 입력받으며, feature map을 출력합니다.
> * 두 번째 단계는 <b>Non-linearity</b>입니다. 활성화 함수를 적용하여 비선형 데이터를 처리합니다.
> * 세 번째 단계는 <b>Pooling</b>입니다. 필요한 data를 제외한 나머지 parameter를 줄입니다. Pooling에 대해서는 다시 다루겠습니다.
> * 결과적으로 나온 모든 기능을 일부 신경망에 적용하여 규모가 큰 이미지를 처리할 수 있습니다.
> * 요소별 곱셈과 덧셈을 하고, 슬라이딩하며 연산하는 것이 Convolution layer의 기초입니다.
## 6. Non-linearity and pooling
> * 이미지 데이터는 비선형적이고, 비선형 데이터를 처리하기 위해 비선형성을 적용하는 것이 중요합니다.
> * Convolution layer의 경우 일반적으로 사용되는 활성화 함수는 <b>Relu</b> 함수입니다. Sigmoid 함수가 아닌 Relu 함수를 사용하는 이유는 역전파(Backpropagation)과정에서 출력층에서 멀어질수록 Gradient 값이 매우 작아지는 <b>Vanishing Gradient</b>(기울기 소실)현상이 발생하기 때문입니다.

![스크린샷 2024-03-29 014704](https://github.com/SangHyeokNam/CNN/assets/149642144/0d4b6adb-cda9-47fc-b73c-0e6a55747d8e)
> * Sigmoid 함수의 미분 값은 입력값이 0일 때 가장 크지만 0.25에 불과하며, x값이 점점 커지거나 작아짐에 따라 0에 수렴해가는 것을 볼 수 있습니다. 따라서 출력층과 멀어질수록 Gradient값이 매우 작아집니다.
> * tanh 함수 역시 sigmoid 함수의 한계점을 개선하기 위한 방안으로 제안되어 출력값의 범위가 2배이지만, 마찬가지로 기울기 소실을 해결하기에는 어려움이 있습니다.

![스크린샷 2024-03-29 015515](https://github.com/SangHyeokNam/CNN/assets/149642144/fdbe7fce-7a92-4c0f-a56f-d58d386fd6e4)
> * 이를 해결하기 위해 사용하는 함수가 <b>Relu</b> 함수입니다. Relu 함수는 입력값이 양수일 경우, 입력값에 상관없이 미분값은 항상 1이기 때문에 역전파 과정의 기울기 소실 문제를 해결할 수 있습니다.
> * 하지만 Relu 함수는 입력값이 음수일 경우 미분값이 항상 0이기에 음수인 뉴런은 회생시키기 어렵다는 단점이 있습니다.
> * 이런 현상을 해결하기 위해 음수일 경우 작은 값이라도 갖는 <b>Leaky Relu</b> 함수가 제안되었습니다.

![스크린샷 2024-03-29 021503](https://github.com/SangHyeokNam/CNN/assets/149642144/92b8c743-6591-421e-9fd2-cfa334d171c4)
> * CNN의 다음 핵심 작업은 <b>Pooling</b>입니다. Convolution layer를 통해 점점 더 깊어짐에 따라 이미지의 차원을 줄이는 것입니다. 모든 데이터를 사용하지 않고, 메모리를 줄여 계산효율성이 좋아지기 때문에 사욯합니다.
> * 일반적으로 <b>Max Pooling</b>(최대 풀링)과 <b>Average Pooling</b>(평균 풀링)이 사용됩니다.
> * <b>Max Pooling</b>은 해당 패치 위치에서 최대값만을 취하여 다운샘플링하고, <b>Average Pooling</b>은 최대값 대신 평균값을 취합니다.
> * Convolution layer 연산과 유사하지만 학습해야 할 가중치가 없으며, 연산 후에 채널 수가 변하지 않습니다.
## 7. Flatten and Padding
> * <b>Flatten</b>은 입력 데이터를 1차원으로 평탄화(flatten)합니다. 2D 혹은 3D의 특징 맵(feature map)을 1D 벡터로 변환하여, 이후의 레이어에서 처리하기 쉽게 만들어주는 역할을 합니다.
> * <b>Padding</b>은 합성곱 연산 전에 입력 데이터 주변에 특정 값을 채우는 것을 말합니다. 1폭짜리 패딩이란 데이터 사방에 1픽셀을 특정 값으로 채운다는 의미입니다. 이미지의 축소를 방지하고, 특히 윤곽자리의 이미지 정보를 보존하기 위해서 사용합니다.
## 8. code example
![스크린샷 2024-03-29 023354](https://github.com/SangHyeokNam/CNN/assets/149642144/1b72efcd-21d7-4179-a24c-07ecdd97910f)
> * 여기서 필터 수가 증가하는 이유는 상세한 패턴은 오히려 Noise나 Overfitting으로 적절하게 동작하기 어렵기 때문에 상세 정보는 손실되지만 보다 일반적인 패턴을 잡아내기 위해 필터 수를 증가시킵니다.
## 9. Object detection
> * 실제로는 이미지를 감지하고 분류하여 답을 내는 것보다 더 많은 것을 할 수 있습니다. 이미지 안에서 서로 다른 각각의 물체에 <b>bounding box</b>(경계 상자)를 그려낼 수도 있습니다. 또 한 이미지에 하나의 객체가 아닌 여러 객체가 있을 수 있기 때문에 동적 개체 수를 추론할 수 있어야 하며 bounding box를 각각 독립적으로 연결할 수 있는 모델이 필요합니다.
> * bounding box는 이미지의 어느 곳이든 존재할 수 있으며, 크기가 다를 수도 있고 비율도 다를 수 있습니다.

> * 예를 들어 임의의 객체에 bounding box를 배치하고, 분류를 수행하여 무엇인지 감지할 때 이미지의 모든 무작위 bounding box에 대하여 이 프로세스를 반복합니다.
> * 여기서 문제는 무작위 객체가 너무 많다는 것입니다. 이를 보완하기 위해 Selective search 알고리즘을 이용해 이미지에서 객체가 있을 것 같은 위치에 bounding box를 생성하는 R-CNN이 있습니다.
> * Selective search 알고리즘은 물체가 있을만한 영역을 모두 조사해보는 Exhaustive Search 방법에 Segmetation을 결합해 개선한 알고리즘으로, 객체가 있을만한 후보 영역을 미리 찾고 그 영역 내에서만 객체를 찾는 Region Proposal 방식의 방법 중 하나입니다.
![스크린샷 2024-04-02 030140](https://github.com/SangHyeokNam/CNN/assets/149642144/23928d11-4744-470c-8d5c-d19c796758dd)
> * R-CNN은 Region proposal로 추출한 수많은 개수의 영역을 모두 CNN에 통과시키기 때문에 오래걸리며, 객체의 비율을 고려하지 않고 모두 같은 크기로 resize하여 정보를 손실할 수 있다는 단점이 있습니다.

> * Fast R-CNN은 전체 이미지에 대해 CNN을 한번 거친 후 출력 된 Feature map에서 객체 탐지를 수행하여 R-CNN의 단점을 보완했으며, Fast R-CNN 역시 CNN Network가 아닌 Selective search 외부 알고리즘을 사용하여 병목현상이 발생하는 단점이 있습니다.

> * 따라서 selective search 알고리즘을 사용하지 않고, Region Proposal Network(RPN)를 통해서 RoI(Region of Interest)를 계산하여 Fast R-CNN의 단점을 개선하는 Faster R-CNN이 요점입니다.
![스크린샷 2024-04-02 041437](https://github.com/SangHyeokNam/CNN/assets/149642144/80793c49-fc3d-4f70-93a1-256c07b00231)
> * 네트워크 구조
> > 1) 원본 이미지를 pre-trained된 CNN 모델에 입력하여 feature map이 추출됩니다.
> > 2) feature map은 RPN에 전달되어 적절한 region proposals을 산출합니다.
> > 3) Region proposals와 1) 과정에서 얻은 feature map을 통해 RoI pooling을 수행하여 고정된 크기의 feature map을 얻습니다.
> > 4) Fast R-CNN 모델에 고정된 크기의 feature map을 입력하여 Classification(분할)과 Bounding box regression(회귀)을 수행합니다. 

> * RPN은 feature map을 input으로, RP를 output으로 하는 네트워크라고 할 수 있고, selective search의 역할을 온전히 대체합니다.
> * RoI Pooling은 feature map의 proposal region에서 미리 정해놓은 크기(FC layer의 input 사이즈)의 격자(grid)에 맞추어 maxpooling 하여 고정된 크기의 vector를 만들어냅니다. proposal region은 사이즈가 제각각이기 때문에 고정된 사이즈로 만들어주기 위해 사용합니다.
## 10. Semantic segmentation
> * 또 한 Object detection과 함께 자율주행, 의료 등에서 가장 많이 활용되고 있는 Semantic segmentation(의미적 분할)이 있습니다.
> * Object detection이 이미지 내 특정 영역에 대한 분류 결과를 보여준다면, Semantic segmentation은 이미지 내 모든 픽셀에 대한 분류 결과를 보여줍니다. 따라서 이미지의 각 부분이 어떤 의미를 가지고 있는 지 구분할 수 있게 합니다.
> * Semantic segmentation은 같은 Class에 속하는 Object를 따로 분류하지 않으며 픽셀이 어떤 클래스인 지만 구분합니다.
> * Semantic segmentation을 위한 학습데이터 만드는 과정
> > 1) Semantic segmentation의 Class(의미 종류)를 설정합니다.
> > 2) 원본 이미지를 미리 정의해둔 Class에 따라 픽셀 단위로 구분합니다.
> > 3) Class에 따라 픽셀의 RGB 값이 변경된 가공 이미지를 생성합니다.
> > 4) Class와 RGB 값의 매핑 정보를 생성합니다.
## 11. Summary
> * 지금까지 기능 추출 및 감지의 핵심 개념과 CNN architecture(구조), Object detection을 배웠습니다.
