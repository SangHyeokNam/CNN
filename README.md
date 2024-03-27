# CNN
> 이 자료는 MIT 강의 영상을 기반으로 작성되었으며, 공부 기록을 목적으로 두고 있습니다.
> > 강의 주소: <http://introtodeeplearning.com/>
## 1. Introduction
> * Deep learning과 Machine learning을 사용하여 시각적 정보만 보고 어디에 무엇이 있는 지 확인하고 예측할 수 있는 Vision System 구축하는 것이 목표입니다.
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
> * 따라서 Computer Vision 알고리즘을 구축하고 유지하는 방법은 데이터의 패턴에 맞는 특징을 감지하고 추출하는 것입니다.
## 3. Learning visual features
> * 완전 연결 네트워크(Fully Connected Neural Network, 1차원 배열의 형태로 평탄화된 행렬을 통해 이미지를 분류하는 데 사용되는 네트워크.)를 사용한다고 가정하겠습니다. (평탄화 : 이 전 Layer의 출력을 "평탄화"하여 다음 Layer의 입력이 될 수 있는 단일 벡터로 변환.)
> * 2차원 배열의 이미지를 1차원 배열로 축소하고, 입력 Layer부터 다음 Layer의 모든 뉴런과 연결되며, 모든 픽셀을 공급합니다. 뉴런이 완전히 연결되어 있기 때문에 엄청난 수의 매개변수를 갖게 됩니다. 따라서 Overfitting 발생 가능성이 높아집니다.

> * 실제 처리해야 하는 이미지의 크기는 이러한 방식이 불가능하기 때문에 2D 이미지를 2차원 숫자 배열로 표현하는 것이 중요합니다. 이 전 Layer의 일부 공간이 다음 Layer의 뉴런 하나에 입력되며, 또 다른 일부 공간이 두 번째 뉴런이 되는 방식으로, 전체 이미지에 걸쳐 픽셀 단위로 적용할 수 있습니다.
