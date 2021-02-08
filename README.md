## 상관계수 정규화와 동적 필터 가지치기를 이용한 심층 신경망 압축
Dynamic Filter Pruning with Decorrelation Regularization for Compression of Deep Neural Network
> 2020 한국소프트웨어종합학술대회 (KSC2020) 학부생논문 경진대회 장려상 수상작

## Prerequisites

* Ubuntu 18.04
* Python 3.7.4
* Pytorch 1.6.0
* numpy 1.18.1
* GPU (cuda)

## Build

```
$ python modeling.py --prune_type structured --prune_rate 0.6 --reg reg_cov --odecay 0.7 > result.txt
```
* `run.sh`에서 parameter 조절 후 `./run.sh`로 진행

## Process
### 0. Data, Model & Parameters
- Data : CIFAR-10
- Model : ResNet-50
- Optimizer : Stochastic Gradient Descent
- Learning Rate : 0.2
- Epoch : 300
- Batch size : 128
- Loss Function : Cross Entropy
- Metric : Accuracy, Sparsity

### 1. 동적 필터 가지치기 (Dynamic Filter Pruning)
L1 norm 크기를 기반으로 필터 마스크를 생성하여 가중치 학습 시 반영
- 필터 마스크 : ![image](https://user-images.githubusercontent.com/41580746/102396051-41616d80-401f-11eb-9738-7b5df9aee0d4.png)
   - i : 층 위치
   - j : 필터 위치
   - t : epoch 값
   - W : 필터 가중치 행렬
   - η : 임계값 (전체 필터 개수 중 가지치기 필터 개수 비율 통해 계산)
- 가중치 학습 : ![image](https://user-images.githubusercontent.com/41580746/102396101-51794d00-401f-11eb-9303-99dd712798ee.png)
   - g : 기울기
   - γ : learning rate

### 2. 상관계수 정규화 (Decorrleation Regularization)
기존 loss function에 상관계수 정규화 식을 더하여 최종 손실 함수 계산
- loss function : ![image](https://user-images.githubusercontent.com/41580746/102396482-d82e2a00-401f-11eb-93b3-7db5fea8f8af.png)
   - α : 정규화 상수
   - ![image](https://user-images.githubusercontent.com/41580746/102396603-01e75100-4020-11eb-933e-f85305dd874d.png)

## Result
가지치기 비율 60%, 정규화 상수 0.7일 때의 모델별 Accuracy 및 Sparsity 비교 결과
- ![image](https://user-images.githubusercontent.com/41580746/102395542-a072b280-401e-11eb-9e47-c3b52d859479.png)
- ![image](https://user-images.githubusercontent.com/41580746/102396706-28a58780-4020-11eb-9ebb-6cc723b4fcbf.png)
- 기존 동적 필터 가지치기 대비 Accuracy 1.47%, Sparsity 1.08% 증가

---

_References_
- [1] Yann LeCun, Yoshua Bengio, Geoffrey Hinton. Deep learning. Nature 521, 436-444, 2015.
- [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jain Sun. Deep Residual Learning for Image Recognition. 2015.
- [3] 조인천, 배성호. 동적 필터 프루닝 기법을 이용한 심층 신경망 압축. 한국방송미디어공학회 하계학술대회, 2020.
- [4] Benoit Jacob. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. 2017.
- [5] Tao Lin, Sebastian U. Stich, Luis Barba, Daniil Dmitriev, Martin Jaggi. Dynamic Model Pruning with Feedback. ICLR, 2020.
- [6] Namhoon Lee, Thalaiyasingam Ajanthan, Philip HS Torr, SNIP: Single-shot network pruningbased on connection sensitivity. ICLR, 2019.
- [7] Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, Hans Peter Graf. Pruning Filters For Effiecient ConvNets. ICLR, 2017.
- [8] Jian-Hao Luo, Jianxin Wu, Weiyao Lin. ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression. ICCV, 2017.
- [9] Song Han, Huizi Mao, William J. Dally. Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding. ICLR, 2016.
- [10] Xiaotian Zhu, Wengang Zhou, Houqiang Li. Improving Deep Neural Network Sparsity through Decorrelation Regularization. IJCAI, 2018.
