# Apartment-auction-price-prediction

### 교내 경진 대회 참여
 * Dacon에서 예전에 진행했던 아파트 경매 가격 예측을 주제로 교내 AI 경진대회가 진행되었다.

### Dacon 아파트 경매 가격 예측
* [Dacon 아파트 경매 가격 예측 링크](https://dacon.io/competitions/official/17801/data)  

### 데이터 설명
* 한국의 서울과 부산 지역 약 2,700여개 최근 2년간 아파트 경매물의 등기부, 임차, 감정가, 유찰 횟수, 낙찰가 등의 정보가 제공됩니다. 아파트 낙찰가를 예측해야 합니다. 
* [국토교통부 실거래가공개시스템](http://rt.molit.go.kr) 등 법적인 제약이 없는 외부 데이터(공공 데이터) 사용이 가능합니다.
(2018년 기준)

### 제공 파일
1. train.csv – 서울/부산 지역의 낙찰가를 포함하여 경매 물건 아파트의 위치, 감정가, 경매 개시/종결일 등의 기본 정보(*최근2년)
2. test.csv – 경매 낙찰가를 제외하고 train.csv와 동일 
3. sample_submission.csv – 예측한 낙찰가를 기입하여 제출  


### 목표

* 기존 대회가 있었던 만큼 기존 대회에서 사용했던 방식을 적용할 수 있는 한 다양하게 적용하는 것을 목표로 삼았다.
* 모델 생성 또한 사용할 수 있는 다양한 모델을 사용하는 것을 목표로 삼았다.

### 진행 과정

* AI에 관한 기초지식이 없었기 때문에 기초 지식을 쌓은 후 유형이 비슷한 다른 예측 모델을 참고하며 제작하는 것을 방향으로 삼았다.
* 모델에 및 하이퍼 파라미터 수정에 따른 오차율 차이는 유의미한 결과가 없다 판단하여 기존 변수를 활용해 다양한 파생 변수들을 생성하는 것에 중점을 두었다.
* Public에서 좋은 결과를 내더라도 Private에서 동일하게 좋은 결과를 낼 것이라는 보장이 없었기 때문에 변수 선정 시 주의를 기울였으며 관련 논문을 참고했다.

1. AI 기초 학습
    1-1. 생활 코딩 - [머신 러닝](https://opentutorials.org/course/4548), [Tensorflow](https://opentutorials.org/course/4570)
![슬라이드10](https://user-images.githubusercontent.com/78258412/149445451-d1be996f-717d-4994-bd87-918a41330370.JPG)

2. [Kaggle 노트북 분석](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)
![슬라이드11](https://user-images.githubusercontent.com/78258412/149445454-f089c22e-5571-4e68-912b-8af058ccc1cf.JPG)

3. 참고 논문
    3-1. 머신러닝 기반의 부동산경매 낙찰가 예측 모델에 관한 연구
![슬라이드12](https://user-images.githubusercontent.com/78258412/149445457-44350b33-bfc1-4a2e-812c-f9fb84a6dcc5.JPG)
    3-2 거주층이 아파트 가격에 미치는 영향에 관한 연구 : 공간, 시간 모형 분석
![슬라이드13](https://user-images.githubusercontent.com/78258412/149445459-7f9fe209-fa6d-4c5b-b5be-ce44bea50b09.JPG)


### 결과

1.  Minimum_sales_price는 예측하고자 하는 값인 Hammer_price와 매우 큰 상관관계를 가지고 있었기 때문에 직접 반영하는 것 대신 아래와 같은 형태로 바꾸었다.
``` python
train['Temporary_price'] = train['Hammer_price'] / train['Minimum_sales_price']
```
2.  scatter_print 함수를 여러 변수에 적용 시킨 결과 한 개의 이상치(극단값, Outlier)이 존재함을 알 수 있었다.
![슬라이드4](https://user-images.githubusercontent.com/78258412/149445440-284168de-9076-44e7-85bb-a7b54af0d132.JPG)
``` python
def scatter_print(features):
  var = features
  var_data = pd.concat([train['Hammer_price'], train[var]], axis=1)
  var_data.plot.scatter(x=var, y='Hammer_price', ylim=(0, 151.51))
  plt.show() 
  
train = train[train.Hammer_price != max(train['Hammer_price'])] # 이상치 제거
```
3. 결측치를 제거하는 것과 제거하지 않고 평균값 등으로 처리한 후 학습 결과 비교시 제거하는 것의 오차율이 더 적었음을 알 수 있었다.
4.  거주층이 아파트 가격에 미치는 영향에 관한 연구 : 공간, 시간 모형 분석 논문을 참고해 중간 층수의 가격이 높지만 아파트 전체 층수에 따라 조금씩 바뀌는 것을 생각해 적용했다. 중간 층수 1을 기준으로 한 층씩 떨어질 때 마다 값이 감소하도록 Loyal_floor 변수를 생성했다.
```
#### 2020-11-27 중간층수 가격 높음
## http://www.kahps.org/data/_research/201905/15593089066584.pdf 참고 논문
## 13층 이상일 경우 7-9 층이 가격이 높은 것으로 나옴 => 8층
## 7층 - 12층 4-6층 => 5층
## 6층 이하 Top - 1
train['Loyal_floor'] = 0.0
for i in range(train.shape[0]):
  if abs((train['Total_floor'][i] / 2) - train['Current_floor'][i]) == 0:
    train['Loyal_floor'][i] = 1
  else: train['Loyal_floor'][i] = ((train['Total_floor'][i] / 2) - abs((train['Total_floor'][i] / 2) - train['Current_floor'][i])) / (train['Total_floor'][i] / 2) 

test['Loyal_floor'] = 0.0
for i in range(test.shape[0]):
  if abs((test['Total_floor'][i] / 2) - test['Current_floor'][i]) == 0:
    test['Loyal_floor'][i] = 1
  else: test['Loyal_floor'][i] = ((test['Total_floor'][i] / 2) - abs((test['Total_floor'][i] / 2) - test['Current_floor'][i])) / (test['Total_floor'][i] / 2) 

```
5. 다양한 파생 변수들을 생성했으며 반영에는 rmse 점수와 부동산 관련 책 및 논문을 참고했다. (노트북 코드 참고)
 
## AI 경진대회 발표자료
<img width="80%" src="https://user-images.githubusercontent.com/78258412/149445434-d90dfd28-4a56-479f-bdc7-54c67a853b03.JPG"/>
<img width="80%" src="https://user-images.githubusercontent.com/78258412/149445439-55e81bae-f9dc-4826-a09f-13967f650986.JPG"/>
<img width="80%" src="https://user-images.githubusercontent.com/78258412/149445440-284168de-9076-44e7-85bb-a7b54af0d132.JPG"/>
<img width="80%" src="https://user-images.githubusercontent.com/78258412/149445442-4b477c65-c48a-4808-88f6-9583b08850ee.JPG"/>
<img width="80%" src="https://user-images.githubusercontent.com/78258412/149445443-3746388b-e165-45b3-951e-f603ea3b2ff6.JPG"/>
<img width="80%" src="https://user-images.githubusercontent.com/78258412/149445444-2ad026ed-edbc-4938-b7e3-d44b8a110e5f.JPG"/>
<img width="80%" src="https://user-images.githubusercontent.com/78258412/149445446-1d61a548-91c8-4111-a6dd-4b93b8f29e05.JPG"/>
<img width="80%" src="https://user-images.githubusercontent.com/78258412/149445449-8573a3ac-6bfb-4ca9-8c5d-5eabb3ae5cd6.JPG"/>
<img width="80%" src="https://user-images.githubusercontent.com/78258412/149445451-d1be996f-717d-4994-bd87-918a41330370.JPG"/>
<img width="80%" src="https://user-images.githubusercontent.com/78258412/149445454-f089c22e-5571-4e68-912b-8af058ccc1cf.JPG"/>
<img width="80%" src="https://user-images.githubusercontent.com/78258412/149445457-44350b33-bfc1-4a2e-812c-f9fb84a6dcc5.JPG"/>
<img width="80%" src="https://user-images.githubusercontent.com/78258412/149445459-7f9fe209-fa6d-4c5b-b5be-ce44bea50b09.JPG"/>
<img width="80%" src="https://user-images.githubusercontent.com/78258412/149445461-4010d33a-2584-40ac-aceb-c4d0d2a65856.JPG"/>
<img width="80%" src="https://user-images.githubusercontent.com/78258412/149445463-b5112fbc-eb11-45a9-aefa-5456042985a7.JPG"/>
<img width="80%" src="https://user-images.githubusercontent.com/78258412/149445465-159f57f1-7f88-4b63-a5c7-c95bab950a2c.JPG"/>
