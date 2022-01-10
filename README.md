# BERT 모델을 통한 게임회사 이용약관 불공정성 평가
![리걸테크_포스터-1](https://user-images.githubusercontent.com/66261167/148675558-a08b453a-eee8-43a5-90ce-586c8102c257.png)

> **Abstract:**  

> 2019년 공정거래위원회에서 주요 10개 게임서비스사업자들의 이용약관상 불공정약관조항에 시정 명령을 내렸다. [기사 바로가기](https://www.ftc.go.kr/www/selectReportUserView.do?key=10&rpttype=1&report_data_no=8206) 뿐만 아니라 최근 게임이용자들의 불만이 고조되어 이른바 트럭시위가 발생하였고, 게임회사 이용약관의 불공정성을 시정하고자 관련 법률 개정안 발의되었다.(『전자상거래등에서의소비자보호에관한법률 』 일부개정법률안, 2021.7.30. 발의)
> 이러한 게임서비스제공자-이용자 간 공정성 제고에 대해 사회적 요구를 딥러닝을 이용하여 해결하고자 하였다.  
>  
> 공정거래위원회의 보도자료와 전문가 1명(법무법인 정앤헌의 정** 변호사), 법적 소양이 있는 사람 49명 대상 설문조사(이른바 델파이기법)를 바탕으로 게임사 이용약관 불공정성 평가지표를 생성하였다.  
>  
> 생성된 평가지표를 바탕으로 주요 8개 게임사의 이용약관 조항 3773개의 불공정/공정 라벨을 직접 입력하였다.  
>  
> 생성한 라벨을 BERT 모델로 fine-tuning하여 국내 약 30개 게임사의 이용약관 불공정성 점수를 자동적으로 예측하게 하였다.  

> In 2019, the Fair Trade Commission ordered correction of unfair terms and conditions of 10 major game service providers. [Shortcut to the article](https://www.ftc.go.kr/www/selectReportUserView.do?key=10&rpttype=1&report_data_no=8206) In addition to, recently, complaints from game users have risen, so-called truck protests have occurred. And related laws have been proposed to correct the unfairness of game company terms and conditions. (Partial Amendment to the Consumer Protection Act in Electronic Commerce, July 30, 2021)
> Therefore, I tried to solve social demands for improving fairness between game service providers and users using deep learning.  
>  
> Based on the Fair Trade Commission's press release and survey (for a lawyer and 49 law school students and preparation students), the so-called Delphi technique was used to generate unfairness evaluation indicators for game companies.
>  
> Based on the generated evaluation indicators, we directly evaluated whether the 3773 terms and conditions used in eight major game companies were unfair or fair.
>  
> The generated label was fine-tuned with the BERT model to automatically predict the unfairness score of about 30 game companies in Korea.
   
<Br>  

## How to Install
Use `conda`:
   ```
    conda env create -n gameTerms -f environment.yaml
    conda activate gameTerms
   ```
   
## Usage
    
+ **Train**
   
   The BERT model is fine-tuned with the dataset in `./label/`,  
   and the trained model is stored in `./model/bert_over.h5`  
    ```
    python run.py -k train -d ./label/ -m ./model/bert_over.h5 -o 10
    ```
+ **Test**
   
   Performance evaluation with data stored in `./test/`  
   by loading a model stored in `./model/bert_over.h5`  
    ```    
    python run.py -k test -d ./test/ -m ./model/bert_over.h5
    ```
+ **Predict**
   
   Load the model stored in `./model/bert_over.h5`  
   to predict whether the data in `./predict/` is unfair  
   and store the results in `./result/
    ```
    python run.py -k predict -d ./predict/ -m ./model/bert_over.h5 -r ./result/
    ```
    
## Citaion
If you refer to the code, please attach to the following phrase.
  ```
  Choi, S. M. (2021). GitHub - saemee007/gameTerms_BERT: Using deep learning, the unfairness score of the game company’s terms and conditions is calculated. GitHub. https://github.com/saemee007/gameTerms_BERT
  ```  
