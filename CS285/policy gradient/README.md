### Lecture 5 : Policy Gradient 


---

- `list`

      PG의 문제점 2가지
      1. high variance   -> sol. reducing variance
      2. numerical issue -> sol. natural pg (or covariant pg)

---


    # PG가 가끔 잘 동작하지 않는 이유 

      objective function J의 gradient 계산은 ∇log_pi(a|s) x reward(s, a)들의 sample mean으로 계산된다. 
      즉, sample ∇log_pi(a|s)들의 가중합 형태.

      이 값들을 가지고 policy 분포를 조정한다.
      
      만일 커다란 음수 reward와 양수 reward의 두 sample에 대해 gradient를 계산한다고 가정하면,
      update 후의 pi 분포의 음수 reward의 부분은 낮아지고, 양수 reward 부분은 커진다. (조금 더 뾰족해지겠지 - variance 작음)
      
      이제, reward에 큰 constant를 더해서 모두 양수의 형태를 갖게 한다고 가정해보자. 
      그럼 policy의 분포는 어떻게 바뀔까?       
      policy 분포의 음수 reward 였던 부분은 확률이 조금 높아지고, 원래 양수였던 reward 부분은 보다 더 커진다. (완만 - variance 큼)
      (+ argmax op를 취하기 때문에, 모든 reward에 constant value를 더해도 optimal policy의 결과는 변하지 않는다.)

      즉, reward 간의 상대적인 차이는 동일할 때, 그 값의 offset에 따라 policy 분포의 모양이 바뀐다.  ex. (-3, 3) & (3, 6)
      또한 update할 때 어떤 sample들이 모였냐도 중요한 point가 된다.
      
      결국 모든 sample들을 거쳐 update하기에 결과로 얻는 policy 분포는 같겠지만, 학습과정에서 그 분포가 크게크게 달라질 수 있다는 것.  
      * 이렇게 분포가 크게크게 바뀌는 것은 학습 불안정성을 유발한다. 
      


      1. 모이는 sample들에 따라서 policy 분포가 크게크게 바뀐다는 점
      2. reward의 offset에 따라서도 policy 분포가 영향을 받는다는 점 (1 과 같은 얘기라고도 볼 수 있다.)

        - sample을 무진장 많이 모아서 계산하면 당근 문제가 없다.
        - 하지만 stochastic하게 계산할 때는 문제가 된다. (학습 불안정)
        
        (RL은 보통 stochastic gradient descent method를 사용한다.) 
      
      * 정리하자면, 'PG는 variance가 크다' 가 문제라고 할 수 있다.


---


    # Reducing Variance in PG

      ref : [https://talkingaboutme.tistory.com/entry/RL-CS285-Reducing-Variance]

      이미 공부하면서 자주 봤듯이, PG의 variance를 줄이는 방법들이 많이 제안된다. 

      이 중 baseline 방법을 좀 살펴보자.
      
      baseline을 빼주는 것으로 variance를 줄인다는 건 알고 있었지만, Levine의 설명으로 조금 더 확실히 이해했다.

      즉, reward가 음수/양수 일때는 bad action에 대해 분포가 감소하고, 양수 일때는 해당 action에 대해 분포가 증가한다.
      
      하지만 reward가 모두 양수 일때는 분포가 모두 증가하며 그 '증가 폭만이 다를 뿐'이다.
      (recall. reward off이 어떻든 optimal policy 결과는 동일하다. )

      Levine은 안좋은 action은 확률을 감소시키고 좋은 action은 증가시키는게 맞다고 한다.
      조금 증가, 많이 증가 요딴게 아니라.

      그래서 여기에 평균값을 빼주면 음수 / 양수 reward의 형태가 나타나니까 분포를 증감하도록 할 수 있다는 것. 이 말이었다!
      (avg 보다 좋은 action은 증가 / avg보다 나쁜 action은 감소)

      이 방법을 쓰면, reward의 offset이 어떻든 상관없다. 


        -> gradient 계산에 아무 영향도 없다. but variance는 줄여준다. Wow...


      Levine은 optimal baseline을 구하는 방법을 알려주는데, 그냥 average reward 쓰라고 한다. (dirty and fast baseline)
  


---


    # PG is originally on-policy but we can modify to off-policy with IS.

      But we can modify with importance sampling 

      Importance sampling -- unbiased. but variance는 바뀔 수 있다. 



--- 


    # PG의 numerical issue (especially in Continuous action spaces.)

    # gradient descent method 
    (1) : θ <- θ + α∇J(θ) 
    
    (1) 에서 step size α를 결정하는 건 매우매우 조심스럽게 진행해야 된다. 
    
    왜? some parameters in θ change probabilities a lot more than other!    
    즉, θ = (a, b)일 때, a는 매우 민감하고, b는 매우 둔감할 때 이 둘 간의 trade-off를 잘 고려한 step size α가 필요!
    
    결론적으로 우리는 step size α를 잘 골라서 모든 parameter가 gradient update를 통해 optimal 한 값으로 가길 원한다. 

    So, parameter space에서 alpha를 잘 조정해서 gradient update하기는 까다롭다.



    # fisrt-order gradient descent method view     
    
    위 (1) 식을 mathematical하게 보자면, 1차 gradient descent의 constraint optimization으로 볼 수 있다.    
        
    (2) : θ* <- argmax(θ*-θ)^T * ∇J(θ)  ,   s.t ||θ*-θ||^2 < ε 

    (2) 식은 parameter space에서 ε bound 내부에서 J(θ)의 gradient를 가장 크게 update하게 해주는 θ*를 찾는 것이다. 

    여기서 step size α는 위 constraint optimization 식의 Lagrange multiplier로 볼 수 있다. 

        위 (2) 식의 ∇J(θ)는 J(θ)의 1차 선형 근사이므로, 아주 작은 region에서만 이 식이 valid하다.
        그래서 step size α가 큰 값이 될 수 없는 것이다!!

    근데 이 region을 선택하는게 좀 이상하다.    
    위에서 말했듯, 그 region 안에서 parameter를 선택해야하는데, 어떤 parameter는 예민하고 또 어떤 parameter는 둔감한데
    이 space 안에서 딱 하나 골라서 그게 해결될까?


    그래서 우리는 이 update 과정을 parameter space에서가 아니라 policy space에서 다루고자 한다. 
    (Covariant : space의 기저가 바뀌어도 vector에 영향을 미치지 않은 것.)
    즉, (2) 식의 이 constraint가 잘못된 space에서 이루어지고 있다. 
    
        - s.t ||θ*-θ||^2 < ε  : Wrong space !!

    우리는 이 constraint space를 parameter space에서 policy space로 옮긴다.

        - s.t D( π(θ*), π(θ) ) < ε  : Parameter independent divergence measure!!

    위 같이 분포가 크게 바뀌지 않는 것으로 제약 조건을 바꾼다! 
    이렇게 하면 우리가 parameter를 어떻게 구성하든지 전혀 영향이 없다. 

    (2)  : θ* <- argmax(θ*-θ)^T * ∇J(θ)  ,   s.t ||θ*-θ||^2     < ε 
    (2*) : θ* <- argmax(θ*-θ)^T * ∇J(θ)  ,   s.t D(π(θ*), π(θ)) < ε


    Natural policy gradient / Trust Region policy optimization paper를 참고하면 알겠지만, 
    KL divergence를 2차 taylor expansion하면 위의 제약조건식을 다음과 같이 바꿀 수 있다!
    
    (3) : D(π(θ*), π(θ)) ≒ (θ*-θ)F(θ*-θ) - KL diverence의 2차 선형 근사

        F : Fisher information matrix (sample mean으로 쉽게 구할 수 있음)

    이제 위 (2*) 식의 Langrange으로 풀고 optimal solution을 구하면 gradient update가 다음과 같은 식으로 정리된다.

    
    (4) : θ <- θ + α(F-1)∇J(θ)
    
        where F-1 is inverse fihser information matrix.
        where α is Lagrange multiplier

    
    
    
    
    
    
    

    
































    
