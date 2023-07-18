### Lecture 7 : Value function methods 

- Value-based method 

      - Policy를 explicit하게 학습하지 않는다.
      - V, Q를 바로 학습 : policy가 implicit하게 학습됨 


- Fitted Q Iteration

      - function approximator 사용 
      - transition probability를 몰라도 학습이 가능함 : model-free
        - Model을 알지 못해서 Dynamic Programming에서 Monte Carlo로 갔던 것과 같이
        - Model을 모를 때 FQI를 이용하는 방법도 있었음을 알게 됨. (s, a, r, s') sample만 있으면 됨.
  
        이 알고리즘이 흘러가는게 다음과 같다. (Growing batch RL과 같은 모양)
            1. 내가 갖고 있는 모~든 sample에 대해 target 계산
            2. fitting
            3. 1번 2번을 일정 횟수 반복 
            4. 더 많은 sample 확보 --- growing batch 

      근데 왜 FQI가 offline 강화학습의 Q-Learning일까? 
      Because ... Q Learning은 FQI의 online version!
  
      *exploration으로 sample들을 얻으러 가지 않고, 기존 sample들로 계속 iteration하면 Batch RL!
       sample들을 더 확보한다? Growing Batch RL!
  
      하지만 exploration이 없는 off policy 학습으로 인해 extrapolation error 발생 -> BCQ 제안

      대충 lecture에서 언급하는 fitted q iteration의 offline 과의 연결고리 정리 ok
              

---

- `topic list`

      1. policy iteration / value itaration
      2. Fitted V iteration (Model-based)
      3. Fitted Q iteration (Model-free)
      4. Q-Learning

      *Fitted Q Iteration은 Batch RL paper에서 다루었던 내용이고,
      Neural network를 이용한 방법은 Neural Fitted Q Iteration 이다.
  
        Linear FA     : FQI
        Non-Linear FA : NFQ
    
---

- `Fitted V Iteration & Fitted Q Iteration (function approximator을 사용)`

  
      PI/VI는 작은 테이블 형태의 discrete state/action space에 사용할 수 있다. (with DP)
  
      근데, state/action space가 매우 큰 discrete space거나, continuous인 경우에는
      이전처럼 table 형태로 만들어서 각 state마다의 value 들을 저장하고 업데이트 하는 방법이 절~대 절대 불가능하다.

      그래서.. function approximator이 사용된다. 
        
      PI,VI는 implicit하게 policy improvement를 수행한다. (explicit ? ex. actor)
        - PI : greedy policy 
        - VI : max Q(st,at)

      # Fitted V Iteration ( fitting : supervised regressior )

            FVI는 사실 Policy Iteration과 같은 일을 한다.
            다만, Policy improvement 과정을 버려버리고 그냥 direct하게 value function을 fitting한다.
            (policy improvement는 (1)단계에서 max operator로 implicit하게 수행한다.)

                1. y = max(r + γE[Vw(st+1)]) , where max(r + γE[Vw(st+1)]) '=. maxQ(st, at)
                2. update parameter w with argmin L(w) = sum ||Vw(st)-y||^2

            
            문제는 r + γE[Vw(st+1)] 를 계산하는 과정에서 우리가 transition probability 를 모른다는 것이다.
            So, off-policy actor-critic에서 처럼 V가 아니라 Q를 이용한다. 


      # Fitted Q Iteration 

        transition probability를 몰라도 되고, 그냥 (s,a,s',r) sample들만 있으면 된다!
        V대신 Q를 이용하는 방법은 거의 대부분의 Value based model-free 알고리즘이 차용하는 방법이다.

          1. y = r + γE[Vw(s')] --- approximate E[Vw(s')] with a single sample Vw(s')
             and we replace Vw(s') with maxQw(s', -) --- because we gonna use Q, not V
          2. so, y = r + γ maxQw(s', -)
          3. update parameter w with argmin L(w) = sum ||Qw(s, a) - y||^2
    
        다시 말하지만, 그냥 이전 policy을 통해 얻었던 sample들만 있으면 된다.
        심지어 걍 off-policy sample 들에 대해서도 아주 잘 작동한다. 
        즉, action이 가장 최근 policy에서 뽑힐 필요도 없고, 그냥 어느 policy에서 뽑힌 것이든 상관없다.
        그래서, 여지껏 뽑은 transition을 잘 모아놓고 update에 사용할 수 있다.
  
        *But, non-linear FA의 경우는 Convergence가 보장되지 않는다. 
  

---

- `Fitted Q Iteration`

        다시 full algorithm으로 정리하자면 ...
  
        1. Collect dataset (s, a, s', r) using some policy (꼭 latest policy 아니어도 된다)        
        2. Calculate a target value for all samples : y = r(s, a) + γmaxQw(s', -)
        3. Train a new Q function. (find a new parameter w)
           : w <- argmin sum ||Qw(s, a) - y||^2

        Iterate K times (2),(3) and, go (1) to collect more samples ~ (Growing batch RL)

        (1) : exploration (we can use greedy / soft-greedy)
        (2) : policy improvement (implicitly)
        (3) : policy evaluation  (fitting value function)

        *그럼 FQI는 어떻게 off-policy가 가능할까?
  
        사용하는 policy가 바뀔 때마다 영향을 받는 것은 (2)의 max operator이다.
        (3)의 Qw(s, a)은 이전 policy로 이미 뽑아놓은 sample s, a이기 때문에 transition prob과 무관하다.
        즉, max operator가 implicit하게 policy π(a|s)를 다뤄주므로,
        sample들이 최신 policy에서 나온 것이든, 옛날 policy에서 나온 것이든 무관하다.

        추가적으로, policy는 transition probability 와는 무관하다.
        policy는 π(a|s) 확률. 전이확률 P(s'|s, a)와는 무관
     
  
---

- `From FQI to Q-learning`


      # Fitted Q Iteration (off-policy)
  
      FQI 알고리즘의 (3) 단계의 fitting 과정을 살펴보면, 
      error를 최소화하는 optimization 단계이다. (fitting) 
      
        Error = ||Qw(s,a) - y|| = ||Qw(s,a) - (r(s, a) + γmaxQw(s', -))||
      
      여기서 Error가 0이 될때의 모양을 보면, 아래식과 같다. Q Learning에서 많이 봤지?
      
        Qw(s,a)  = r(s, a) + γmaxQw(s', -)
  
        TD error = Qw(s,a) - (r(s, a) + γmaxQw(s', -))
      
      즉, 위에서 이야기한 Error가 익히 알고 있는 TD error다.
      
      *보통 tabular case에서는 error가 0으로 수렴하는데, 이외의 상황에서는 보장되지 않는다.

      
      # Q learning (online version of Fitter Q Iteration, off-policy)

      1. take some action a and observe (s, a, s', r) 
      2. y = r + γmaxQw(s', )
      3. w <- w - α(Qw(s, a) - y)∇Qw
      
      (1)에서 action을 취하기 위해서는 soft-greedy policy를 사용한다. (for exploration)

      So ... 뭐 큰게 아니고..
  
      Q learning은 FQI의 online version.
      FQI는 Q Learning의 offline batch version.
 



