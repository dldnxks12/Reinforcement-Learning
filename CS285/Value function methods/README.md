### Lecture 7 : Value function methods 

- Value-based method 

      - Policy를 explicit하게 학습하지 않는다.
      - V, Q를 바로 학습 : policy가 implicit하게 학습됨 


- Fitted Q Iteration

      - function approximator 사용 
      - transition probability를 몰라도 학습이 가능함 : model-free
        - Model을 알지 못해서 Dynamic Programming에서 Monte Carlo로 갔던 것과 같이
        - Model(전이확률)을 모를 때 FQI를 이용하는 방법도 있었음을 알게 됨.
  
      근데 왜 FQI가 offline 강화학습의 Q-Learning일까? 
        1. off-policy 학습이 가능해서 그런게 아닐까?
          (지난 policy들로 모은 transition들로 fitting이 가능)
  
        2. growing batch랑 비슷한 양상인데, 이 알고리즘이 일단 다음과 같다.
            1. 내가 갖고 있는 모든 sample에 대해 target 계산
            2. fitting
            3. 1번 2번을 일정 횟수 반복
            4. 더 많은 sample 확보

      *exploration으로 sample들을 얻으러 가지 않고, 그 sample들로 계속 iteration하면 그게 offline RL!
      하지만 exploration이 없는 off policy 학습으로 인해 extrapolation error 발생 -> BCQ 제안

      대충 lecture에서 언급하는 fitted q iteration의 offline 과의 연결고리 정리 ok
              

---

- `topic list`

      1. policy iteration
      2. value iteration
      3. Fitted V iteration  (Model-based)
      4. Fitted Q iteration (Model-free)
      5. Q Iteration

      *Fitted Q Iteration은 Batch RL paper에서 다루었던 내용이고,
      Neural network를 이용한 방법은 Neural Fitted Q Iteration 이다.
  
        Linear FA     : FQI
        Non-Linear FA : NFQ
    
---

- `Fitted V Iteration & Fitted Q Iteration (function approximator을 사용)`

  
      PI/VI는 작은 테이블 형태의 discrete state/action space에 사용할 수 있다. (with DP)
  
      근데, state/action space가 매우 큰 discrete space거나, continuous인 경우에는
      이전처럼 table 형태로 만들어서 각 state마다의 value 들을 저장하고 업데이트 하는 방법이 절~대절대 불가능하다.

      그래서.. function approximator(NN)가 도입된다. 
        
      PI,VI는 implicit하게 policy improvement를 수행한다. (explicit ? ex. actor)
        - PI : greedy policy 
        - VI : max Q(st,at)

      # Fitted V Iteration ( fitting : supervised regressior 을 이용한다. )

          FVI는 사실 Policy Iteration과 같고, 같은 일을 한다.
          다만, Policy improvement 과정을 버려버리고 그냥 direct하게 value function을 fitting한다.
          

          1. y = max(r + γE[Vw(st+1)]) , where maxQ(st, at) '=. max(r + γE[Vw(st+1)]) 
          2. update parameter w with argmin L(w) = sum ||Vw(st)-y||^2
  
        문제는 r + γE[Vw(st+1)] 를 계산하는 과정에서 우리가 transition probability 를 모른다는 것이다.
        So! V가 아니라 Q를 이용한다. 

      # Fitted Q Iteration 

        더이상 transition probability를 몰라도 되고, 그냥 (s,a,s',r) sample들만 있으면 된다!
        V대신 Q를 이용하는 방법은 거의 대부분의 Value based model-free 알고리즘이 차용하는 방법이다.

          1. y = r + γE[Vw(s')] --- approximate E[Vw(s')] with a single sample Vw(s')
          2. and we replace Vw(s') with maxQw(s', -) --- because we gonna use Q, not V
          3. so, y = r + γ maxQw(s', -)
          3. update parameter w with argmin L(w) = sum ||Qw(s, a) - y||^2
    
        다시 말하지만, 그냥 내가 이전 policy를 통해 얻었던 action sample들만 있으면 된다.
        심지어 걍 off-policy sample 들에 대해서도 아주 잘 작동한다. (unlike actor-critic)
        즉, action이 가장 최근 policy에서 뽑힐 필요도 없고, 그냥 어느 policy에서 뽑힌 것이든 상관없다.
        그래서, 여지껏 뽑은 transition을 저장해놓고 update에 사용할 수 있다.
  
        But, non-linear FA의 경우는 Convergence가 보장되지 않는다. 
  

---

- `Fitted Q Iteration`

        다시 full algorithm으로 정리하자면 ... 

        1. Collect dataset (s, a, s', r) using some policy (꼭 latest policy 아니어도 된다!)
        2. Choose batch size N, choose policy (usually latest policy 쓰긴 한다.)
        3. Calculate a target value for all samples : y = r(s, a) + γmaxQw(s', a').
        4. Train a new Q function. (find a new parameter w)
           : w <- argmin sum ||Qw(s, a) - y||^2

        Iterate K times 3,4 and, go collect more samples ~ (Growing batch 였지)
  
  

