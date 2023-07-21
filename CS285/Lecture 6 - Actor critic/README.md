### Lecture 6 : Actor Critic 

Replay buffer에 들어있는 지난 actor(old parameter)들이 생성해냈던 old transition을   
지금의 actor를 update하는데에 써도 괜찮은 지에 대한 궁금증을 여기서 좀 해소할 수 있었다.  

다시 복기할 때 좀 헷갈리는 부분도 있었지만 그냥 간단하게 정리하면, 

    1) PG는 policy를 parameter화한 모델 
       AC는 policy and value function 둘 다 parameter화한 모델
    
    2) AC는 PG의 variance를 상당히 줄인 모델 

---

    # What is Actor-Critic? 

      1. Actor-Critic은 policy와 value function 둘 다 parameterize한 모델
      2. PG의 variance를 줄인 모델.
      
      PG : unbiased / high variance
      AC : biased   / low variance -- 하지만 만약 Critic이 compatible FA? unbiased!

---

    # Policy update : ∇J(θ)
        
      PG의 REINFORCE 알고리즘에서 ∇J의 Q(st, at)부분은 Gt로 stochastic하게 Q(s,a)를 근사할 수 있다.
        - Q(st,at) ≈ Gt

      근데, 이 방법은 variance를 크게 증가시키기 때문에 학습 속도나 수렴 속도를 매우 느리게 만든다.
      또한 reward를 모두 모아야하기 때문에, episodic case에만 사용이 가능. 
      
      그래서 Q(st,at)를 'w로 parameterize'된 Qw(st, at)로 근사하는 방법이다.

        - compatible FA를 따르면, Qw도 여전히 unbiased estimator이다. 


    # Value function update : ∇L(w)

        Value function의 ideal target은 당연히 true value이다.
        하지만, true value를 찾기 어려우므로, r + γV*로 대체한다. (bootstrap)

        supervised regression : L = sum(||r+γV* - V||^2)
    
---


    # 그럼 Critic을 Q, V, A 중 뭘로 해야 좋은가?

      - Q와 A 모두 V로 표현할 수 있으니, V가 일반적으로 선택된다. 


---

    
    # online actor-critic algorithm

      # on-policy actor-critic (online)

      online으로 agent가 env랑 상호작용하면서 얻는 sample을 사용해서 그때그때 update할 수 있다.
      But as we know, stochastic GD에서 sample 1개로 하는 건 너무 variance가 크다.
      그래서 보통 batch update , mini-batch update를 수행한다.
      
      아래 두 방법은 on-policy actor-critic에서 batch update를 하는 방법들이다.
      
      1) synchronous parallel actor-critic :
        1. run multiple simulators and get transitions from multiple simulators.
        2. update with those transitions synchronously

        
      2) asynchronous parallel actor-critic :
        1. run multiple simulators with their own speed and get transitions.
        2. 각 thread가 batch size만큼의 transition이 모이면 actor update.        

        *transition이 같은 parameter 상에서 생성되지 않는 경우가 있다.
          -> 즉, 앞선 thread가 먼저 32개 transition을 모아서 update하면, 
          뒤의 thread가 만들어내는 transition은 이전과 다른 parameter 상에서 생성된다.

          근데 levine은 이 thread들 간의 time lag가 그렇게 크지 않다면 별~ 상관없다고 한다. 

                      
      *batch size 만큼의 worker가 필요하다. 
      *동기적 방법보다 비동기적 방법이 더 빠르다.
      *비동기적 방법에서 약간의 time lag는 괜찮다. 잘 동작한다.

      
      # off-policy actor-critic (online & use replay buffer)
      
      위 asynchronous actor-critic에서는 조~금 오래된 actor에서 뽑아낸 transition을 써도 상관없다고 했다.
      근데 만약 좀 많~이 오래된 actor에서 뽑아낸 transition을 사용해서 update 해도 된다면??
      그럼 multiple threads가 필요없을지도 모른다!
  
      즉, transition들을 바구니에 잘 담아두고서, 지금의 actor(parameter)로 뽑아낸 transition으로도 update하고,
      그리고 옛날 transition들로도 update해도 된다는 얘기잖아?
  
      이게 off-policy  actor-critic의 개념이다. 
      
        1. 1개의 thread를 사용한다.
        2. 생성되는 transition들을 모두 replay buffer에 담는다.
        3. replay buffer에서 transition들을 batch 만큼 load해서 update한다. 

      근데 사실 이렇게 naive하게 쓰는건 잘 안된다. 
      비동기적 on policy actor-critic에서는 조~금 lag된 transition을 사용했는데, 
      이건 좀 다르다. 완전 옛날 transition을 쓰는거라서 알고리즘의 수정이 좀 필요하다. 

      -> lectrue 참고 
      

---
      
