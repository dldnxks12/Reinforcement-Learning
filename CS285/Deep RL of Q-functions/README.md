### Lecture 8 : Deep RL with Q-Functions 

      lecture 7에서 다룬 Value-based method는 일반적으로 수렴이 보장되지 않는다.
      But, 그중 Q-Learning을 변형하면 그럼에도 아주 유용하고 강력한 RL 알고리즘이 탄생한다. 

---
      
      # Convergence of Q-learning (online / off-policy)

        *Gradient descent? X, Supervised Regression
        
          Q learning은 gradient descent를 이용해서 update하는 것 처럼 보이지만, 
          사실 단순히 regression을 수행하는 것 뿐이다. 그래서 수렴이 보장되지 않는다. 
            - y_target : r + γmaxQ(s', -)에서 γmaxQ(s', -) term에 gradient가 계산되지 않는다.
            - 즉, chain rule이 적용되지 않고, backpropagation이 수행될 수 없다. 
            - lecture 7에서 얘기했 듯, bellman backup + projection은 contraction mapping이 아니다. 
            - 따라서 fixed point로의 convergence는 보장되지 않는다.

        *Correlated samples 
        
          만약 Q learning이 gradient descent 방법이라고 할지라도, 여전히 단점이있다.
          Q-learning은 online 으로 하나의 transition을 얻고, 이를 통해 Q value를 update한다.      
          이렇게 sequential하게 하나씩 sample을 얻고 update하는 방법은 좋지 않다.
            - Why? Sequential transition 들을 서로 highly correlated하다. 
              (sequential states are strongly correlated)
            - 이 방법은 stochastic gradient descent의 가정을 위배한다. (i.i.d)
    
      아무튼 위 이유로 Q-learning의 수렴성이 보장되지 않고 불안정하다는 이야기를 마칠 수 있을 것 같다.

---

      # Deep Q Learning (DQN)

      Deep Q-Learning : Q-learning과 Fitted Q Iteration의 mixture with replay buffer & target 
    
            # Solution of correlated samples in Q-Learning - replay buffer
      
              Method 1) Parallel Q Learning
              
              Levine은 online actor-critic 방법과 같이 Q learning도 parallel하게 해서 correlation 문제를 완화한다.
                - Actor-Critic 때와 동일하게 동기 / 비동기 방법이 있다.
                - 다만 Q learning은 off-policy의 성격을 띄기 때문에 비동기 방법을 쓸 때, 
                  조금 더 오래된 sample을 써도 문제가 거의 없다.           
              즉, multi-thread로 agent를 여러 놈 만들고 각각 transition을 가져오게 해서 batch update 한다. 
              correlation 문제를 완전히 해결하지는 못해도 완화시킬 수는 있다. 
      
              Method 2) Q Learning with Replay buffer - unstable Deep Q-learning (DQN)
              
              또 한가지 방법으로는, 아주 잘 알고 있는 replay buffer를 사용하는 것이다. 
              이렇게 하면, Fitted Q Iteration과 아주 비슷한 형태가 된다.
              Fitted Q Iteration의 (2), (3) 과정은 동일하고, 
              dataset을 모으는 과정이 replay buffer에서 load 해오는 것으로 바뀌면 된다. 
              이제 sample들은 더이상 correlated 하지 않다. -> SGD의 i.i.d 가정이 성립.
      
              사실 이 형태가 DQN의 초기 형태이다. 
              Replay buffer를 통해 기존 Q-learning의 correlation 문제를 해결했다.
              
              이제 학습 안정성을 위해 한 가지 요소만 더 추가해보자.
      
            
            
            # Improve stability - Target Network 
              
              Q learning의 unstable 문제는 moving target이 원인 중 하나이다.
              그럼? frozen target을 만들어주면 된다. This is target network!
        

- `General view `


        Q-Learning, Fitted Q Iteration은 DQN의 special case이다.
        Lecture를 통해 확인해볼 수 있고, 요약하자면 process의 순서, 횟수의 차이일 뿐이다.


---

- `Improving Q-Learning`


      # Overetimation bias in Q-learning

            Are the Q-values accurarte?
            There's overestimation bias!
                  - Q-learning의 성능 저하의 큰 원인

            Q value가 정확한게 왜 중요하지?
                  - 부정확한 target을 만들어내기 때문
                  - moving target은 학습 불안정성을 야기했는데
                  - 부정확한 Q value는 학습 자체를 망칠 수 있다!!
  
            So, 'Double Q-Learning' has proposed!!


---

- `Q-Learning with continuous actions`


      # Continuous acion이 왜 문제인데?
              - 더이상 greedy policy를 쓸 수가 없다...!
              - max operator로 implict하게 policy improvement를 수행해왔다..

      # How do we perform the max operator?
            Option 1) Gradient based optimization (e.g., SGD)
            Option 2) Use function class that is easy to optimize
                  - Q function을 quadratic function으로 구성 (e.g., NAF Architecture)
            Option 3) Learn an approximate maximizer
                  - train another network X_θ(s) such that X_θ(s) ≈ argmax Q_ϕ(s, a).
                  - e.g., DDPG (really approximate Q-learning)
                    
      
      # Option 1 : Q-Learning with stochastic optimization 
            1) Simple solution : approximate max operation
                  max Q(s,a) ≈ max {Q(s, a1), Q(s, a2), Q(s, a3), ... Q(s, aN),}
                        - action 들은 uniform distribution 같은 어떤 분포에서 sampling한다.
                        - exact max는 아니고, approximate하게 max를 수행할 수 있다. 

            2) More accurate solution : cross-entropy method (CEM) (* works OK for up to 40 dims)
                        - used in model-based
                        - simple iterative stochastic optimization
  
            3) Fancier version of CEM : CMA-ES
                        - substantially less simple iterative stochastic optimization 
  

      # Option 3 : Learn an approximate maximizer

            이미 알고 있듯이,
            DPG를 통해 discrete action space -> continuous action space로 확장되었고,
            DDPG를 통해 더 고차원의 continuous action space에도 RL을 적용할 수 있게 되었다.

  
  
---

- `Practical tips for Q-Learning`

            *Q-Learning을 base로 하는 알고리즘에도 효과적일지는 잘 모르겠다.

            1) Bellman error gradients can be big 
              - use clip gradient or Huber loss

            2) Double Q-Learning 
              - simple and no downsides

            3) N-step returns
  
            4) Schedule 
                  Exploration   : high to low
                  Learning rate : high to low
                  - Adam optimizer can help, too

            5) Run multiple random seeds
                  - it's very inconsistent between runs 



      

  
  
      
            
            
  

        

  

            

      
      
      
      
      
