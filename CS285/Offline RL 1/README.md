### Lecture 15 : Offline RL 

offline RL / batch RL / fully off-policy RL --- someone else fullfilled my buffer for me with some policy beforehand.

behavior policy : somebody else collected buffer by running some other policy, this is behavior policy !

offline RL : behavior policy ? any kind of policy !
imiatation learning : behavior policy ? expert policy !

---

types of offline RL problems

  1. off-policy evaluation (OPE)
  
    given D, estimate J(some policy)
    
      -> Dataset을 collect한 policy가 아닌 다른 policy의 J를 계산 

      So, it is called as off-policy evaluation. 
  
  2. offline RL 
  
    given dataset D, learn the best 'possible' policy !

    inner loop에서 OPE 수행 
    
    why best? why not optimal? because in given dataset D, there could be never exist optimal policy 

    it is not a best policy in MDP, it is a best policy for a given D.

---

offline RL이 항상 dataset의 best behavior 만큼만 좋냐?  in general, NO!
given dataset 내에서의 best behavior 보다 더 좋은 policy를 학습할 수 있다.  ---  generalization 

* generalization : good behavior in one place may suggest good behavior in another place!

  driving human을 생각해보자.
  
  -> 회전 교차로를 잘 도는 agent
  -> 직선 도로를 잘 달리는 agent
  -> 좌회전을 잘하는 agent

  즉, 각각 다른 장소에서의 좋은 behavior들을 (dataset을) 잘 합쳐서, dataset에서 보지 못한 place에서의 좋은 behavior를 얻는데 사용할 수 있다.
  
  So, it is possible to do 'better' than the behavior policy in the dataset!

  stitching is one of the example.

  A -> B로 가는 behavior가 있고,
  B -> C로 가는 behavior가 있을 때,

  두 dataset을 잘 합쳐서 학습하면, A -> C로 가는 behavior를 학습할 수 있다. 


bad intuition : it's like imitation learning!

better intuition : get order from chaos!

  *Macro : 큰
  Macro-scale stitching : 내 dataset이 goal 근처에도 못가는 subpotimal trajectory들의 data들로 구성되있을 때, 
  
  offline RL은 generalization을 통해 더 나은 behavior를 찾을 수 있다. 

  *Micro : 작은 
  Micro-scale stitching : 예를 들어 A -> B 지점으로 가는 여러 명의 human driver suboptimal dataset이 있을 때, 

  recombine the best part of them, we can get super human driver. 


  So, 하고 싶은 말은 highly suboptimal dataset 들로 near-optimal policy를 뽑아낼 수 있다는 것. 
  
  

---

Why is offline RL Hard???

  - out-of-distribution / generalization

    
    직선 도로를 달리는 driving human을 통해 얻은 데이터셋을 생각해보자.

    물론 이 human은 expert 가 아닌 random skillfull human이다. (random policy)

    근데 여기서 생각해볼 것은, 이 human driver이 하지 않는 결코 행동이 있다는 것이다.

    예를 들어, 길을 달리다 갑자기 바다로 뛰어드는 등의 말도 안되는 crazy한 행동은 하지 않지 않냐는 말이다.

    하지만 바다로 뛰어든다던지, 아니면 산 비탈길로 차를 몰아서 떨어진다던지 하는 행동이 나쁘다는 건,

    경험을 해봐야 그 행동이 나쁘다는 걸 학습할 수 있다.

    우리는 policy를 업데이트할 때, 현재 state에서 가능한 모든 action을 살펴보고, 평가한 뒤 어느 행동이 가장 좋은지를 학습한다.
    
    online RL의 경우 가능한 모든 action을 취해보고, good or bad를 평가할 수 있다.

    하지만 offline RL의 경우 모든 action에 대한 sample transition이 없기에, 모든 action에 대한 평가를 내릴 수가 없다.

    well.. 몇몇 action들은 good or bad를 알 수도 있다. -> because of feneralization, we might have seen similar action in similar state.

    But, 당연하게도 우리의 dataset의 coverege는 완벽하지 않다. 

    So, there are certain behaviors which 우리가 이전에 본적은 없는, 하지만 수행은 가능한 action들이 ...

    이러한 문제를 out-of-distribution 문제라고 한다.

    이 ood 문제는 매우매우 중요한데, 이 문제를 해결하지 못하면 아까 말헀던 stitching과 같은 generalization을 building 하지 못한다.

    우리는 ood 문제에 집중하는 동시에 dataset에서의 best behavior보다 더 좋은 behavior를 학습하는 generalization을 달성해야한다.


---
    
  - Distribution shift in a nutshell (see ood in mathematically)

    distribution shift : one particular distribution we use in training, and we need to perform well.. under a 'different' distribution.

    because of this distribution mismatch, the performance may poor. 
