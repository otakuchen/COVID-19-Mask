import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd



#-----------------------------------------------------------------------
# Complete Model 
#-----------------------------------------------------------------------

class Threeperiod_SIRD:
    
    def __init__(self,S=0.9,        # initial susceptible
                      I=0.1,        # initial infected
                      R=0,          # initial recovered
                      D=0,          # initial died
                      𝛽=2.4/(18/14),# basic transmission rate. R0=2.4 and it takes 18 days to leave I state in average.
                                    # Furthermore, a time unit is 14 days here.
                      𝛾=1-(17/18)**14,# propotion of people that will leave state I is one minus those does not leave in fourteen days 
                      λ=0.0138,     # propotion that will die after leave state I.
                                     
                      
                      T=0,          # model period
                      q_0=0.2,      # mask issued during period 0 

                      σo=0.5,       # old facemask inward protection
                      σn=0.7,       # new facemask inward protection
                      δo=0.5,       # old facemask outward protection
                      δn=0.7,       # new facemask outward protection

                      q1_0=0.2,     # mask issued during period 1 for those who claim he does not own a mask during period 0
                      q1_1=0.2,     # mask issued during period 1 for those who claim he owns a mask during period 0
                      
                      # (x,y) 
                      # x=0 if one claim he does not own a mask during period 0, x=1 otherwise 
                      # y=0 if one does not receive a mask during period 1, y=1 otherwise
                      q2_00=0.2,    # mask issued during period 2 for (0,0)
                      q2_10=0.2,    # mask issued during period 2 for (1,0) 
                      q2_01=0.2,    # mask issued during period 2 for (0,1) 
                      q2_11=0.2):   # mask issued during period 2 for (1,1) 

                                         
        self.S    = np.array([S,0,0,0,0,0,0,0])
        self.I    = np.array([I,0,0,0,0,0,0,0])
        self.R, self.D  = R, D
        self.𝛽, self.𝛾, self.𝜆 = 𝛽, 𝛾, 𝜆
        self.σo, self.σn, self.δo, self.δn = σo, σn, δo, δn

        self.T, self.q_0 = T, q_0
        self.q1_0, self.q1_1, self.q2_00, self.q2_10, self.q2_01, self.q2_11 = q1_0, q1_1, q2_00, q2_10, q2_01, q2_11
        
    def evaluate_change(self):
        T = self.T
        𝛽, 𝛾, λ = self.𝛽, self.𝛾, self.𝜆
        σo, σn, δo, δn = self.σo, self.σn, self.δo, self.δn
        q_0, q1_0, q1_1, q2_00, q2_10, q2_01, q2_11 = self.q_0, self.q1_0, self.q1_1, self.q2_00, self.q2_10, self.q2_01, self.q2_11

        if T==0:
            # population distribution after issuing mask
            transition_0  = np.array([[1-q_0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [  q_0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0]])
            S_mask = transition_0.dot(self.S) # 8x1
            I_mask = transition_0.dot(self.I) # 8x1

            # masking state: ϕ o n n n ϕ o o
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),(1-δn),1,(1-δo),(1-δo)],[1,(1-σo),(1-σn),(1-σn),(1-σn),1,(1-σo),(1-σo)])
            𝛽0 = 𝛽 * matrix
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        
        if T==1:
            ##### 這裡要先算有口罩跟沒口罩的死亡人數，作為之後限制式要用
            self.d1_0, self.d1_1 = 𝜆 * self.I[0], 𝜆 * self.I[1]
            # population distribution after issuing mask
            transition_1 =  np.array([[1-q1_0,     0,0,0,0,0,0,0],
                                      [     0,1-q1_1,0,0,0,0,0,0],
                                      [     0,     0,0,0,0,0,0,0],
                                      [  q1_0,     0,0,0,0,0,0,0],
                                      [     0,  q1_1,0,0,0,0,0,0],
                                      [     0,     0,0,0,0,0,0,0],
                                      [     0,     0,0,0,0,0,0,0],
                                      [     0,     0,0,0,0,0,0,0]])

                                      
            S_mask = transition_1.dot(self.S) # 8x1
            I_mask = transition_1.dot(self.I) # 8x1

            # masking state: ϕ o n n n ϕ o o
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),(1-δn),1,(1-δo),(1-δo)],[1,(1-σo),(1-σn),(1-σn),(1-σn),1,(1-σo),(1-σo)])
            𝛽0 = 𝛽 * matrix
            # transmission
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        elif T==2:
            # population distribution after issuing mask
            # Because after issuing masks 
            # 第二期發口罩，但是發完之後就只分 無 舊 新 三個狀態，所以變換矩陣是 8x3
            transition_2 = np.array([[1-q2_00,0,0,0,0,1-q2_10,      0      ,0],
                                     [      0,0,0,0,0,      0,1-q2_01,1-q2_11],
                                     [  q2_00,0,0,0,0,  q2_10,  q2_01,  q2_11]])
            S_mask = transition_2.dot(self.S) # 3x1
            I_mask = transition_2.dot(self.I) # 3x1

            # 無 舊 新 
            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            # 因為 interaction 改變狀態
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            
        elif T>=3:
            # 進入第三期開始不用管，大家都有口罩
            transition = np.array([[0,0,0],[0,0,0],[1,1,1]])
            S_mask = transition.dot(self.S) # 3x1
            I_mask = transition.dot(self.I) # 3x1

            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            # 因為 interaction 改變狀態
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        
        # I 的改變
        dR = 𝛾 * (1-𝜆) * I_mask # 3x1 vector 因為 I 轉到 R 不管有沒有口罩都是相同
        dD = 𝛾 * 𝜆 * I_mask # 3x1 vector 因為 I 轉到 D 不管有沒有口罩都是相同

        nS = S_mask - dS
        nI = I_mask + dS - dR - dD
        nR = self.R + sum(dR)
        nD = self.D + sum(dD)

        # 因為這期用過口罩，所以新口罩的狀態變成舊口罩，先定義轉換矩陣
        if T<=1:
            transition_mask = np.array([[1,0,0,0,0,0,0,0],   # S0→S0
                                        [0,0,1,0,0,0,0,0],   # S2→S1 第零期新口罩到舊口罩
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],   
                                        [0,1,0,0,0,0,0,0],   # S1→S5 第一期沒拿到口罩，舊口罩變成沒口罩
                                        [0,0,0,1,0,0,0,0],   # S3→S6 第一期有拿到口罩，變成新口罩
                                        [0,0,0,0,1,0,0,0]])  # S4→S7 第一期有拿到口罩，變成新口罩
            
        else:
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])
        
        # 人口轉換
        nS = transition_mask.dot(nS)
        nI = transition_mask.dot(nI)
        return(np.array([nS,nI,nR,nD]))
    
    def update(self):
        
        change = self.evaluate_change()
        self.S, self.I, self.R, self.D = change
        self.T = self.T+1
    
    def generate_sequence(self, t):
        "Generate and return a time series of length t"
        S_path = [0]*t
        I_path = [0]*t
        R_path = [0]*t
        D_path = [0]*t
        
        for i in range(t):
            S_path[i], I_path[i], R_path[i], D_path[i]=sum(self.S), sum(self.I), self.R, self.D
            self.update()
        return(S_path,I_path,R_path,D_path)

    def serveralupdates(self,t):
        for _ in range(t):
            self.update()
        return(self.D)



def evaluate_death(q1_0=0.2,q1_1=0.2,q2_00=0.2,q2_01=0.2,q2_10=0.2,q2_11=0.2,
                   S=0.9,I=0.1,R=0,D=0,
                   𝛽=2.4/(18/14),𝛾=1-(17/18)**14,λ=0.0138,
                   T=0,t=10,q_0=0.2,
                   σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    S = np.array([S,0,0,0,0,0,0,0])
    I = np.array([I,0,0,0,0,0,0,0])
    R, D = R, D
    T=0
    
    for _ in range(t):
        # 發口罩之後的人口比例
        if T==0:
            transition_0  = np.array([[1-q_0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [  q_0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0]])
            S_mask = transition_0.dot(S) # 8x1
            I_mask = transition_0.dot(I) # 8x1

            # 無 舊 新 新 新 無 舊 舊
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),(1-δn),1,(1-δo),(1-δo)],[1,(1-σo),(1-σn),(1-σn),(1-σn),1,(1-σo),(1-σo)])
            𝛽0 = 𝛽 * matrix
            # 因為 interaction 改變狀態
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            # I 的改變
            dR = 𝛾 * (1-𝜆) * I_mask # 3x1 vector 因為 I 轉到 R 不管有沒有口罩都是相同
            dD = 𝛾 * 𝜆 * I_mask # 3x1 vector 因為 I 轉到 D 不管有沒有口罩都是相同

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            # 因為這期用過口罩，所以新口罩的狀態變成舊口罩，先定義轉換矩陣
            transition_mask = np.array([[1,0,0,0,0,0,0,0],   # S0→S0
                                        [0,0,1,0,0,0,0,0],   # S2→S1 第零期新口罩到舊口罩
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],   
                                        [0,1,0,0,0,0,0,0],   # S1→S5 第一期沒拿到口罩，舊口罩變成沒口罩
                                        [0,0,0,1,0,0,0,0],   # S3→S6 第一期有拿到口罩，變成新口罩
                                        [0,0,0,0,1,0,0,0]])  # S4→S7 第一期有拿到口罩，變成新口罩

            # 人口轉換
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        
        if T==1:

            transition_1 =  np.array([[1-q1_0,     0,0,0,0,0,0,0],
                                      [     0,1-q1_1,0,0,0,0,0,0],
                                      [     0,     0,0,0,0,0,0,0],
                                      [  q1_0,     0,0,0,0,0,0,0],
                                      [     0,  q1_1,0,0,0,0,0,0],
                                      [     0,     0,0,0,0,0,0,0],
                                      [     0,     0,0,0,0,0,0,0],
                                      [     0,     0,0,0,0,0,0,0]])

                                    
            S_mask = transition_1.dot(S) # 8x1
            I_mask = transition_1.dot(I) # 8x1

            # 無 舊 新 新 新 無 新 新
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),(1-δn),1,(1-δo),(1-δo)],[1,(1-σo),(1-σn),(1-σn),(1-σn),1,(1-σo),(1-σo)])
            𝛽0 = 𝛽 * matrix
            # 因為 interaction 改變狀態
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            # I 的改變
            dR = 𝛾 * (1-𝜆) * I_mask # 3x1 vector 因為 I 轉到 R 不管有沒有口罩都是相同
            dD = 𝛾 * 𝜆 * I_mask # 3x1 vector 因為 I 轉到 D 不管有沒有口罩都是相同

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            # 因為這期用過口罩，所以新口罩的狀態變成舊口罩，先定義轉換矩陣
            transition_mask = np.array([[1,0,0,0,0,0,0,0],   # S0→S0
                                        [0,0,1,0,0,0,0,0],   # S2→S1 第零期新口罩到舊口罩
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],   
                                        [0,1,0,0,0,0,0,0],   # S1→S5 第一期沒拿到口罩，舊口罩變成沒口罩
                                        [0,0,0,1,0,0,0,0],   # S3→S6 第一期有拿到口罩，變成新口罩
                                        [0,0,0,0,1,0,0,0]])  # S4→S7 第一期有拿到口罩，變成新口罩

            # 人口轉換
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        elif T==2:
            # 第二期發口罩，但是發完之後就只分 無 舊 新 三個狀態，所以變換矩陣是 8x3
            transition_2 = np.array([[1-q2_00,0,0,0,0,1-q2_10,      0      ,0],
                                     [      0,0,0,0,0,      0,1-q2_01,1-q2_11],
                                     [  q2_00,0,0,0,0,  q2_10,  q2_01,  q2_11]])
            S_mask = transition_2.dot(S) # 3x1
            I_mask = transition_2.dot(I) # 3x1

            # 無 舊 新 
            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            # 因為 interaction 改變狀態
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            # I 的改變
            dR = 𝛾 * (1-𝜆) * I_mask # 3x1 vector 因為 I 轉到 R 不管有沒有口罩都是相同
            dD = 𝛾 * 𝜆 * I_mask # 3x1 vector 因為 I 轉到 D 不管有沒有口罩都是相同

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)
            
            # 因為這期用過口罩，所以新口罩的狀態變成舊口罩，先定義轉換矩陣
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])

            # 人口轉換
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        elif T>=3:
            # 進入第三期開始不用管，大家都有口罩
            transition = np.array([[0,0,0],[0,0,0],[1,1,1]])
            S_mask = transition.dot(S) # 3x1
            I_mask = transition.dot(I) # 3x1

            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            # 因為 interaction 改變狀態
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            # I 的改變
            dR = 𝛾 * (1-𝜆) * I_mask # 3x1 vector 因為 I 轉到 R 不管有沒有口罩都是相同
            dD = 𝛾 * 𝜆 * I_mask # 3x1 vector 因為 I 轉到 D 不管有沒有口罩都是相同

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            # 因為這期用過口罩，所以新口罩的狀態變成舊口罩，先定義轉換矩陣
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])

            # 人口轉換
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        T=T+1
    
    return(D)



def GRBT_evalDeath(p=1,q=0,
                   q1_0=0.2,q1_1=0.2,q2_00=0.2,q2_10=0.2,q2_11=0.2,q2_01=0.2,
                   q_0=0.2,
                   S=0.9,I=0.1,R=0,D=0,
                   𝛽=2.4/(18/14),𝛾=1-(17/18)**14,λ=0.0138,
                   T=0,t=10,
                   σo=0.5,σn=0.7,δo=0.5,δn=0.7):

    S = np.array([S,0,0,0,0,0,0,0])
    I = np.array([I,0,0,0,0,0,0,0])
    R, D = R, D
    T=0
    
    for _ in range(t):
        # 發口罩之後的人口比例
        if T==0:
            transition_0  = np.array([[1-q_0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [  q_0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0],
                                      [    0,0,0,0,0,0,0,0]])
            S_mask = transition_0.dot(S) # 8x1
            I_mask = transition_0.dot(I) # 8x1

            # 無 新 舊 新 無 新 舊 新
            matrix = np.outer([1,(1-δn),(1-δo),(1-δn),1,(1-δn),(1-δo),(1-δn)],[1,(1-σn),(1-σo),(1-σn),1,(1-σn),(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            # 因為 interaction 改變狀態
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            # I 的改變
            dR = 𝛾 * (1-𝜆) * I_mask # 3x1 vector 因為 I 轉到 R 不管有沒有口罩都是相同
            dD = 𝛾 * 𝜆 * I_mask # 3x1 vector 因為 I 轉到 D 不管有沒有口罩都是相同

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            # 因為這期用過口罩，所以新口罩的狀態變成舊口罩，先定義轉換矩陣
            transition_mask = np.array([[1,0,0,0,0,0,0,0],   # S0→S0
                                        [0,0,0,0,0,0,0,0],   
                                        [0,0,0,1,0,0,0,0],   # S3→S2 第零期新口罩到舊口罩
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0],   
                                        [0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0], 
                                        [0,0,0,0,0,0,0,0]])

            # 人口轉換
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        
        if T==1:

            # 先登記是否要買口罩
            signup= np.array([[1-p,0,  0,0,0,0,0,0],
                              [  0,0,  0,0,0,0,0,0],
                              [  0,0,1-q,0,0,0,0,0],
                              [  0,0,  0,0,0,0,0,0],
                              [  p,0,  0,0,0,0,0,0],
                              [  0,0,  0,0,0,0,0,0],
                              [  0,0,  q,0,0,0,0,0],
                              [  0,0,  0,0,0,0,0,0]])
            
            S_signup = signup.dot(S)
            I_signup = signup.dot(I)

            # 再來根據登記與否發口罩
            transition_1 =  np.array([[1-q1_1,0,     0,0,     0,0,     0,0],
                                      [  q1_1,0,     0,0,     0,0,     0,0],
                                      [     0,0,1-q1_1,0,     0,0,     0,0],
                                      [     0,0,  q1_1,0,     0,0,     0,0],
                                      [     0,0,     0,0,1-q1_0,0,     0,0],
                                      [     0,0,     0,0,  q1_0,0,     0,0],
                                      [     0,0,     0,0,     0,0,1-q1_0,0],
                                      [     0,0,     0,0,     0,0,  q1_0,0]])

                                    
            S_mask = transition_1.dot(S_signup) # 8x1
            I_mask = transition_1.dot(I_signup) # 8x1

            # 無 新 舊 新 無 新 舊 新
            matrix = np.outer([1,(1-δn),(1-δo),(1-δn),1,(1-δn),(1-δo),(1-δn)],[1,(1-σn),(1-σo),(1-σn),1,(1-σn),(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            # 因為 interaction 改變狀態
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            # I 的改變
            dR = 𝛾 * (1-𝜆) * I_mask # 3x1 vector 因為 I 轉到 R 不管有沒有口罩都是相同
            dD = 𝛾 * 𝜆 * I_mask # 3x1 vector 因為 I 轉到 D 不管有沒有口罩都是相同

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            # 這期人口不用轉換位置。
            transition_mask = np.eye(8)

            # 人口轉換
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        elif T==2:
            # 第二期發口罩，但是發完之後就只分 無 舊 新 三個狀態，所以變換矩陣是 8x3
            transition_2 = np.array([[1-q2_10,      0,1-q2_10,      0,1-q2_00,      0,1-q2_00,      0],
                                     [      0,1-q2_11,      0,1-q2_11,      0,1-q2_01,      0,1-q2_01],
                                     [  q2_10,  q2_11,  q2_10,  q2_11,  q2_00,  q2_01,  q2_00,  q2_01]])
            S_mask = transition_2.dot(S) # 3x1
            I_mask = transition_2.dot(I) # 3x1

            # 無 舊 新 
            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            # 因為 interaction 改變狀態
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            # I 的改變
            dR = 𝛾 * (1-𝜆) * I_mask # 3x1 vector 因為 I 轉到 R 不管有沒有口罩都是相同
            dD = 𝛾 * 𝜆 * I_mask # 3x1 vector 因為 I 轉到 D 不管有沒有口罩都是相同

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)
            
            # 因為這期用過口罩，所以新口罩的狀態變成舊口罩，先定義轉換矩陣
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])

            # 人口轉換
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        elif T>=3:
            # 進入第三期開始不用管，大家都有口罩
            transition = np.array([[0,0,0],[0,0,0],[1,1,1]])
            S_mask = transition.dot(S) # 3x1
            I_mask = transition.dot(I) # 3x1

            matrix = np.outer([1,(1-δo),(1-δn)],[1,(1-σo),(1-σn)])
            𝛽0 = 𝛽 * matrix
            # 因為 interaction 改變狀態
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))
            # I 的改變
            dR = 𝛾 * (1-𝜆) * I_mask # 3x1 vector 因為 I 轉到 R 不管有沒有口罩都是相同
            dD = 𝛾 * 𝜆 * I_mask # 3x1 vector 因為 I 轉到 D 不管有沒有口罩都是相同

            nS = S_mask - dS
            nI = I_mask + dS - dR - dD
            nR = R + sum(dR)
            nD = D + sum(dD)

            # 因為這期用過口罩，所以新口罩的狀態變成舊口罩，先定義轉換矩陣
            transition_mask = np.array([[1,1,0],[0,0,1],[0,0,0]])

            # 人口轉換
            nS = transition_mask.dot(nS)
            nI = transition_mask.dot(nI)

            S,I,R,D = nS,nI,nR,nD

        T=T+1
    
    return(D)

def pmix_func(x,m,I_0=0.05,vo=0.5):

    # 前兩期死亡人數
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = Threeperiod_SIRD(S=1-I_0,I=I_0,q_0=m,q1_0=0,q1_1=0)
    Nation.serveralupdates(2)
    d1_phi, d1_n = Nation.d1_0, Nation.d1_1

    # t=1 q1_0 去登記拿到機率， q1_1 不去登記拿到機率，分配先給登記的有多才給不登記的
    q1_0_coef=x*(1-d0)*(1-m)
    q1_1_coef=(1-x)*(1-d0)*(1-m)+(1-d0)*m
    q1_0 = min(m/q1_0_coef,1)
    q1_1 = (m-q1_0_coef)/q1_1_coef if q1_0==1 else 0

    # t=2 的兩種情況
    if q1_0<1:
        # 上一期口罩不滿足所有登記的人，會有 1. 有登記沒拿到 q2_00 ，2.有登記有拿到 q2_1，3.沒登記沒拿到 q2_10
        q2_00_coef=(1-q1_0)*(1-d0)*x*(1-d1_phi)*(1-m)
        q2_10_coef=(1-d0)*( (1-x)*(1-d1_phi)*(1-m)+(1-d1_n)*m )
        q2_01_coef=q1_0*(1-d0)*x*(1-d1_phi)*(1-m)

        # 分配順序是 3→1→2
        q2_10 = min(m/q2_10_coef,1)
        q2_00 = min((m-q2_10_coef)/q2_00_coef,1) if q2_10==1 else 0
        q2_11 = 1 if q2_00==1 else 0
        q2_01 = (m-q2_10_coef-q2_00_coef)/q2_01_coef if q2_11==1 else 0

    else:
        # 口罩滿足所有有登記的人，會有 1.有登記有拿到 q2_1  ，2.沒登記沒拿到 q2_10 ，3.沒登記有拿到 q2_1
        q2_10_coef=(1-q1_1)*(1-d0)*( (1-x)*(1-d1_phi)*(1-m) + (1-d1_n)*m )
        q2_11_coef= q1_1 *(1-d0)*( (1-x)*(1-d1_phi)*(1-m) + (1-d1_n)*m )
        q2_01_coef= x*(1-d1_phi)*(1-d0)*(1-m)

        # 分配順序是 2→3→1
        q2_10 = min(m/q2_10_coef,1)
        q2_00 = 1 if q2_10==1 else 0
        q2_11 = min((m-q2_10_coef)/q2_11_coef,1) if q2_00==1 else 0
        q2_01 = (m-q2_10_coef-q2_11_coef)/q2_01_coef if q2_11==1 else 0

    phi_sign = q1_0*(1+q2_01+(1-q2_01)*vo) + (1-q1_0)*q2_00
    phi_nsign= q1_1*(1+q2_11+(1-q2_11)*vo) + (1-q1_1)*q2_10

    return phi_sign-phi_nsign

def Mix_computeProb(x,y,m,I_0=0.05,vo=0.5):
    '''
        x 跟 y 分別代表 phi 跟 n 去登記的機率，然後要返回所有參數
    '''
    # 前兩期死亡人數
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = Threeperiod_SIRD(S=1-I_0,I=I_0,q_0=m,q1_0=0,q1_1=0)
    Nation.serveralupdates(2)
    d1_phi, d1_n = Nation.d1_0, Nation.d1_1

    # t=1 q1_0 去登記拿到機率， q1_1 不去登記拿到機率，分配先給登記的有多才給不登記的
    q1_0_coef=x*(1-d0)*(1-m)+y*(1-d0)*m
    q1_1_coef=(1-x)*(1-d0)*(1-m) + (1-y)*(1-d0)*m
    q1_0 = min(m/q1_0_coef,1)
    q1_1 = (m-q1_0_coef)/q1_1_coef if q1_0==1 else 0

    # t=2 的兩種情況
    if q1_0<1:
        # 上一期口罩不滿足所有登記的人，會有 1. 有登記沒拿到 q2_00 ，2.有登記有拿到 q2_1，3.沒登記沒拿到 q2_10
        q2_00_coef=(1-q1_0)*(1-d0)*( x*(1-d1_phi)*(1-m)+y*(1-d1_n)*m )
        q2_10_coef=(1-d0)*( (1-x)*(1-d1_phi)*(1-m)+(1-y)*(1-d1_n)*m )
        q2_01_coef=q1_0*(1-d0)*( x*(1-d1_phi)*(1-m)+y*(1-d1_n)*m )

        # 分配順序是 3→1→2
        q2_10 = min(m/q2_10_coef,1)
        q2_00 = min((m-q2_10_coef)/q2_00_coef,1) if q2_10==1 else 0
        q2_11 = 1 if q2_00==1 else 0
        q2_01 = (m-q2_10_coef-q2_00_coef)/q2_01_coef if q2_11==1 else 0

    else:
        # 口罩滿足所有有登記的人，會有 1.有登記有拿到 q2_1  ，2.沒登記沒拿到 q2_10 ，3.沒登記有拿到 q2_1
        q2_10_coef=(1-q1_1)*(1-d0)*( (1-x)*(1-d1_phi)*(1-m)+(1-y)*(1-d1_n)*m )
        q2_11_coef= q1_1 *(1-d0)*( (1-x)*(1-d1_phi)*(1-m)+(1-y)*(1-d1_n)*m )
        q2_01_coef= (1-d0)*( x*(1-d1_phi)*(1-m)+y*(1-d1_n)*m )

        # 分配順序是 2→3→1
        q2_10 = min(m/q2_10_coef,1)
        q2_00 = 1 if q2_10==1 else 0
        q2_11 = min((m-q2_10_coef)/q2_11_coef,1) if q2_00==1 else 0
        q2_01 = (m-q2_10_coef-q2_11_coef)/q2_01_coef if q2_11==1 else 0
    
    return x,y,q1_0,q1_1,q2_10,q2_00,q2_11,q2_01,m

def pmix_func_growth(x,m,I_0=0.05,vo=0.5,growth='hop'):

    # 前兩期死亡人數
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = Threeperiod_SIRD(S=1-I_0,I=I_0,q_0=m,q1_0=0,q1_1=0)
    Nation.serveralupdates(2)
    d1_phi, d1_n = Nation.d1_0, Nation.d1_1

    # 假設第三期口罩就是 1 的線性成長
    r = (1-m)/3
    if growth=='linear':
        m0,m1,m2=m,m+r,m+2*r
    elif growth=='hop':
        m0,m1,m2=m,m,1.15*m

    # t=1 q1_0 去登記拿到機率， q1_1 不去登記拿到機率，分配先給登記的有多才給不登記的
    q1_0_coef=x*(1-d0)*(1-m0)
    q1_1_coef=(1-x)*(1-d0)*(1-m0)+(1-d0)*m0
    q1_0 = min(m1/q1_0_coef,1)
    q1_1 = (m1-q1_0_coef)/q1_1_coef if q1_0==1 else 0

    # t=2 的兩種情況
    if q1_0<1:
        # 上一期口罩不滿足所有登記的人，會有 1. 有登記沒拿到 q2_00 ，2.有登記有拿到 q2_1，3.沒登記沒拿到 q2_10
        q2_00_coef=(1-q1_0)*(1-d0)*x*(1-d1_phi)*(1-m0)
        q2_10_coef=(1-d0)*( (1-x)*(1-d1_phi)*(1-m0)+(1-d1_n)*m0 )
        q2_01_coef=q1_0*(1-d0)*x*(1-d1_phi)*(1-m0)

        # 分配順序是 3→1→2
        q2_10 = min(m2/q2_10_coef,1)
        q2_00 = min((m2-q2_10_coef)/q2_00_coef,1) if q2_10==1 else 0
        q2_11 = 1 if q2_00==1 else 0
        q2_01 = (m2-q2_10_coef-q2_00_coef)/q2_01_coef if q2_11==1 else 0

    else:
        # 口罩滿足所有有登記的人，會有 1.有登記有拿到 q2_1  ，2.沒登記沒拿到 q2_10 ，3.沒登記有拿到 q2_1
        q2_10_coef=(1-q1_1)*(1-d0)*( (1-x)*(1-d1_phi)*(1-m0) + (1-d1_n)*m0 )
        q2_11_coef= q1_1 *(1-d0)*( (1-x)*(1-d1_phi)*(1-m0) + (1-d1_n)*m0 )
        q2_01_coef= x*(1-d1_phi)*(1-d0)*(1-m0)

        # 分配順序是 2→3→1
        q2_10 = min(m2/q2_10_coef,1)
        q2_00 = 1 if q2_10==1 else 0
        q2_11 = min((m2-q2_10_coef)/q2_11_coef,1) if q2_00==1 else 0
        q2_01 = (m2-q2_10_coef-q2_11_coef)/q2_01_coef if q2_11==1 else 0

    phi_sign = q1_0*(1+q2_01+(1-q2_01)*vo) + (1-q1_0)*q2_00
    phi_nsign= q1_1*(1+q2_11+(1-q2_11)*vo) + (1-q1_1)*q2_10

    return phi_sign-phi_nsign





def Mix_computeProb_growth(x,y,m,I_0=0.05,vo=0.5,growth='hop'):
    '''
        x 跟 y 分別代表 phi 跟 n 去登記的機率，然後要返回所有參數
    '''
    # 前兩期死亡人數
    d0 = (1-(17/18)**14) * 0.0138 * I_0
    Nation = Threeperiod_SIRD(S=1-I_0,I=I_0,q_0=m,q1_0=0,q1_1=0)
    Nation.serveralupdates(2)
    d1_phi, d1_n = Nation.d1_0, Nation.d1_1

    # 假設第三期口罩就是 1 的線性成長
    r = (1-m)/3
    if growth=='linear':
        m0,m1,m2=m,m+r,m+2*r
    elif growth=='hop':
        m0,m1,m2=m,m,1.15*m

    # t=1 q1_0 去登記拿到機率， q1_1 不去登記拿到機率，分配先給登記的有多才給不登記的
    q1_0_coef=x*(1-d0)*(1-m0)+y*(1-d0)*m0
    q1_1_coef=(1-x)*(1-d0)*(1-m0)+(1-y)*(1-d0)*m0
    q1_0 = min(m1/q1_0_coef,1)
    q1_1 = (m1-q1_0_coef)/q1_1_coef if q1_0==1 else 0

    # t=2 的兩種情況
    if q1_0<1:
        # 上一期口罩不滿足所有登記的人，會有 1. 有登記沒拿到 q2_00 ，2.有登記有拿到 q2_1，3.沒登記沒拿到 q2_10
        q2_00_coef=(1-q1_0)*(1-d0)*( x*(1-d1_phi)*(1-m0)+y*(1-d1_n)*m0 )
        q2_10_coef=(1-d0)*( (1-x)*(1-d1_phi)*(1-m0)+(1-y)*(1-d1_n)*m0 )
        q2_01_coef=q1_0*(1-d0)*( x*(1-d1_phi)*(1-m0)+y*(1-d1_n)*m0 )

        # 分配順序是 3→1→2
        q2_10 = min(m2/q2_10_coef,1)
        q2_00 = min((m2-q2_10_coef)/q2_00_coef,1) if q2_10==1 else 0
        q2_11 = 1 if q2_00==1 else 0
        q2_01 = (m2-q2_10_coef-q2_00_coef)/q2_01_coef if q2_11==1 else 0

    else:
        # 口罩滿足所有有登記的人，會有 1.有登記有拿到 q2_1  ，2.沒登記沒拿到 q2_10 ，3.沒登記有拿到 q2_1
        q2_10_coef=(1-q1_1)*(1-d0)*( (1-x)*(1-d1_phi)*(1-m0) + (1-y)*(1-d1_n)*m0 )
        q2_11_coef= q1_1 *(1-d0)*( (1-x)*(1-d1_phi)*(1-m0) + (1-y)*(1-d1_n)*m0 )
        q2_01_coef= (1-d0)*( x*(1-d1_phi)*(1-m0)+y*(1-d1_n)*m0 )

        # 分配順序是 2→3→1
        q2_10 = min(m2/q2_10_coef,1)
        q2_00 = 1 if q2_10==1 else 0
        q2_11 = min((m2-q2_10_coef)/q2_11_coef,1) if q2_00==1 else 0
        q2_01 = (m2-q2_10_coef-q2_11_coef)/q2_01_coef if q2_11==1 else 0
    
    return x,y,q1_0,q1_1,q2_10,q2_00,q2_11,q2_01,m0,m1,m2
