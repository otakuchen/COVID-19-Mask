from functools import partial
import multiprocessing
import numpy as np
import pandas as pd
import time

def main():
    data = simdata(R=40,mpattern='nogrowth')
    data.to_csv("0728 SB R40.csv",index=False)
    data3 = simdata(R=40,mpattern='hop')
    data3.to_csv("0728 SB mm115mR40.csv",index=False)

    data4 = simdata(R=40,IC=False,mpattern='nogrowth')
    data4.to_csv("0728 FB R40.csv",index=False)
    data6 = simdata(R=40,IC=False,mpattern='hop')
    data6.to_csv("0728 FB mm115mR40.csv",index=False)



def simdata(T=40,R=40,IC=True,I=0.01,mpattern='nogrowth'):
    m_vals = np.linspace(0.1,0.8,T)

    #m_vals=np.linspace(0.4,0.6,20)
    #m_vals=np.linspace(0.15,0.75,3)
    # m_vals=np.array([0.2,0.4,0.5,0.7])
    result=map(lambda x: grid_search_vec_growth(m=x,R=R,I_0=I,IC_constrain=IC,vo=0.5,mpattern=mpattern),m_vals)
    result=list(result)

    data=pd.DataFrame.from_records(result, columns =['func', 'q1_0','q1_1','q2_00','q2_01','q2_10','q2_11','fircon','seccon','ICphi','ICn'])
    data['m_vals']=m_vals
    data=data.rename(columns={'func':'D_vals'})
    
    return data




def grid_search_vec_growth(m=0.1,I_0=0.05,R=40,IC_constrain=True,vo=0.5,mpattern='nogrowth'):
    
    # 假設第三期口罩就是 1
    r = (1-m)/3
    if mpattern=='nogrowth':
        m0,m1,m2=m,m,m
    elif mpattern=='linear':
        m0,m1,m2=m,m+r,m+2*r
    elif mpattern=='hop':
        m0,m1,m2=m,m,1.15*m

    # 第零期死亡人數被 I_0 固定，第一期死亡人數決定 m 之後就固定
    d0 = (1-(17/18)**14) * 0.0138 * I_0

    Nation = Threeperiod_SIRD(S=1-I_0,I=I_0,q_0=m0,q1_0=0,q1_1=0)
    Nation.serveralupdates(2)
    d1_0, d1_1 = Nation.d1_0, Nation.d1_1

    # 下面開始決定參數的範圍
    q1_0_ub = min( m1/((1-d0)*(1-m0)),1)
    q1_0_vals = np.linspace(0,q1_0_ub , R) ## Rx1 

    q1_1_ub = np.minimum(m1/((1-d0)*m0) - (1-m0)/m0*q1_0_vals,1)
    q1_1_vals = np.linspace(0,q1_1_ub, R)  ## RxR

    q2_00_coef = (1-d1_0)*(1-m0)*(1-q1_0_vals) ## Rx1
    q2_01_coef = (1-d1_0)*(1-m0)*q1_0_vals     ## Rx1
    q2_10_coef = (1-d1_1)*   m0*(1-q1_1_vals)  ## RxR
    q2_11_coef = (1-d1_1)*   m0*q1_1_vals      ## RxR
    tot_mask2 = (m1+m2)/(1-d0) - (1-m0)*q1_0_vals - m0*q1_1_vals
    # 第一期拿的不夠的那些 q1_0, q1_1 會使得後面數值都會變成 nan
    tot_mask2 = np.where(tot_mask2-q2_00_coef-q2_01_coef-q2_10_coef-q2_11_coef>0,np.nan,tot_mask2 )

    # 從這邊開始會發現上界的值有可能會小於零 (因為分子數值上是很小但不是 0，但是分母為零)，所以要控制
    # 另外這邊開始如果係數等於零 (沒有這個族群)，那任何值都沒影響，所以統一改成都是 0
    q2_00_ub = np.where(q1_0_vals==1,0,np.minimum(tot_mask2/q2_00_coef,1))
    q2_00_vals = np.linspace(0,q2_00_ub,R) ## RxRxR
    ## q2_00 拿的不夠就直接變成 nan
    q2_00_vals = np.where(tot_mask2 - q2_00_coef*q2_00_vals>q2_01_coef+q2_10_coef+q2_11_coef,
                          np.nan,
                          q2_00_vals)


    q2_01_ub = np.where(q1_0_vals==0,0,np.minimum(np.maximum((tot_mask2 - q2_00_coef*q2_00_vals)/q2_01_coef,0),1))
    q2_01_vals = np.linspace(0,q2_01_ub,R) ## RxRxRxR
    ## q2_01 拿的不夠就直接變成 nan
    q2_01_vals = np.where(tot_mask2 - q2_00_coef*q2_00_vals - q2_01_coef*q2_01_vals>q2_10_coef+q2_11_coef,
                          np.nan,
                          q2_01_vals)


    q2_10_ub = np.where(q1_1_vals==1,0,np.minimum(np.maximum((tot_mask2 - q2_00_coef*q2_00_vals - q2_01_coef*q2_01_vals)/q2_10_coef,0),1))
    q2_10_vals = np.linspace(0,q2_10_ub,R) ## RxRxRxRxR
    ## q2_10 拿的不夠就直接變成 nan
    q2_10_vals = np.where(tot_mask2 - q2_00_coef*q2_00_vals - q2_01_coef*q2_01_vals-q2_10_coef*q2_10_vals>q2_11_coef,
                          np.nan,
                          q2_10_vals)

    # 前面的範圍都控制住了，q2_11 的腳色是要維持滿足限制式，但有可能值小於 0 或 大於 1，這些就是不合法的點
    q2_11_vals = np.where(q1_1_vals==0,0,(tot_mask2 - q2_00_coef*q2_00_vals - q2_01_coef*q2_01_vals -  q2_10_coef*q2_10_vals)/q2_11_coef)


    
    #-------------------------------------------------#
    #                 IC constrain                    #
    #-------------------------------------------------#
    # 如果不考慮的話，就是假設政府有完全資訊去決定最適分配；反之，則是考慮不完全資訊下的分配
    if IC_constrain:
        phi_cons = q1_0_vals*(1+q2_01_vals+(1-q2_01_vals)*vo) + (1-q1_0_vals)*q2_00_vals >= q1_1_vals*(1+q2_11_vals+(1-q2_11_vals)*vo) + (1-q1_1_vals)*q2_10_vals
        old_cons = q1_1_vals*(1+q2_11_vals+(1-q2_11_vals)*vo) + (1-q1_1_vals)*(vo+q2_10_vals) >= q1_0_vals*(1+q2_01_vals+(1-q2_01_vals)*vo) + (1-q1_0_vals)*(vo+q2_00_vals)
        q2_11_vals = np.where(phi_cons & old_cons,q2_11_vals,np.nan)

    q1_0_in = np.broadcast_to(q1_0_vals,(R,R,R,R,R))
    q1_1_in = np.broadcast_to(q1_1_vals,(R,R,R,R,R))
    q2_00_in = np.broadcast_to(q2_00_vals,(R,R,R,R,R))
    q2_01_in = np.broadcast_to(q2_01_vals,(R,R,R,R,R))

    # 下面就是將所有點帶進算死亡人數的函數，並且取最小值
    # 將不合法的點函數值是1，這邊會花很多時間算
    tempf = partial(evaluate_death,S=1-I_0, I=I_0, R=0, D=0, T=0, t=150, q_0=m0, σo=0.5, σn=0.7, δo=0.5,δn=0.7)
    p=multiprocessing.Pool(12)
    task = [*zip(q1_0_in,q1_1_in,q2_00_in,q2_01_in,q2_10_vals,q2_11_vals)]
    
    # 取最佳條件 (死最少人) 的時候，不合法的點函數值是 1
    temp=np.where((q2_11_vals<=1) & (~np.isnan(q2_11_vals)) & (q2_11_vals>=0),
                p.starmap(tempf,iterable=task), ## 合法的取函數值
                1)
    ind = np.unravel_index(np.nanargmin(temp, axis=None), temp.shape)


    # 需要的資訊都記錄下來
    q1_0,q1_1,q2_00,q2_01,q2_10,q2_11 = q1_0_in[ind],q1_1_in[ind],q2_00_in[ind],q2_01_in[ind],q2_10_vals[ind],q2_11_vals[ind]
    
    firstcon = m1-(1-d0)*(1-m0)*q1_0-(1-d0)*m0*q1_1
    secondcon= (m1+m2-(1-d0)*(1-m0)*q1_0-(1-d0)*m0*q1_1-
                (1-d1_0)*(1-m0)*(1-q1_0)*(1-d0)*q2_00-
                (1-d1_0)*(1-m0)*q1_0*(1-d0)*q2_01-
                (1-d1_1)*m0*(1-q1_1)*(1-d0)*q2_10-
                (1-d1_1)*m0*q1_1*(1-d0)*q2_11)
    
    phi_sign = q1_0*(1+q2_01+(1-q2_01)*vo) + (1-q1_0)*q2_00
    phi_nsign= q1_1*(1+q2_11+(1-q2_11)*vo) + (1-q1_1)*q2_10
    n_sign   = q1_0*(1+q2_01+(1-q2_01)*vo) + (1-q1_0)*(vo+q2_00)
    n_nsign  = q1_1*(1+q2_11+(1-q2_11)*vo) + (1-q1_1)*(vo+q2_10)


    result = {"func":temp[ind],
              "q1_0":q1_0,
              "q1_1":q1_1,
              "q2_00":q2_00,
              "q2_01":q2_01,
              "q2_10":q2_10,
              "q2_11":q2_11,
              "fircon":firstcon,
              "seccon":secondcon,
              "ICphi":phi_sign -phi_nsign,
              "ICn"  :n_sign-n_nsign}
    return(result)

class Threeperiod_SIRD:
    
    def __init__(self,N=1,          # 人數，1 的話就是考慮社會有連續的 1 單位人，其他數字可以變成離散
                      S=0.9,        # susceptible
                      I=0.1,        # infected
                      R=0,          # recovered
                      D=0,          # died
                      𝛽=2.4/(18/14),# 基礎的 transmition rate，R0=2.4，從 I 移出是 18 天，一單位時間是兩周 14 天
                      𝛾=1-(17/18)**14,# 移出 I 的天數是 18 天，兩個禮拜就是 1 減掉 14 天都不移出的機率
                      λ=0.0138,     # 死亡率
                                     
                      
                      T=0,          # 模型的期數
                      q_0=0.2,      # 第 0 期發出去的口罩，也是口罩數 

                      σo=0.5,       # old facemask inward protection
                      σn=0.7,       # new facemask inward protection
                      δo=0.5,       # old facemask outward protection
                      δn=0.7,       # new facemask inward protection

                      q1_0=0.2,     # 第 1 期要發給第 0 期沒口罩的比例
                      q1_1=0.2,     # 第 1 期要發給第 0 期沒口罩的比例
                                    # 限制條件在 (1-q_0)*q1_0 + q_0*q1_1 <= m/(1-d_0)
                                    # or m/() - (1-q_0)*q1_0 + q_0*q1_1  = 負的口罩剩餘  

                      q2_00=0.2,      # 第 2 期要發給 (0,0) 沒口罩的比例
                      q2_10=0.2,      # 第 2 期要發給 (1,0) 有口罩的比例 
                      q2_01=0.2,      # 第 2 期要發給 (0,1) 有口罩的比例 
                      q2_11=0.2):     # 第 2 期要發給 (1,1) 有口罩的比例 
                                    # 限制條件在
                                    # (1-q_0)(1-q1_0)(1-d_0-d_1)q2_00 + q_0(1-q1_1)(1-d_0-d_1)q2_10 +
                                    # (1-q_0)   q1_0 (1-d_0-d_1)q2_01 + q_0   q1_1 (1-d_0-d_1)q2_11 =  (m + 上期的剩餘口罩)/(1-d_0-d_1)
                                         
        self.S    = np.array([N*S,0,0,0,0,0,0,0])
        self.I    = np.array([N*I,0,0,0,0,0,0,0])
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
            S_mask = transition_0.dot(self.S) # 8x1
            I_mask = transition_0.dot(self.I) # 8x1

            # 無 舊 新 新 新 無 舊 舊
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),(1-δn),1,(1-δo),(1-δo)],[1,(1-σo),(1-σn),(1-σn),(1-σn),1,(1-σo),(1-σo)])
            𝛽0 = 𝛽 * matrix
            # 因為 interaction 改變狀態
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        
        if T==1:
            ##### 這裡要先算有口罩跟沒口罩的死亡人數，作為之後限制式要用
            self.d1_0, self.d1_1 = 𝜆 * self.I[0], 𝜆 * self.I[1]

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

            # 無 舊 新 新 新 無 新 新
            matrix = np.outer([1,(1-δo),(1-δn),(1-δn),(1-δn),1,(1-δo),(1-δo)],[1,(1-σo),(1-σn),(1-σn),(1-σn),1,(1-σo),(1-σo)])
            𝛽0 = 𝛽 * matrix
            # 因為 interaction 改變狀態
            dS = np.diag(np.outer(S_mask,I_mask).dot(𝛽0))

        elif T==2:
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

if __name__ =="__main__":
    main()
