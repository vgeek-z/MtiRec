import pandas as pd
import numpy as np
import heapq
from collections import Counter

class assessment_criteria:
    """
        del only one AVP
    """
    def __init__(self,gain_value,del_column):
        self.gain_value=gain_value
        self.del_column=del_column
        
        #accuracy
        self.feature_topN=None
        self.Accuracy=None
        self.Accuracy_list=None
        
        #MRR
        self.MRR=None
        self.MRR_list=None
        
        #Recall
        self.recall_list=None
        self.Weight_recall=None
        
        self.recall_df=None
        
        pass
    
    def top_n(self,rec_list, n):
        lists = []
        for i in range(len(rec_list)):
            lists.append(heapq.nlargest(n, range(len(rec_list[i])), rec_list[i].take))
        return lists
    
    def count_Accuracy(self,value_N):
        if self.gain_value.columns[0]=='Unnamed: 0':
            self.gain_value=self.gain_value.drop('Unnamed: 0',axis=1)
        np_gain_value=np.array(self.gain_value.iloc[:,:-1])
        lists = self.top_n(np_gain_value, value_N)
        feature_topN=[]
        k=0
        for i in lists:
            temp=[]
            for j in i:
                temp.append(self.gain_value.columns[j])
            feature_topN.append(temp)
            k+=1
        count=0.0
        for i in range(len(feature_topN)):
            if self.del_column[i] in feature_topN[i]:
                count+=1
        Accuracy=count/self.gain_value.shape[0]*100
        self.feature_topN=feature_topN
        self.Accuracy=Accuracy
        return feature_topN,Accuracy        
    
    def MRR_MeanReciprocal(self,value_N):
        #value: top 1,top3,top value
        #gain_value: the final result /ranked
        #del_column:the real answer
        if self.gain_value.columns[0]=='Unnamed: 0':
            self.gain_value=self.gain_value.drop('Unnamed: 0',axis=1)
        feature_topN,Accuracy=self.count_Accuracy(value_N)
        MRR=0.0
        for i in range(len(self.del_column)):
            for j in range(value_N):
                if self.del_column[i]==feature_topN[i][j]:
                    MRR+=1/(j+1)
                    break
        MRR=MRR/len(self.del_column)*100    
        self.MRR=MRR    
        return MRR
    
    def Accuracy_List(self,value_N):
        Accuracy_list=[]
        for i in range(1,value_N+1):
            feature_topN,Accuracy=self.count_Accuracy(i)
            print('Accuracy_List:当前N的取值为：',i,'精确度为：',Accuracy)
            Accuracy_list.append(Accuracy)
        self.Accuracy_list=Accuracy_list
        return Accuracy_list
    
    def MRR_List(self,value_N):
        MRR_list=[]
        for i in range(1,value_N+1):
            mrr=self.MRR_MeanReciprocal(i)
            MRR_list.append(mrr)
            print('MRR_List:当前N的取值为：',i,'MRR:',mrr)
        self.MRR_list=MRR_list
        return MRR_list
    
class assessment_criteria_del_list:
    """
    这个标准是面向 删除元素为 列表 形式存在的测试集 的评价指标
    也就是说 每个患者删除的AVP不止一个 因此每个患者删除的元素以列表形式存在
    """
    def __init__(self,gain_value,del_column):
        self.gain_value=gain_value
        self.del_column=del_column
        
        #accuracy
        self.feature_topN=None
        self.Accuracy=None
        self.Accuracy_list=None
        
        #MRR
        self.MRR=None
        self.MRR_list=None
        
        #Recall
        self.recall=None
        self.Recall_list = None
        
        
        #self.recall_df=None#获得从0-value n的top N 的表
        
        pass
    
    def top_n(self,rec_list, n):
        '''
        :param rec_list:
        :param n:
        :return:
        '''
        lists = []
        for i in range(len(rec_list)):
            # a = np.argpartition(my_array[i],-3)[-3:]
            lists.append(heapq.nlargest(n, range(len(rec_list[i])), rec_list[i].take))
            # lists[i] = np.append(a)
        return lists
    
    def count_Accuracy(self,value_N):
        if self.gain_value.columns[0]=='Unnamed: 0':
            self.gain_value=self.gain_value.drop('Unnamed: 0',axis=1)
        np_gain_value=np.array(self.gain_value.iloc[:,:-1])
        lists = self.top_n(np_gain_value, value_N)
        feature_topN=[]
        k=0
        for i in lists:#对于每一个用户前top n的下标
            temp=[]
            for j in i:
                temp.append(self.gain_value.columns[j])
            feature_topN.append(temp)
            k+=1
        count=0.0
        for i in range(len(feature_topN)): #对于每一个用户
            for j in self.del_column[i]:# 被删掉的属性 现在是j
                if j in feature_topN[i]: # 
                    count+=1
                    break
        Accuracy=count/self.gain_value.shape[0]*100
        self.feature_topN=feature_topN
        self.Accuracy=Accuracy
        return feature_topN,Accuracy        
    
    def MRR_MeanReciprocal(self,value_N):
        #value: top 1,top3,top value
        #gain_value: the final result /ranked
        #del_column:the real answer
        if self.gain_value.columns[0]=='Unnamed: 0':
            self.gain_value=self.gain_value.drop('Unnamed: 0',axis=1)
        feature_topN,Accuracy=self.count_Accuracy(value_N)
        MRR=0.0
        
        for i in range(len(self.del_column)):# 对于每一个患者
            single_mrr =0.0
            single_count=0
            for MTI_item in self.del_column[i]:
                for j in range(value_N):# 相当于K
                    if MTI_item ==feature_topN[i][j]:
                        single_mrr+=1/(j+1)
                        single_count +=1
                        break
            if single_count == 0:
                single_mrr =0
            else:
                single_mrr = single_mrr/len(self.del_column[i])
            MRR+=single_mrr
        MRR=MRR/len(self.del_column)*100    
        self.MRR=MRR    
        return MRR
       
    def recall_score(self,value_N):#TP(TP + FN)
        #真阳性（TP）: 预测为正， 实际也为正    假阳性（FP）: 预测为正， 实际为负  假阴性（FN）: 预测为负，实际为正  真阴性（TN）: 预测为负， 实际也为负
        np_gain_value = np.array(self.gain_value.iloc[:, :-1])
        lists = self.top_n(np_gain_value, value_N)
        feature_topN = []
        #k = 0
        for i in lists:
            #m = 0
            temp = [] 
            for j in i:
                temp.append(self.gain_value.columns[j])
                #m += 1
            feature_topN.append(temp)
            #k += 1            # print(feature_topN) #[['age','weight'...],['age1','age2','age3'...],[]]
        
        count =0
        for i in range(len(feature_topN)): 
            count_single_patient =0
            for j in self.del_column[i]:
                if j in feature_topN[i]: 
                    count_single_patient+=1
            score_single_patient =  float(count_single_patient/len(self.del_column[i]))
            count+=score_single_patient
        recall =count/self.gain_value.shape[0]*100
        self.recall=recall
        return recall
    
    def Recall_List(self,value_N):
        Recall_list=[]
        for i in range(1,value_N+1):
            recall=self.recall_score(i)
            Recall_list.append(recall)
            print('Recall_list:当前N的取值为：',i,'Recall:',recall)
        self.Recall_list=Recall_list
        return Recall_list
    
    def Accuracy_List(self,value_N):
        Accuracy_list=[]
        for i in range(1,value_N+1):
            feature_topN,Accuracy=self.count_Accuracy(i)
            print('Accuracy_List:当前N的取值为：',i,'精确度为：',Accuracy)
            Accuracy_list.append(Accuracy)
        self.Accuracy_list=Accuracy_list
        return Accuracy_list
    
    def MRR_List(self,value_N):
        MRR_list=[]
        for i in range(1,value_N+1):
            mrr=self.MRR_MeanReciprocal(i)
            MRR_list.append(mrr)
            print('MRR_List:当前N的取值为：',i,'MRR:',mrr)
        self.MRR_list=MRR_list
        return MRR_list    
    
class Frame_assessment:
    def __init__(self,del_column):
        self.del_column=del_column
        pass
    
    def count_Accuracy(self,feature_topN,value_N):
        count=0.0
        for i in range(len(feature_topN)): 
            if self.del_column[i] in feature_topN[i][:value_N]:
                count+=1
        Accuracy=count/len(self.del_column)*100
        return Accuracy
    
    def mrr(self,feature_topN,value_N):
        MRR=0.0
        for i in range(len(self.del_column)):
            for j in range(value_N):
                if self.del_column[i]==feature_topN[i][j]:
                    MRR+=1/(j+1)
                    break
        MRR=MRR/len(self.del_column)*100    
        return MRR
    def recall(self,test_stat_df,feature_topN,value_N):
        result = Counter(self.del_column)
        recall_df=pd.DataFrame(index=range(1,value_N+1),columns=test_stat_df.columns)
        recall_df['weighted_recall']=''
        for k in range(1,value_N+1):
            recall_list=[]
            Weight_recall=[]
            for j in range(len(test_stat_df.columns)):
                TP=0
                FN=0
                feature=test_stat_df.columns[j] 
                weight=result[feature]/len(self.del_column)
                Weight_recall.append(weight)
                for i in range(len(feature_topN)):
                    if feature ==self.del_column[i]: 
                        if self.del_column[i] in feature_topN[i][:k]:
                            TP += 1
                        else:
                            FN +=1
                print('feature,weight,TP:',feature,weight,TP)
                if result[feature]!=0:
                    recall=TP/(TP+FN)*100
                else:
                    recall=0
                recall_list.append(recall)
            
            recall_df.loc[k,:-1]=recall_list
            recall_df.loc[k]['weighted_recall']=float(sum(map(lambda x,y:x*y,recall_list,Weight_recall)))        
        return recall_df
            