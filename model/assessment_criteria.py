import pandas as pd
import numpy as np
import heapq
from collections import Counter

class assessment_criteria:
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
       
    def recall_score(self,value_N):#TP(TP + FN)
        np_gain_value = np.array(self.gain_value.iloc[:, :-1])
        lists = self.top_n(np_gain_value, value_N)
        feature_topN = []
        k = 0
        for i in lists:  
            m = 0
            temp = []
            for j in i:
                temp.append(self.gain_value.columns[j])
                m += 1
            feature_topN.append(temp)
            k += 1

        result = Counter(self.del_column)
        recall_list=[]
        Weight_recall=[]
        for j in range(len(self.gain_value.columns)-1):
            TP=0
            feature=self.gain_value.columns[j] 
            weight=result[feature]/len(self.del_column)
            Weight_recall.append(weight)
            for i in range(len(feature_topN)):
                if feature == self.del_column[i]:
                    if self.del_column[i] in feature_topN[i]:
                        TP += 1
            if result[feature]!=0:
                recall=TP/result[feature]*100
            else:
                recall=0
            recall_list.append(recall)
        self.recall_list=recall_list
        self.Weight_recall=Weight_recall
        return recall_list,Weight_recall
    
    def Recall_DataFrame(self,value_N):
        if self.gain_value.columns[0]=='Unnamed: 0':
            self.gain_value=self.gain_value.drop('Unnamed: 0',axis=1)
        recall_df=pd.DataFrame(index=range(1,value_N+1),columns=self.gain_value.columns[:-1])
        recall_df['weighted_recall']=''
        for i in range(1,value_N+1):
            recall_list ,Weight_recall=self.recall_score(i)
            recall_df.loc[i,:-1]=recall_list
            recall_df.loc[i]['weighted_recall']=float(sum(map(lambda x,y:x*y,recall_list,Weight_recall)))
        self.recall_df=recall_df
        return recall_df
    
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
            