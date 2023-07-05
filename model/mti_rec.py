# This file primarily stores methods for obtaining group information, matching degree, and gain value, which are applicable to non-DHTM methods. 
# It facilitates the replacement of topic models in the MtiRec framework.
from cProfile import label
from cmath import isnan
import enum
from itertools import count
import math
import pandas as pd
import numpy as np
import random


class Mti_Recommend():
    def __init__(self,labeled_documents=None,terms=None,label=None,enum=None,text=None):
        if labeled_documents==None:
            print('--WARNING--:There is no processed document.')
        self.labeled_documents=labeled_documents
        self.label=label
        self.terms=terms
        self.enum=enum
        self.text=text
       
       
    def None2NaN(self,x):
        if x == '':
            x = np.nan
        return x    
    

    
    def get_therapy_2(self,start_id,end_id):
        """得到所有 # # 现存的 # # 二元组组合。
        
        Parameters
        ----------
        我们设定标签数量都只考虑4个，id取值{0,1,2,3}
        start_id: label列表中第start_id个标签
        end_id: label列表中第end_id个标签
        返回值: group 治疗方案组合为元组形式 的列表
        """
        group=[]
        for i in range(len(self.labeled_documents)):
            if math.isnan(self.labeled_documents[i][1][start_id])or math.isnan(self.labeled_documents[i][1][end_id]):
                continue
            else:
                group.append(tuple((self.labeled_documents[i][1][start_id],self.labeled_documents[i][1][end_id])))
        group=list(set(group))
        print('当前我们进行的label二元组合为：',self.label[start_id],self.label[end_id],'前五（例）为：',group[:5])
        return group

    
    def get_therapy_3(self,id1,id2,id3):
        """
        得到所有 # # 现存的 # # 三元组组合。
        
        Parameters
        ----------
        我们设定标签数量都只考虑4个，id取值{0,1,2,3}
        id1: label列表中第id1个标签
        id2:label列表中第id2个标签
        id3:label列表中第id3个标签
    
        返回值:group 治疗方案组合为元组形式 的列表
        """
        group=[]
        for i in range(len(self.labeled_documents)):
            if math.isnan(self.labeled_documents[i][1][id1])or math.isnan(self.labeled_documents[i][1][id2]) or math.isnan(self.labeled_documents[i][1][id3]):
                continue
            else:
                group.append(tuple((self.labeled_documents[i][1][id1],self.labeled_documents[i][1][id2],self.labeled_documents[i][1][id3])))#从1开始是去掉了年份
        group=list(set(group))
        print('当前我们进行的label三元组合为：',self.label[id1],self.label[id2],self.label[id3],'前五（例）为：',group[:5])
        return group
    
    def get_therapy_4(self):
        group_4=[]
        for i in range(len(self.labeled_documents)):
            count=0
            for j in self.labeled_documents[i][1][0:4]:
                if math.isnan(j):
                    count+=1
            if count>0:
                continue
            else:
                group_4.append(tuple(self.labeled_documents[i][1][0:4]))
                
        group_4=list(set(group_4))

        return group_4

    def sub3_get_therapy2term(self,group_N,id1,id2,id3):
        """
            Obtain the therapy conbination (binary) or (ternary) group and word-based statistical probability distribution.

            Statistical method: Calculate the occurrence count (co-occurrence count) of each word under this therapy,
            and then divide it by the total number of documents (probability).

            To facilitate numerical calculations, we multiply the above numbers by 1000 to obtain the probability in parts per thousand.
            
            Returns
            -------
            therapy2term: dataframe,index=group_N ,columns=terms.
        """
        AA=np.zeros((len(group_N),len(self.terms)))
        therapy2term=pd.DataFrame(AA,index=group_N,columns=self.terms)
        if id3==0:#二元组
            for i in range(len(self.labeled_documents)): 
                if math.isnan(self.labeled_documents[i][1][id1]) or math.isnan(self.labeled_documents[i][1][id2]):
                    continue
                else:
                    templabel=tuple((self.labeled_documents[i][1][id1],self.labeled_documents[i][1][id2]))
                    k=group_N.index(templabel)
                    for j in self.labeled_documents[i][0].split('##'):
                        m=self.terms.index(j)
                        therapy2term.iloc[k,m]+=1
        else:
            for i in range(len(self.labeled_documents)): 
                if math.isnan(self.labeled_documents[i][1][id1]) or math.isnan(self.labeled_documents[i][1][id2]) or math.isnan(self.labeled_documents[i][1][id3]):
                    continue
                else:
                    templabel=tuple((self.labeled_documents[i][1][id1],self.labeled_documents[i][1][id2],self.labeled_documents[i][1][id3]))
                    k=group_N.index(templabel)
                    for j in self.labeled_documents[i][0].split('##'): 
                        m=self.terms.index(j)   
                        therapy2term.iloc[k,m]+=1
        therapy2term=therapy2term.div(len(self.labeled_documents)).multiply(1000)
        for i in range(len(group_N)):
            group_N[i]=str(group_N[i])
        therapy2term.index=group_N
        return therapy2term

    def sub4_get_therapy2term(self,group_N):
        """
            Returns
            -------
            therapy2term: dataframe,index=group_N, columns=terms.
        """
        AA=np.zeros((len(group_N),len(self.terms)))
        therapy2term=pd.DataFrame(AA,index=group_N,columns=self.terms)

        for i in range(len(self.labeled_documents)): #对于每一个患者
            if tuple(self.labeled_documents[i][1][0:4]) in group_N:##找到当前患者的therapy的index =k
                k=group_N.index(tuple(self.labeled_documents[i][1][0:4]))
            for j in self.labeled_documents[i][0].split('##'): #对于这个患者的每一个单词
                m=self.terms.index(j)
                therapy2term.iloc[k,m]+=1
        therapy2term=therapy2term.div(len(self.labeled_documents)).multiply(1000)
        #print(therapy2term)
        for i in range(len(group_N)):
            group_N[i]=str(group_N[i])
        therapy2term.index=group_N
        return therapy2term

    
    def get_therapy2term(self,M,id1,id2,id3):
        """
        Applicable to the popular method for treatment plan-word probability distribution.

        Other machine learning methods cannot invoke this function to obtain the probability distribution of therapy-term.

        Pop method: Calculate the occurrence count (co-occurrence count) of each word under this treatment plan, and then divide it by the total number of documents (probability).

        To facilitate numerical calculations, we multiply the above numbers by 1000 to obtain the probability in parts per thousand.
        
        Parameters
        ----------
        M:{1,2,3,4} 代表M元组
        M=1: id1代表label列表中第id1个标签类别组成的集合列表
        M=2: id1,id2 代表label列表中第id1、id2个标签类别组成的集合列表
        M=3: id1,id2,id3 代表label列表中第id1、id2、id3个标签类别组成的集合列表
        M=4: label的前4个标签类别下所有标签组成的集合 此时的id_i可以随便填
        """
        if M==1:
            group=[]
            for i in range(len(self.labeled_documents)):
                if math.isnan(self.labeled_documents[i][1][id1]):
                    continue
                else:
                    group.append(self.labeled_documents[i][1][id1])
            group=list(set(group))
            
            AA=np.zeros((len(group),len(self.terms)))
            therapy2term=pd.DataFrame(AA,index=group,columns=self.terms)
            
            for i in range(len(self.labeled_documents)): #对于每一个患者
                if math.isnan(self.labeled_documents[i][1][id1]):
                    continue
                else:
                    templabel=self.labeled_documents[i][1][id1]
                    k=group.index(templabel)
                    for j in self.labeled_documents[i][0].split('##'): #对于这个患者的每一个单词
                        m=self.terms.index(j)
                        #print('单词 j,单词j的 index:',j,m)
                        therapy2term.iloc[k,m]+=1
                    
            therapy2term=therapy2term.div(len(self.labeled_documents)).multiply(1000)
            
            for i in range(len(group)):
                group[i]=str(group[i])
            therapy2term.index=group
            
            group_temp=group

        if M==2:
            group_temp=self.get_therapy_2(id1,id2)
            print('当前的治疗方案组:',group_temp[:3])
            therapy2term=self.sub3_get_therapy2term(group_temp,id1,id2,0)
        elif M==3:
            group_temp=self.get_therapy_3(id1,id2,id3)
            print('当前的治疗方案组:',group_temp[:3])
            therapy2term=self.sub3_get_therapy2term(group_temp,id1,id2,id3)
        elif M==4:
            group_temp=self.get_therapy_4()
            print('当前的治疗方案组:',group_temp[:3])
            therapy2term=self.sub4_get_therapy2term(group_temp)
            
        return group_temp,therapy2term
    
    
    
    def get_topN_terms(self,N,therapy2term):
        """
        Find the top N features ranked by occurrence in each combination of therapies.
        
        Parameters
        ----------
        N: The baseline combination has N elements.
        therapy2term: Probability distribution of therapies combinations and AVPs.
        
        Returns
        -------
        therapy_topN_terms:dataframe,index=therapy2term.index, N+1=range(N)+sum(p)
        
        """
        #BB=np.zeros((len(therapy2term),N))
        therapy_topN_terms=pd.DataFrame(index=therapy2term.index,columns=range(N))       
        for i in therapy2term.index:
            topN_terms=sorted(list(zip(self.terms,therapy2term.loc[i])) ,key=lambda x: x[1], reverse=True)
            therapy_topN_terms.loc[i]=topN_terms[:N]
        print('therapy_topNterms over!')
        sum_list = []
        for i in range(therapy_topN_terms.shape[0]):
            sum = 0
            for key, value in therapy_topN_terms.iloc[i, :therapy_topN_terms.shape[1]]:
                sum += value
            sum_list.append(sum)
        therapy_topN_terms['概率之和'] = sum_list
        return therapy_topN_terms
        
    
    
    def test_data_pre(self,test_stat_df):
        """
        Returns
        -----------
        test_stat_df:dataframe,
        del_column:list,存放被删除的属性名称（每个用户删除一个）
        """
        if 'Unnamed: 0' in test_stat_df.columns:
            test_stat_df=test_stat_df.drop(['Unnamed: 0'],axis=1)
        for i in self.label: 
            if i in test_stat_df.columns:
                test_stat_df = test_stat_df.drop([i], axis=1)  
                
        for i in self.text:  
            if i in test_stat_df.columns:
                test_stat_df = test_stat_df.drop([i], axis=1)
                
        test_stat_df = test_stat_df.applymap(self.None2NaN)  
        del_column = []
        print('test_stat_df.shape:',test_stat_df.shape)
        for i in range(test_stat_df.shape[0]):
            x = random.randint(0, test_stat_df.shape[1]-1)
            while pd.isna(test_stat_df.iloc[i,x]):
                x = random.randint(0, test_stat_df.shape[1]-1)            
            del_column.append(test_stat_df.columns[x])
            test_stat_df.iloc[i, x] = np.nan
        print('随机删除的内容有：',del_column[:10])
        return test_stat_df,del_column    
    
    def get_new_df(self,test_stat_df,therapy_topN_terms,weight_t1):
        """
        得到用户和治疗方案组合的匹配度。
        具体算法:
        如果用户i有治疗方案组合j基准集合中的单词t，用户和治疗方案的匹配度+上单词t在治疗方案中的概率，
        如果没有这个单词，但是有这个单词的类别（同属性不同值），匹配度减去这个同类别的单词的value。
        
        Parameters
        ----------
        test_stat_df:测试集
        therapy_topN_terms:治疗方案topN的单词基准集合
        
        Returns
        -------
        new_df:dataframe,index:用户编号,columns:治疗方案组合

        """
        test_stat_df = test_stat_df.applymap(self.None2NaN)
        AA = np.zeros((test_stat_df.shape[0], therapy_topN_terms.shape[0]))
        new_df = pd.DataFrame(AA, columns=therapy_topN_terms.index) 
        test_documents = []
        for user_index in range(test_stat_df.shape[0]):
            k=0
            list_user = []
            for feature in range(test_stat_df.shape[1]):
                if pd.notna(test_stat_df.iloc[user_index, feature]):
                    list_user.append('%s:%s'%(test_stat_df.columns[feature],test_stat_df.iloc[user_index, feature]))
            
            for therapy_index in range(therapy_topN_terms.shape[0]):
                temp=0
                for key, value in therapy_topN_terms.iloc[therapy_index, :therapy_topN_terms.shape[1] - 1]:
                    if key in list_user:
                        temp+=value
                    else:
                        for j in list_user:
                            if key.split(':')[0]==j.split(':')[0]:
                                if weight_t1 <0:
                                    temp-= weight_t1*value
                                    #temp-= abs(weight_t1)*value
                                    #new_df.iloc[user_index, therapy_index] -= abs(weight_t1)
                                else:
                                    temp-= weight_t1*value
                                    #new_df.iloc[user_index, therapy_index] -= weight_t1*value
                new_df.iloc[user_index, therapy_index] = temp

        new_df=new_df.div(therapy_topN_terms.iloc[:,therapy_topN_terms.shape[1]-1])
        for i in range(new_df.shape[0]):
            for j in range(new_df.shape[1]):
                if new_df.iloc[i,j]<0:
                    new_df.iloc[i,j]=0
        return new_df
        
    
    def get_Feature2value(self,terms,test_df):
        """
        得到特征和特征值的词袋
        #时间复杂度： O(feature_num  * terms_num) + O(feature_num * k)
        Returns
        -------
        feature_wordsbag:dataframe,index:test_df.columns(属性),columns:range(700):随意设置的较大数值  可以更改
        """
        feature_wordsbag=pd.DataFrame(index=test_df.columns,columns=range(700))
        temp=[]
        #group2feature
        for feature in range(len(test_df.columns)):
            k=0
            #print(test_df.columns[feature])
            for i in terms:
                if feature_wordsbag.index[feature] == i.split(':')[0]: 
                    feature_wordsbag.iloc[feature,k]=i
                    k+=1
            temp.append(k)
        return feature_wordsbag
        
    def get_group2feature(self,therapyGroups,therapygroup2term,test_df,feature_wordsbag):
        """
        得到治疗方案和 特征(属性 而非属性值)的概率关系
        计算方法:
        对于每一个治疗方案组i,我们对每一个特征f进行统计,我们统计这个特征f下所有的特征值a和治疗方案组i的概率之和。
        
        Returns
        -------
        group2feature:dataframe,
        index=therapyGroups(治疗方案组合),columns=test_df.columns(测试集的属性(已经去掉了label的意思))
        """
        AA=np.zeros((len(therapyGroups),len(test_df.columns)))
        group2feature=pd.DataFrame(AA,index=therapyGroups,columns=test_df.columns)
        list3=list(therapygroup2term.index)

        for i in range(len(therapyGroups)):#group2feature.index[i]=(1,3,5,8)#每一个therapy
            for j in range(group2feature.shape[1]):#group2feature.columns[j]='age'/'weight'#每一个特征
                temp=0
                for k in feature_wordsbag.loc[group2feature.columns[j]]:#对于每一个term
                    try:
                        if pd.notna(k):
                            m2=self.terms.index(k)    
                            temp+=therapygroup2term.iloc[i,m2]    
                        else:
                            break
                    except:
                        pass
                group2feature.iloc[i,j]=temp
        return group2feature
    
    
    def get_gain_value(self,train_stat_df,test_stat_df,new_df,therapygroup2term,feature_wordsbag):
        """
        计算增益值
        计算方法:
        对于每一个方案来说 , 用户i与方案j匹配度(new df)  乘以  方案j和特征f的概率值 。
        累加后就是用户i和特征f的增益值
        
        其中概率最大 的就是推荐的下一项 
        
        Returns
        -------
        gain_value:dataframe,index=用户编号,columns为特征名称,最后一列为'next_test'.
        """
        
        temp=np.zeros_like(test_stat_df)
        gain_value=pd.DataFrame(temp,index=test_stat_df.index,columns=test_stat_df.columns)
        
        user2term_array=np.dot(new_df.values,therapygroup2term)
        user2term_df=pd.DataFrame(user2term_array,index=new_df.index,columns=therapygroup2term.columns )
        
        user2feature_df=pd.DataFrame(temp,index=test_stat_df.index,columns=test_stat_df.columns)
        for user_id in user2term_df.index:
            if user_id % 500==0:
                print('——user_id:——',user_id)
            for f in user2feature_df.columns:
                temp=0
                if pd.notna(test_stat_df.loc[user_id,f]):
                    pass
                else:
                    for word in feature_wordsbag.loc[f,:]:
                        if word == None or pd.isna(word) :
                            break
                        else:
                            temp+=user2term_df.loc[user_id,word]
                user2feature_df.loc[user_id,f]=temp

        gain_value=user2feature_df

        """
            获得最推荐的下一项检查
        """
        gain_value['next_test']=0
        gain_value['next_test']=gain_value.astype(float).idxmax(axis=1)
        #print(gain_value)
        return gain_value

    
class Impersonality_Rec():
    def __init__(self,label,enum,text):
        self.label=label
        self.enum=enum
        self.text=text
         
    
    def test_data_pre_delGroup(self,test_stat_df,train_series):
        """
        删除测试集中所有的label集中的字段，将测试集中的空格也变为nan类型
        对于测试集中的每一个用户，随机删除一个属性，在这里要求所有当前特征中删除一个
        
        Returns
        -----------
        test_stat_df:dataframe,
        del_column:list,存放被删除的属性名称（每个用户删除n个）
        """
        if 'Unnamed: 0' in test_stat_df.columns:
            test_stat_df=test_stat_df.drop(['Unnamed: 0'],axis=1)
        for i in self.label: 
            if i in test_stat_df.columns:
                test_stat_df = test_stat_df.drop([i], axis=1)  
                
        for i in self.text:  
            if i in test_stat_df.columns:
                test_stat_df = test_stat_df.drop([i], axis=1)
                
        test_stat_df = test_stat_df.applymap(self.None2NaN)  
        
        del_column = []
        print('test_stat_df.shape:',test_stat_df.shape)
        
        del_index=[]
        for i in range(test_stat_df.shape[0]):
            del_user=[]
            del_user_index=[]
            x = random.randint(0, test_stat_df.shape[1]-1)
            try:
                while pd.isna(test_stat_df.iloc[i,x]):
                    x = x+1
            except:
                x = random.randint(0, test_stat_df.shape[1]-1)
                while pd.isna(test_stat_df.iloc[i,x]):
                    x=random.randint(0, test_stat_df.shape[1]-1)
            
            temp=train_series[test_stat_df.columns[x]]
            for i in range(len(train_series)):
                if train_series[i]==temp:
                    del_user.append(train_series.index[i])
                    
            columns_temp=list(test_stat_df.columns)
            for feature in del_user:
                x=columns_temp.index(feature)
                del_user_index.append(x)
            del_column.append(del_user)
            for x in del_user_index:
                test_stat_df.iloc[i, x] = np.nan
        print('随机删除的内容有：',del_column[:10])
        return test_stat_df,del_column
    
    def test_data_pre(self,test_stat_df):
        """
        删除测试集中所有的label集中的字段，将测试集中的空格也变为nan类型
        对于测试集中的每一个用户，随机删除一个属性，在这里要求所有当前特征中删除一个
        
        Returns
        -----------
        test_stat_df:dataframe,
        del_column:list,存放被删除的属性名称（每个用户删除n个）
        
        
        """
        if 'Unnamed: 0' in test_stat_df.columns:
            test_stat_df=test_stat_df.drop(['Unnamed: 0'],axis=1)
        for i in self.label: 
            if i in test_stat_df.columns:
                test_stat_df = test_stat_df.drop([i], axis=1)  
                
        for i in self.text:  
            if i in test_stat_df.columns:
                test_stat_df = test_stat_df.drop([i], axis=1)
                
        test_stat_df = test_stat_df.applymap(self.None2NaN)  
        
        del_column = []
        print('test_stat_df.shape:',test_stat_df.shape)
        
        for i in range(test_stat_df.shape[0]):
            x = random.randint(0, test_stat_df.shape[1]-1)
            try:
                while pd.isna(test_stat_df.iloc[i,x]):
                    x = x+1
            except:
                x = random.randint(0, test_stat_df.shape[1]-1)
                while pd.isna(test_stat_df.iloc[i,x]):
                    x=random.randint(0, test_stat_df.shape[1]-1)
            
            
            test_stat_df.iloc[i, x] = np.nan
            del_column.append(test_stat_df.columns[x])

        print('随机删除的内容有：',del_column[:10])
        return test_stat_df,del_column
    
    def None2NaN(self,x):
        if x == '':
            x = np.nan
        return x
    
    def Attribute_select(self,train_df):
        """
            和Mti_Rec保持对属性选择的一致
            
            Parameters
            ----------
            train_df:dataframe,columns=[label,text,enum],index=[natural ids, have been reseted]
            
            Returns
            ----------
            train_df:dataframer,columns=[enum],index=[...](just like before)

        """
        print('train_df:',train_df.shape)
        for i in self.label:
            if i in train_df.columns:
                train_df=train_df.drop(i,axis=1)
        for i in self.text:
            if i in train_df.columns:
                train_df=train_df.drop(i,axis=1)
        print('now train_df:',train_df.shape)
        return train_df
            
        
    def popularity_W(self,train_df):
        """
        Parameters
        ----------
        train_df:dataframe,columns=[attributes]
        
        
        Returns
        ----------
        df_notna:series,index=train_df.columns,
                tips:the value is the probability.(每个属性出现的非空的次数除以文章的总数)
        
        """
        train_df = train_df.applymap(self.None2NaN)#首先先把空格键变成nan 然后再进行统计
        df_notna=train_df.count()
        df_notna=df_notna.div(train_df.shape[0])#变成概率
        return df_notna
    

    def popularity_predict(self,trained_series,test_stat_df,N):
        """
        比较测试集中的所有用户的 空缺属性 的概率大小
        进行排序后,排名前N 的放进一个新的df中
        
        Parameters
        ----------
        trained_series:series,索引是属性,也就是上一个函数的train_df.columns。值是概率。
        test_stat_df:dataframe,columns是属性 index是自然顺序(1,2,3...)
        N:得到对于每一个用户来说,最重要的的前N个缺失属性。
        
        
        Returns
        ----------
        predicy_top_N:dataframe,columns是1-N,也就是对于每个用户来说最重要的前N个属性和概率值,index是用户编号。
                    填充内容的格式是 tuple
                    
        """
        predicy_top_N=pd.DataFrame(columns=range(1,N+1),index=test_stat_df.index)#
        
        for i in range(test_stat_df.shape[0]):#对于每一个用户
            dict_temp={}
            list_temp=[]
            for j in range(test_stat_df.shape[1]):#对于每一个属性
                feature=test_stat_df.columns[j]
                if pd.isna(test_stat_df.iloc[i,j]):#
                    value=trained_series[feature]
                    dict_temp[feature]=value
            list_temp=sorted(dict_temp.items(),key=lambda x :x[1],reverse=True)
            
            for k in range(N):
                #print(list_temp[k])
                try:
                    predicy_top_N.iloc[i,k]=list_temp[k]
                except:
                    predicy_top_N.iloc[i,k]=np.nan
            
        return predicy_top_N
            
    def get_topN_features(self,predicy_top_N,value_N):
        """
        Parameters
        ----------
        predicy_top_N:dataframe,columns=range(1,N+1),index=natural user_id
        value_N:选定的predicy_top_N的前N个元组的第一个元素(特征名)
        
        
        Returns
        ----------
        feature_topN:list,length=user_num,
                        [[feature1,feature2,feature3...],[userid=2],]
        """
        feature_topN=[]
        for i in range(predicy_top_N.shape[0]):#对于每一个用户来说
            user_featureTopN=[]
            for j in predicy_top_N.iloc[i,:]:#j是元组
                if pd.isna(j):
                    break
                else:
                    string1=j[0]
                    user_featureTopN.append(string1)
                    
            feature_topN_k=[]
            for k in range(value_N):
                try:
                    feature_topN_k.append(user_featureTopN[k])
                except:
                    pass
            feature_topN.append(feature_topN_k)
        return feature_topN
    
    def get_group_2(self,start_id,end_id,train_df):
        """得到现存的二元组组合

        Args
        ----------
            start_id (int): label的第start_id类标签
            end_id (int): label的第end_id类标签
            train_df (dataframe): 训练集

        Returns
        ----------
            group (list(tuple(int,int))): 治疗方案组合
        """
        group=[]
        label1=self.label[start_id]
        label2=self.label[end_id]
        print('当前的组合为 :',label1,label2)
        column_list=list(train_df.columns)
        index1=column_list.index(label1)
        index2=column_list.index(label2)
        for j in range(train_df.shape[0]):#对于每一个用户
            group.append(tuple((train_df.iloc[j][index1],train_df.iloc[j][index2])))
        group=list(set(group))
        print('产生的组合长度为:',len(group))
        return group
    
    def get_M_2(self,train_df,M):
        if M==2:
            group1=self.get_group_2(0,1,train_df)
            group2=self.get_group_2(0,2,train_df)
            group3=self.get_group_2(0,3,train_df)
            group4=self.get_group_2(1,2,train_df)
            group5=self.get_group_2(1,3,train_df)
            group6=self.get_group_2(2,3,train_df)
        return group1,group2,group3,group4,group5,group6
    
    def count_p(self,start_id,end_id,group,train_df):
        """
        group:治疗方案组合
        """
        new_df=pd.DataFrame(columns=train_df.columns)
        label1=self.label[start_id]
        label2=self.label[end_id]
        print('当前的组合为 :',label1,label2)
        column_list=list(train_df.columns)
        index1=column_list.index(label1)
        index2=column_list.index(label2)
        
        p=pd.Series(data=[0 for _ in train_df.columns],index=train_df.columns)
        for i in group:#对于每一个治疗方案组合
            new_df=pd.DataFrame(columns=train_df.columns)
            for j in range(train_df.shape[0]):#对每一个用户
                if train_df.iloc[j,index1]==i[0] and train_df.iloc[j,index2]==i[1]:
                    new_df.loc[j,:]=train_df.iloc[j,:]
            p1=self.popularity_W(new_df)
            p+=p1
        
        p=p/len(group)
        print(p)
        return new_df,p



class TM_Inference_Rec():
    def __init__(self,label,enum,text):
        self.label=label
        self.enum=enum
        self.text=text

    def get_new_df(self,test_df,therapygroup2term,llda_model,labeled_documents2,M,therapy_freq):
        """
        Difference
        ----------
        和之前的推荐比起来，其实就是用dhtm的inference推断代替了之前的匹配度(一致性)
        Parameters
        ----------
        test_df,group2feature,llda_model,labeled_documents2:略
        M:M元组
        
        
        Returns
        ----------
        new_df:dataframe,index=test_df.index(用户),columns=group2feature.index(治疗方案组)
        """
        AA=np.zeros((test_df.shape[0],therapygroup2term.shape[0]))
        new_df=pd.DataFrame(AA,index=test_df.index,columns=therapygroup2term.index)
        if M==1:
            for i in range(test_df.shape[0]):
                document_example=labeled_documents2[i]
                return_inference=llda_model.inference(document_example)
                for j in therapygroup2term.index:
                    new_df.loc[i,j]=return_inference[llda_model.topic_vocabulary[int(j)]]
            return new_df
        elif M==2:
            for i in range(test_df.shape[0]):
                document_example=labeled_documents2[i]
                return_inference=llda_model.inference(document_example)
                for j in therapygroup2term.index:
                    list_1=j.split(',')
                    topic_1=int(list_1[0][1:])
                    topic_2=int(list_1[1][0:-1])
                    i_fre=therapy_freq[topic_1]
                    j_fre=therapy_freq[topic_2]
                    i_inf=return_inference[llda_model.topic_vocabulary[topic_1]]
                    j_inf = return_inference[llda_model.topic_vocabulary[topic_2]]

                    ijkl_fin=i_fre * i_inf + j_fre * j_inf

                    new_df.loc[i,j]+=ijkl_fin
            return new_df
        elif M==3:
            for i in range(test_df.shape[0]):
                document_example=labeled_documents2[i]
                return_inference=llda_model.inference(document_example)
                #temp=0
                for j in therapygroup2term.index:
                    list_1=j.split(',')
                    topic_1=int(list_1[0][1:])
                    topic_2=int(list_1[1])
                    topic_3=int(list_1[2][0:-1])
                    #temp+=return_inference[topic_1]*return_inference[topic_2]*return_inference[topic_3]
                    new_df.loc[i,j]=return_inference[llda_model.topic_vocabulary[topic_1]]*return_inference[llda_model.topic_vocabulary[topic_2]]*return_inference[llda_model.topic_vocabulary[topic_3]]
            return new_df
        elif M==4:
            for i in range(test_df.shape[0]):
                document_example=labeled_documents2[i]
                return_inference=llda_model.inference(document_example)
                for j in therapygroup2term.index:
                    list_1=j.split(',')
                    topic_1=int(list_1[0][1:])
                    topic_2=int(list_1[1])
                    topic_3=int(list_1[2])
                    topic_4=int(list_1[3][0:-1])

                    i_fre=therapy_freq[topic_1]
                    j_fre=therapy_freq[topic_2]
                    k_fre=therapy_freq[topic_3]
                    l_fre=therapy_freq[topic_4]

                    i_inf=return_inference[llda_model.topic_vocabulary[topic_1]]
                    j_inf = return_inference[llda_model.topic_vocabulary[topic_2]]
                    k_inf = return_inference[llda_model.topic_vocabulary[topic_3]]
                    l_inf = return_inference[llda_model.topic_vocabulary[topic_4]]

                    ijkl_fin=i_fre * i_inf + j_fre * j_inf + k_fre * k_inf + l_fre * l_inf

                    new_df.loc[i,j]+=ijkl_fin
            return new_df              
                
