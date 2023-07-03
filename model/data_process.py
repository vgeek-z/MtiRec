# This file contains functions for normalizing data, which may overlap with some data processing code in dhtm.py. 
# It facilitates flexible adjustments to the MtiRec framework.
import pandas as pd
import math
from sklearn import model_selection
import random
import numpy as np

class data_processing():
    def scale(self,x, start, end, step):
        res = 0
        if x == None or math.isnan(x):
            return  np.nan
        for s in range(int(start), int(end), step):
            if int(x) in range(s, s + step):
                return int((s + s + step)/2)
        return int(res)

    def height_scale(self,x, start, end, step):
        if x == None or math.isnan(x):
            return None
        if x < 100:
            x *= 100
        for s in range(int(start), int(end), step):
            if int(x) in range(s, s + step):
                return (s + s + step)/200.
        return None
    
    def None2NaN(self,x):
        if x == '':
            x = np.nan
        return x
    """
    :param labeled_documents: A list where each element is a tuple representing the information of a patient.The tuple contains two components: enum_tokens, which represents the set of features, and label_tokens, which represents the set of treatment plans. 
    Example: [('side:0 age:65 is_bilateral:0.0 weight:65 height:1.675 rs_21_gene_side:30', [2020, 55, 22, 37, 41]), ..]
    :param all_labels: A list that stores all the labels. For example, [2017, 1, 2, 3, 4, 2018, 1, 2, ..]
    :param label_tokens: A list that stores the current patient's treatment plans. Example: [ct, et, ..,]
    """
    def get_label_documents(self,train_stat_df,label,enum,text):
        labeled_documents = []
        all_labels=[]
        for index, row in train_stat_df.iterrows():
            #print(row)
            label_tokens = []
            enum_tokens = []
            text_content = ""
            for l in label:
                    if np.isnan(row[l]):
                        label_tokens.append(np.nan)
                    else:
                        label_tokens.append(int(row[l]))
            for attr in enum:
                if str(row[attr]).strip() != '' and row[attr] != None and not pd.isna(row[attr]):
                    if type(row[attr]) == pd._libs.tslibs.timestamps.Timestamp:
                        continue
                    if type(row[attr]) != str and math.isnan(row[attr]): 
                        print('**'+str(row[attr]))
                        continue
                    enum_tokens.append(attr + ':' +str(row[attr]))
            for attr in text:
                if str(row[attr]).strip() != '':   
                    text_content = text_content + " " + str(row[attr])
            labeled_documents.append(("##".join(enum_tokens), label_tokens))
            all_labels.extend(label_tokens)
        return labeled_documents
    
    """
    找到了所有特征的列表 terms
    """
    def get_terms(self,labeled_documents):
        terms=[]
        for i in range(len(labeled_documents)):
            terms.extend(labeled_documents[i][0].split('##'))
        terms=list(set(terms))
        return terms
    
    
    def make_label_dictation(self,labeled_documents):
        labels0=[]
        labels1=[]
        labels2=[]
        labels3=[]
        #labels4=[]
        temp1=[]
        for j in range(len(labeled_documents)):
            temp1.append(labeled_documents[j][1])
        
        for i in range(len(labeled_documents)):
            labels0.append(temp1[i][0])
            labels1.append(temp1[i][1])
            labels2.append(temp1[i][2])
            labels3.append(temp1[i][3])
            #labels4.append(temp1[i][4])
        labels0=list(set(labels0))
        labels1=list(set(labels1))
        labels2=list(set(labels2))
        labels3=list(set(labels3))
        #labels4=list(set(labels4))

        labels_0={ids  : labels for ids,labels in enumerate(labels0)}
        labels_1={ids  : labels  for ids,labels in zip(range(len(labels0),len(labels0)+len(labels1)),labels1)}
        labels_2={ids  : labels for ids,labels in zip(range(len(labels_1)+len(labels_0),len(labels_0)+len(labels_1)+len(labels2)),labels2)}
        labels_3={ids  : labels for ids,labels in zip(range(len(labels_1)+len(labels_0)+len(labels_2),len(labels_0)+len(labels_1)+len(labels2)+len(labels3)),labels3)}
        #labels_4={ids  : labels  for ids,labels in zip(range(len(labels_1)+len(labels_0)+len(labels_2)+len(labels_3),len(labels_0)+len(labels_1)+len(labels_2)+len(labels_3)+len(labels4)),labels4)}
                
        labels={}
        for key,value in labels_0.items():
            labels[key]=value
        for key,value in labels_1.items():
            labels[key]=value
        for key,value in labels_2.items():
            labels[key]=value
        for key,value in labels_3.items():
            labels[key]=value
        '''for key,value in labels_4.items():
            labels[key]=value'''
        return labels_0,labels_1,labels_2,labels_3,labels # ,labels_4 {id:label具体内容}

    def turn_label_to_id(self,labeled_documents):
        labels_0_dict,labels_1_dict,labels_2_dict,labels_3_dict,label_dict=self.make_label_dictation(labeled_documents)
        temp1=[]
        for j in range(len(labeled_documents)):
            temp1.append(labeled_documents[j][1])
        
        for i in range(len(labeled_documents)):#第i个用户
            temp3=[]
            t=list(labels_0_dict.keys())[list(labels_0_dict.values()).index(temp1[i][0])]
            
            temp3.append(list(labels_0_dict.keys())[list(labels_0_dict.values()).index(temp1[i][0])])
            temp3.append(list(labels_1_dict.keys())[list(labels_1_dict.values()).index(temp1[i][1])])
            temp3.append(list(labels_2_dict.keys())[list(labels_2_dict.values()).index(temp1[i][2])])
            temp3.append(list(labels_3_dict.keys())[list(labels_3_dict.values()).index(temp1[i][3])])
            #temp3.append(list(labels_4_dict.keys())[list(labels_4_dict.values()).index(temp1[i][4])])
            labeled_documents[i]=list(labeled_documents[i])
            labeled_documents[i][1]=temp3
        return labeled_documents