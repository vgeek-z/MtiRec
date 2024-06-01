# coding:utf-8
import pandas as pd
import math
import numpy as np
import random
import heapq
import sys
import itertools
sys.path.append('../')
from model.data_process import data_processing
from model.mti_rec import Mti_Recommend
from model.mti_rec import TM_Inference_Rec
import model.dhtm as dhtm
import model.assessment_criteria as assessment_model
import time 
from multiprocessing.dummy import Pool as ThreadPool
import logging
import os
import pickle

adjuvant=pd.read_csv('data/data.csv')


label = [
"chemotherapy_id",
"radiotherapy_id",
"endocrine_therapy_id",
"targeted_therapy_id",   
]

enum = ["side",
"age",
"is_bilateral",
"weight",
"height",
"surface_area",
"kps",
"phase",
"aln",
"scln",
"pre_nac_aln_fna",
"pre_nac_scln_fna",
"nac_scheme",
"nac_course",
"rs_21_gene",
"skin_violated",
"bmi",
"is_menopause", #
"is_neo_adjuvant",#
"is_ct", #
"ct_scheme",#
"is_rt",#
"is_tt",#
"et_1_scheme",#
"et_1_duration",#
"et_2_scheme",#
"et_2_duration",#
"et_course",#
"adjuvant_remarks",
"totle_cholesterol",
"fasting_blood_triglyceride",
"fasting_plasma_ldl",
"fasting_plasma_hdl",
"fbg",
"cea",
"ca153",
"rt_scheme",
"tt_scheme",
"is_et",
"cnb_er_value_1",
"cnb_er_value_2",
"cnb_cerbb_2_1",
"cnb_cerbb_2_2",
"cnb_her2_fish_1",
"cnb_her2_fish_2",
"cnb_ki67_value_1",
"cnb_ki67_value_2",
"post_nac_pe_size",
"post_nac_us_size",
"post_nac_mri_size",
"post_nac_aln_size",
"post_nac_scln_size",
"pre_nac_aln_size",
"pre_nac_scln_size",
"lump_quadrant_1",
"lump_quadrant_2",
"slnb_biopsy_count",
"slnb_metastasize_count",
"alnd_clean_count",
"alnd_metastasize_count",
"psln_biopsy_count",
"psln_metastasize_count",
"histological_grade_1",
"histological_grade_2",
"ki67_value_2",
"cerbb_2_1",
"cerbb_2_2",
"her2_fish_1",
"her2_fish_2",
"surgical_margin",
"embolus",
"pre_nac_pe_size_2",
"pre_nac_mri_size_2",
"pre_nac_mmg_size_2",
"pe",
"nac_aln_1",
"nac_aln_2",
"nac_scln_1",
"nac_scln_2",
"post_clinic_phase",
"aln_side",
"scln_side",
"clinical_stages",
"rs_21_gene_side"]

text = []

from sklearn import model_selection
def adjuvant_split(adjuvant):
    train_df,test_df=model_selection.train_test_split(adjuvant,test_size=0.2)
    train_df=train_df.reset_index(drop=True)
    test_df=test_df.reset_index(drop=True)
    return train_df,test_df


id_signal=[]#自定义
def TrainSetProcess(train_df):
    labeled_documents = []
    all_labels=[]
    #print(label)
    print('The len of list ENUM:',len(enum))
    print('The len of list TEXT',len(text))
    for index, row in train_df.iterrows():
        #print(row)
        label_tokens = []
        label_exist=[]     #对于是否存在该类别的mask 
        enum_tokens = []
        text_content = ""
        for l in label:
            label_tokens.append(int(row[l]))
            if int(row[l]) in id_signal: #如果该治疗方案为‘不需要’
                label_exist.append(0)
            else:
                label_exist.append(1)
        for attr in enum:
            if str(row[attr]).strip() != '' and row[attr] != None:
                if type(row[attr]) != str and math.isnan(row[attr]): 
                    continue
                enum_tokens.append(attr + ':' +str(row[attr]))
        for attr in text:
            if str(row[attr]).strip() != '':   
                text_content = text_content + " " + str(row[attr])
        labeled_documents.append(("##".join(enum_tokens), label_tokens,label_exist))
        all_labels.extend(label_tokens)
    return labeled_documents

def TestSetProcess(test_df):
    labeled_documents = []
    print('The len of list ENUM:',len(enum))
    print('The len of list TEXT',len(text))
    for index, row in test_df.iterrows():
        enum_tokens = []
        text_content = ""
        for attr in enum:
            if str(row[attr]).strip() != '' and row[attr] != None:
                if type(row[attr]) != str and math.isnan(row[attr]): 
                    continue
                try:
                    enum_tokens.append(attr + ':' +str(row[attr]))
                except:
                    print('!!!!!!!!!!!!!')
                    return 0
        labeled_documents.append("##".join(enum_tokens))
    return labeled_documents

"""
    调用、初始化模型。
"""
def model_instantiation(labeled_documents):
    dhtm_model = dhtm.DhtmModel(labeled_documents=labeled_documents)
    for i in range(150):
        print("iteration %s sampling..." % (dhtm_model.iteration + 1))
        dhtm_model.training(1) 
        print("after iteration: %s, perplexity: %s" % (dhtm_model.iteration, dhtm_model.perplexity()))
        print("delta beta: %s" % dhtm_model.delta_beta)
        print('*'*40)
        if dhtm_model.is_convergent(method="beta", delta=8.5): #10,0.01
            break
    # save to disk
    save_model_dir = "result_data"
    model_file_path = os.path.join(save_model_dir, 'dhtm_model.pkl')
    with open(model_file_path, 'wb') as file:
        pickle.dump(dhtm_model, file)
    return dhtm_model

"""
    :param typename:一个list,label列表中具体的治疗方案类别
    :param name:（type:list）,治疗方案类别的具体操作
    :param therapygroup2term:
    :param therapyGroups
"""  
def GenerateTherapyGroups(M,K,dhtm_model):
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108]
    names = ["不需要", "其它", "弃权", "P", "CMF", "EC(AC)", "TC*4", "TCb", "PCb", "TEC", "EC-T", "EC-wP", "A-P-C", "ddAC-ddP", "ddAC-wP", "FEC-T", "EC-wPCb", "卡培他滨", "不需要", "其它", "弃权", "需要", "APBI=部分乳腺照射", "全乳", "全乳+瘤床补量", "全乳+瘤床补量+锁骨上/下区", "全乳+瘤床补量+锁骨上/下区+内乳区", "胸部+锁骨上/下区", "胸部+锁骨上/下区+内乳区", "不需要", "其它", "弃权", "SERM*5y", "SERM*10y", "SERM*2-3y序贯AI*3-2y", "SERM*5y序贯AI*5y", "AI*5y", "AI*2-3y-SERM", "SERM+OFS", "AI+OFS", "不需要", "其它", "弃权", "待FISH结果", "曲妥珠单抗（与化疗同时）", "与化疗序贯", "继续曲妥珠单抗治疗（新辅助后）", "AI*10y", "待定", "待定", "待定", "待定", "SERM预防", "CEF", "TC*6", "E--卡培他滨", "出错", "继续2-3年AI", "继续5年SERM", "继续5年AI", "其他", "待定", "SERM*2-3y序贯AI*5y", "（满5年）不继续内分泌治疗", "待定", "EC->wP/T", "EC->T", "ddAC->w/ddP", "EC→T+Cb", "TEC", "PCb", "THP", "PCbH", "EC→TH", "AI", "AI + OFS", "不建议新辅助", "其他", "弃权", "卡培他滨", "卡培他滨+曲妥珠单抗", "T", "不化疗，待21基因RS", "待FISH结果", "待21基因RS和FISH结果", "化疗，待21基因RS", "继续OFS+SERM", "继续OFS+AI", "曲妥珠单抗+帕妥珠单抗（与化疗同时）", "T-DM1治疗（新辅助后）", "继续曲妥珠单抗+帕妥珠单抗（新辅助后）", "T/PCbH+帕妥珠单抗", "EC→TH+帕妥珠单抗", "待FISH检测", "白蛋白紫杉醇+HP+/-吡咯替尼", "白蛋白紫杉醇+卡铂+/-HP", "内分泌治疗", "化疗", "化疗+曲妥珠单抗", "化疗+HP靶向治疗", "手术", "其他", "弃权", "内分泌治疗+靶向治疗", "待FISH检测"]
    typename = ["Chemotherapy", "Chemotherapy", "Chemotherapy", "Chemotherapy", "Chemotherapy", "Chemotherapy", "Chemotherapy", "Chemotherapy", "Chemotherapy", "Chemotherapy", "Chemotherapy", "Chemotherapy", "Chemotherapy", "Chemotherapy", "Chemotherapy", "Chemotherapy", "Chemotherapy", "Chemotherapy", "Radiotherapy", "Radiotherapy", "Radiotherapy", "Radiotherapy", "Radiotherapy", "Radiotherapy", "Radiotherapy", "Radiotherapy", "Radiotherapy", "Radiotherapy", "Radiotherapy", "EndocrineTherapy", "EndocrineTherapy", "EndocrineTherapy", "EndocrineTherapy", "EndocrineTherapy", "EndocrineTherapy", "EndocrineTherapy", "EndocrineTherapy", "EndocrineTherapy", "EndocrineTherapy", "EndocrineTherapy", "TargetedTherapy", "TargetedTherapy", "TargetedTherapy", "TargetedTherapy", "TargetedTherapy", "TargetedTherapy", "TargetedTherapy", "EndocrineTherapy", "Chemotherapy", "Radiotherapy", "EndocrineTherapy", "TargetedTherapy", "EndocrineTherapy", "Chemotherapy", "Chemotherapy", "Chemotherapy", "ReinforceEndocrineTherapy", "ReinforceEndocrineTherapy", "ReinforceEndocrineTherapy", "ReinforceEndocrineTherapy", "ReinforceEndocrineTherapy", "ReinforceEndocrineTherapy", "EndocrineTherapy", "ReinforceEndocrineTherapy", "NeoTherapy", "NeoTherapy", "NeoTherapy", "NeoTherapy", "NeoTherapy", "NeoTherapy", "NeoTherapy", "NeoTherapy", "NeoTherapy", "NeoTherapy", "NeoTherapy", "NeoTherapy", "NeoTherapy", "NeoTherapy", "NeoTherapy", "NeoTherapy", "NeoTherapy", "Chemotherapy", "Chemotherapy", "Chemotherapy", "Chemotherapy", "Chemotherapy", "ReinforceEndocrineTherapy", "ReinforceEndocrineTherapy", "TargetedTherapy", "TargetedTherapy", "TargetedTherapy", "NeoTherapy", "NeoTherapy", "NeoTherapy", "NeoTherapy", "NeoTherapy", "DeNovoFourTherapy", "DeNovoFourTherapy", "DeNovoFourTherapy", "DeNovoFourTherapy", "DeNovoFourTherapy", "DeNovoFourTherapy", "DeNovoFourTherapy", "DeNovoFourTherapy", "DeNovoFourTherapy"]

    Chemotherapy_id_list=[]
    Radiotherapy_id_list=[]
    Endocrine_therapy_id_list=[]
    Targeted_therapy_id_list=[]
    
    therapy_freq=dict()
    for i in ids:
        count=0
        for j in range(len(labeled_documents)):
            if i in labeled_documents[j][1][1:]:
                #print('i:',i)
                count+=1
        count=count/len(labeled_documents)
        therapy_freq[i]=count

    for i in range(len(typename)):
        if ids[i] in dhtm_model.topic_vocabulary:
            if typename[i] == "Chemotherapy":
                Chemotherapy_id_list.append(ids[i])
            elif typename[i] == "Radiotherapy":
                Radiotherapy_id_list.append(ids[i])
            elif typename[i] == "EndocrineTherapy":
                Endocrine_therapy_id_list.append(ids[i])
            elif typename[i] == "TargetedTherapy":
                Targeted_therapy_id_list.append(ids[i])
    terms=dhtm_model.terms
    therapygroup2term=pd.DataFrame(columns=terms)
    if M ==1:
        if K==0:
            print('当前只考虑ct一种情况！')
            for i in Chemotherapy_id_list:
                temp=dhtm_model.beta_k(dhtm_model.topic_vocabulary[i])
                temp=therapy_freq[i]*temp
                therapygroup2term.loc[str(i)]=temp
        elif K==1:
            print('当前只考虑rt一种情况')
            for i in Radiotherapy_id_list:
                temp=dhtm_model.beta_k(dhtm_model.topic_vocabulary[i])
                temp=therapy_freq[i]*temp
                therapygroup2term.loc[str(i)]=temp
        elif K==2:
            print('当前只考虑et一种情况')
            for i in Endocrine_therapy_id_list:
                temp=dhtm_model.beta_k(dhtm_model.topic_vocabulary[i])
                temp=therapy_freq[i]*temp
                therapygroup2term.loc[str(i)]=temp
        elif K==3:
            print('当前只考虑tt一种情况')
            for i in Targeted_therapy_id_list:
                temp=dhtm_model.beta_k(dhtm_model.topic_vocabulary[i])
                temp=therapy_freq[i]*temp
                therapygroup2term.loc[str(i)]=temp
    if M==2:
        print('当前组合为ct-rt')
        therapygroup2term=Mis2(Chemotherapy_id_list,Radiotherapy_id_list,Endocrine_therapy_id_list,Targeted_therapy_id_list,dhtm_model,therapygroup2term,therapy_freq)
        print('当前的therapygroup2term形状为：',therapygroup2term.shape)
    elif M==4:
        print('ct-rt-et-tt')
        therapygroup2term=Mis4(Chemotherapy_id_list,Radiotherapy_id_list,Endocrine_therapy_id_list,Targeted_therapy_id_list,dhtm_model,therapygroup2term,therapy_freq)
    
    therapyGroups=list(therapygroup2term.index)
    print('治疗方案组合的例子',therapyGroups[:10])
    return therapyGroups,therapygroup2term,therapy_freq
    #therapygroup2term.to_excel('D:/python_projectsss/LLDA/therapygroup2term.xlsx')
def Mis2(Chemotherapy_id_list,Radiotherapy_id_list,Endocrine_therapy_id_list,Targeted_therapy_id_list,dhtm_model,therapygroup2term,therapy_freq):
    list_all=[Chemotherapy_id_list,Radiotherapy_id_list,Endocrine_therapy_id_list,Targeted_therapy_id_list]
    for n1 in range(3):
        for n2 in range(n1+1,4):
            list1=list_all[n1]
            list2=list_all[n2]
            for (i,j) in itertools.product(list1, list2):
                if i not in dhtm_model.topic_vocabulary:
                    print('缺少的Chemotherapy_id_list为%s'%i)
                    continue
                if j not in dhtm_model.topic_vocabulary:
                    print('当前的i为%s,缺少的Radiotherapy_id_list为%s'%(i,j))
                    continue
                i_temp=dhtm_model.beta_k(dhtm_model.topic_vocabulary[i])
                j_temp=dhtm_model.beta_k(dhtm_model.topic_vocabulary[j])
                i_fre=therapy_freq[i]
                j_fre=therapy_freq[j]
                temp=map(lambda a, b: a + b, i_fre*i_temp, j_fre*j_temp)
                temp=list(temp)
                therapygroup2term.loc[str((i,j))]=temp
    return therapygroup2term
def Mis4(Chemotherapy_id_list,Radiotherapy_id_list,Endocrine_therapy_id_list,Targeted_therapy_id_list,dhtm_model,therapygroup2term,therapy_freq):
    for (i,j,k,l) in itertools.product(Chemotherapy_id_list, Radiotherapy_id_list,Endocrine_therapy_id_list,Targeted_therapy_id_list):
        if i not in dhtm_model.topic_vocabulary:
            print('缺少的Chemotherapy_id_list为%s'%i)
            continue
        if j not in dhtm_model.topic_vocabulary:
            print('当前的i为%s,缺少的Radiotherapy_id_list为%s'%(i,j))
            continue
        if k not in dhtm_model.topic_vocabulary:
            print('当前的i为%s,缺少的Endocrine_therapy_id_list为%s'%(i,j))
            continue
        if l not in dhtm_model.topic_vocabulary:
            print('当前的i为%s,缺少的Targeted_therapy_id_list为%s'%(i,l))
            continue
        i_temp=dhtm_model.beta_k(dhtm_model.topic_vocabulary[i])
        j_temp=dhtm_model.beta_k(dhtm_model.topic_vocabulary[j])
        k_temp=dhtm_model.beta_k(dhtm_model.topic_vocabulary[k])
        l_temp=dhtm_model.beta_k(dhtm_model.topic_vocabulary[l])
        i_fre=therapy_freq[i]
        j_fre=therapy_freq[j]
        k_fre=therapy_freq[k]
        l_fre=therapy_freq[l]
        temp=map(lambda a, b,c,d: a + b+c+d, i_fre*i_temp, j_fre*j_temp,k_fre*k_temp,l_fre*l_temp)
        temp=list(temp)
        therapygroup2term.loc[str((i,j,k,l))]=temp
    return therapygroup2term

"""
    随机删除测试集中的一项数据，并把删除的测试的属性名保留在del_column。
"""
def None2NaN(x):
    if x == '':
        x = np.nan
    return x


##训练一次 然后多次计算测试
train_df,test_df=adjuvant_split(adjuvant)

train_df.to_excel('result_data/train_df.xlsx')
test_df.to_excel('result_data/test_df.xlsx')
print('...划分训练集、测试集结束')

labeled_documents=TrainSetProcess(train_df)
print('...训练集处理结束')
print('Step1: Data process (finished)')

#当需要使用已经训练好的模型时，注释掉模型训练的代码
#并取消以下代码的注释
#with open('result_data/dhtm_model.pkl', 'rb') as file: 
    #dhtm_model = pickle.load(file)
dhtm_model=model_instantiation(labeled_documents)  #时间复杂度：M*MN*K
print('...dhtm_model.iteration:',dhtm_model.iteration)
print('Step2: Train DHTM (finished)')
terms=dhtm_model.terms
df_terms = pd.DataFrame(terms)
df_terms.to_excel('result_data/terms.xlsx')

TM_infer=TM_Inference_Rec(label,enum,text)
recommend=Mti_Recommend(labeled_documents,terms,label,enum,text)

feature_wordsbag=recommend.get_Feature2value(terms,test_df)#得到一个特征的词袋，具体的值是特征：特征值的词 
feature_wordsbag.to_excel('result_data\\feature_wordsbag.xlsx')

test_temp=test_df

M = 2 
print('当前是%d元组'%M)
therapyGroups,therapygroup2term,therapy_freq=GenerateTherapyGroups(M,0,dhtm_model)
print('Step3: Form therapy tuples. (finished)')

test_df=test_temp
test_df=test_df.reset_index(drop=True)
test_df,del_column=recommend.test_data_pre(test_df)
labeled_documents2=TestSetProcess(test_df)
new_df2=TM_infer.get_new_df(test_df,therapygroup2term,dhtm_model,labeled_documents2,M,therapy_freq)
print('Step4: Calculate the connections between patients and therapy tuples by DHTM (new_df2). (finished)')

# 多线程比较对于当前数据集而言可能最佳的参数
lambda_df_list=[0.5]  #参数可能的取值列表，该参数取值范围在0~1
weight_list=[0.2]     #该参数取值范围在 -1~+1，这是一个惩罚系数
num_PCs_list=[5]      #该参数取值要求为正整数，过大的主成分集合长度会可能会导致计算资源的浪费

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s[%(levelname)s %(message)s'
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

def process(lambda_df):
    print('origin:',lambda_df,'inference:',1-lambda_df)
    new_df=lambda_df*new_df1+(1-lambda_df)*new_df2
    print('Step6: Combine these two kinds of connections by a weighted fusion. (finished)')
      
    gain_value=recommend.get_gain_value(train_df,test_df,new_df,therapygroup2term,feature_wordsbag)
    print('当前得到的是患者的属性增益值')
    print('Step7: Calculate the accumulated gain of null attribute for patients. (finished)')

    #gain_value.to_excel('result_data/gain_value_lambda{}_weight{}_bench{}.xlsx'.format(int(lambda_df*100),int(weight_t1*100),num_PCs))
    locals()['Accuracy_lambda{}_weight{}_bench{}'.format(int(lambda_df*100),int(weight_t1*100),num_PCs)]=[]
    Assessment=assessment_model.assessment_criteria(gain_value,del_column)
    acc_list=Assessment.Accuracy_List(10)   
    mrr_list=Assessment.MRR_List(10)    
    print('Step8: Recommend MTIs and evaluate the performance of the model on three metrics. (finished)')
    locals()['Accuracy_lambda{}_weight{}_bench{}'.format(int(lambda_df*100),int(weight_t1*100),num_PCs)]=acc_list
    locals()['MRR_lambda{}_weight{}_bench{}'.format(int(lambda_df*100),int(weight_t1*100),num_PCs)]=mrr_list
    Accuracy_df=pd.DataFrame(index=[0],columns=range(1,11))
    Accuracy_df.iloc[0,:]=locals()['Accuracy_lambda{}_weight{}_bench{}'.format(int(lambda_df*100),int(weight_t1*100),num_PCs)]
    Accuracy_df.to_excel('result_data/Accuracy_lambda{}_weight{}_bench{}.xlsx'.format(int(lambda_df*100),int(weight_t1*100),num_PCs))
    mrr_df=pd.DataFrame(index=[0],columns=range(1,11))
    mrr_df.iloc[0,:]=locals()['MRR_lambda{}_weight{}_bench{}'.format(int(lambda_df*100),int(weight_t1*100),num_PCs)]
    mrr_df.to_excel('result_data/MRR_lambda{}_weight{}_bench{}.xlsx'.format(int(lambda_df*100),int(weight_t1*100),num_PCs))

for num_PCs in num_PCs_list:
    therapy_topN_terms=recommend.get_topN_terms(num_PCs,therapygroup2term)  #得到每个方案最重要的N个特征词
    for weight_t1 in weight_list:
        print('当前的权重系数weight_t为：',weight_t1)
        new_df1=recommend.get_new_df(test_df,therapy_topN_terms,weight_t1)
        print('Step5: Calculate the connections between patients and therapy tuples by their same AVPs. (finished)')
        pool=ThreadPool()
        pool.map(process,lambda_df_list)
        pool.close
        pool.join
    #print('(num_PCs)END:%s'%time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) 
