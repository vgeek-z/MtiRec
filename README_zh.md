# 1、项目环境
输入以下代码创建虚拟环境 MtiRec

    conda env create -f environment.yaml

# 2、项目目录结构

```
│  DHTMRec.py              #训练+预测
│  environment.yaml        #环境依赖
│  README.md               #
├─data                     #存放数据的文件夹
│                          *注1 
│      
├─model
│  │  assessment_criteria.py #评价指标 
│  │  data_process.py        #数据处理
│  │  dhtm.py                #dhtm模型
│  │  mti_rec.py             #工具 一些关系函数 
│  └─__pycache__             #存放.pyc文件       
│          
└─result_data                #保存模型训练和预测的相关结果
        dhtm_model.pkl       #训练好的dhtm的模型
        terms.xlsx           #AVPs (*2)
        test_df.xlsx         #测试集 (*2)
        train_df.xlsx        #训练集 (*2)
        Accuracy_lambda50_weight20_bench5.xlsx  #一个结果示例：参数取值为 0.5 0.2 5的测试集精确度保存结果
```    

*1：数据可以从BCDB官网下载：http://bcdb.mdt.team:8080
*2: 只展示部分数据作为示例

# 3、进行训练和预测
    python DHTMRec.py

# 4、预期运行结果
## 4.1 进度提示信息
训练集\测试集处理完成提示、吉布斯采样进行过程提示、元组信息提示等
## 4.2 模型保存
dhtm模型保存在result_data文件夹中，名为 dhtm_model.pkl
## 4.3 预测结果
将测试集的模型评价指标Accuracy和MRR的结果保存在文件result_data中，文件名应该是对应参数下的评价指标结果，如Accuracy_lambda50_weight20_bench5.xlsx，对应的参数分别为0.5,0.2,5。

# 5、其它信息
1、考虑到隐私保护等原因，医疗数据不进行上传，可向BCDB官网（http://bcdb.mdt.team:8080）申请下载。

2、DHTMRec.py中提供了多线程方法进行模型效果评估，目的是确定自定义数据集的最佳参数；

