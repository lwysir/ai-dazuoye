## 1.LogisticRegression模型

### （1）包和库

- 导入所需要的包

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
```

### （2）数据准备并分析

- 导入原始数据集

```
df = pd.read_csv(r'D:\develop\train.csv')
```

- 绘制饼状图显示5G用户分类的情况

```
df.describe()
df['target'].value_counts()
df['target'].value_counts().plot(kind='pie',autopct='%.3f%%')
```

![image-20240526173838382](README.assets/image-20240526173838382.png)

### （3）拆分数据集并训练模型

- 提取特征矩阵和标签

```
x = df.iloc[:, 1:-1]
y = df.iloc[:, -1]
```

-  切分训练集和测试集, 并且进行随机数打乱

```
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.3, random_state=12)
```

-  重置训练集和测试集的索引

```
for i in [Xtrain, Xtest, Ytrain, Ytest]:
    i.reset_index(drop=True, inplace=True)
```

-  对所有特征进行归一化处理

```
mms = MinMaxScaler()
Xtrain = pd.DataFrame(mms.fit_transform(Xtrain), columns=Xtrain.columns)
Xtest = pd.DataFrame(mms.transform(Xtest), columns=Xtest.columns)
```

- 对训练集进行过采样

```
model_smote = SMOTE()
Xtrain, Ytrain = model_smote.fit_resample(Xtrain, Ytrain)
```

- 训练逻辑回归模型，尝试使用 'saga' 优化算法并增加迭代次数

```
clf = LR(max_iter=20000, C=9.4, solver='saga')
clf = clf.fit(Xtrain, Ytrain)

print('训练集上的预测准确率为：', clf.score(Xtrain, Ytrain))
print('测试集上的预测准确率为：', clf.score(Xtest, Ytest))
```

### （4）性能评估

- 输出混淆矩阵

```
print('混淆矩阵：\n', confusion_matrix(Ytest, clf.predict(Xtest)))
```

- 查看测试集上AUC面积

```
area = roc_auc_score(Ytest, clf.predict_proba(Xtest)[:, 1])
print('AUC面积为：', area)

print(classification_report(Ytest, clf.predict(Xtest)))
```

- 绘制ROC曲线

```
FPR, recall, thresholds = roc_curve(Ytest, clf.predict_proba(Xtest)[:, 1])
plt.figure()
plt.plot(FPR, recall, color='red', label='ROC curve (area = %0.2f)' % area)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.show()
```

![image-20240526174504688](README.assets/image-20240526174504688.png)

## 2.xgboost模型

#### (1)包和库
* 最核心的算法库
```
from xgboost import XGBClassifier
```
* 导入其他必要的库
```
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, roc_curve
```
#### (2)数据准备并分析数据
* 从文件中读取数据，查看数据的基本信息，打印数据集中每个特征的描述性统计
```
train_df = pd.read_csv(r'D:\develop\train.csv')
display(train_df.head())
train_df.info()
print(train_df.describe())
```
* 绘制条形图显示5G用户分类的情况
```
print(train_df['target'].value_counts())
train_df['target'].value_counts().plot(kind='bar')
plt.show()
```
![dataset](./images/5G-pre.png)
* 数据预处理，提取特征矩阵x和标签y
```
x = train_df.iloc[:, 1:-1]  
y = train_df.iloc[:, -1]    
```
#### (3)拆分数据集并训练模型
* 使用train_test_split函数切分数据集为训练集和测试集
```
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.2, random_state=seed)
```
* 定义xgboost模型参数
```
xgb_params = {
    'booster': 'gbtree', 
    'objective': 'binary:logistic',
    'n_estimators': 200,  
    'max_depth': 8, 
    'lambda': 10,  
    'subsample': 0.7, 
    'colsample_bytree': 0.8,  
    'colsample_bylevel': 0.7,  
    'eta': 0.1,  
    'tree_method': 'hist', 
    'seed': seed, 
    'nthread': 16 
}
```
* 训练xgboost模型
```
xgb_model = XGBClassifier(**xgb_params, eval_metric='auc', early_stopping_rounds=20)
xgb_model.fit(Xtrain, Ytrain, eval_set=[(Xtrain,Ytrain),(Xval, Yval)], verbose=True)
```
#### (4)性能评估
* 计算并打印测试集上的AUC分数
```
test_auc = roc_auc_score(Ytest, Ypred)
print(f"Test AUC: {test_auc}")
```
* 绘制ROC曲线
```
fpr, tpr, thresholds = roc_curve(Ytest, Ypred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {test_auc:0.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Test Set')
plt.legend(loc="lower right")
plt.show()
```
![dataset](./images/xgboost-roc.png)

