import warnings
import configparser
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ------------------------------------------设置参数-----------------------------------------------------

# 可视化SC效果
data1 = pd.read_csv(r'D:\python_common\代码_感知质量大论文\公开数据处理\cp_感知质量数据集\skep_ernie_1.0_large_ch\_Normal_Lstm_Res_max_8_cross_entropy\confusion_matrix.csv')
fig, ax = plt.subplots()
ax.xaxis.tick_top()
# 不显示右侧的刻度表


sns.heatmap(data1, cmap='YlGnBu', annot=True, linewidths=0.5, fmt='g', annot_kws={'size': 20, 'weight': 'bold', 'family': 'Times New Roman'},
            cbar=False)
# plt.title('情感分类混淆矩阵',fontfamily='STsong',size=20,weight='bold',pad=10,loc='center',color='black')
plt.xticks([0.5,1.5,2.5],['消极','中性','积极'],fontfamily='stsong',size=20)
plt.yticks([0.5,1.5,2.5],['消极','中性','积极'],fontfamily='stsong',size=20,rotation=0)
plt.tight_layout()
plt.show()