import argparse
import numpy as np; np.random.seed(0)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

parser = argparse.ArgumentParser(description='ログファイルからヒートマップを作成します')
parser.add_argument('input', type=str,
                    help='ログファイル')
args = parser.parse_args()

lr_value_list = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064,0.128, 0.256, 0.512]#縦
batchsize_value_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]#横
dataset =  np.zeros((len(lr_value_list), len(batchsize_value_list)))
with open(args.input,"r") as f:
    lr = 0
    batchsize = 0
    acc = 0
    for line in f:
        line = line.rstrip()
        if "batchsize" in line:
            line = line.split(" ")
            lr = float(line[1])
            batchsize = int(line[3])
        if "PERFORMANCE" in line:
            acc = float(line.split(" ")[4].rstrip("%")) / 100
            lr_index = lr_value_list.index(lr)
            batchsize_index = batchsize_value_list.index(batchsize)
            if dataset[lr_index, batchsize_index] < acc:   
                dataset[lr_index, batchsize_index] = acc
print(dataset)

df = pd.DataFrame({ "Learning Rate":[],
                    "Batch Size":[],
                    "Accuracy":[]
                    })
for i,lr in enumerate(lr_value_list):
    for j,batch_size in enumerate(batchsize_value_list):
        acc = dataset[i][j]
        ser = pd.Series([lr,batch_size,acc], index=df.columns, name=f"{i}_{j}")
        df = df.append(ser)
df = df.pivot("Learning Rate", "Batch Size", "Accuracy")
sns.heatmap(df)
# plt.show()
plt.savefig("plot.png")