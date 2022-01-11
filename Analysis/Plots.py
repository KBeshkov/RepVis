#Plots
import matplotlib.pyplot as plt


def gen_subplots(data,x,y,res=200):
    plt.figure(dpi=res)
    count = 1
    for i in range(x):
        for j in range(y):
            plt.subplot(x,y,count)
            plt.plot(data[i][j])
            count += 1
            