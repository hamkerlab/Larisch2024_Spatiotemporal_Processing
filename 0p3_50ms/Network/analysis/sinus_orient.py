import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    maxDegree = 360
    orienSteps = 8
    contLVL = 5
    np.random.seed(10101)
    print(maxDegree/orienSteps)

    orient_Exc = np.load('./work/TuningCurves_sinus_Exc_orientation.npy')
    orient_Exc_grad = (orient_Exc[:,contLVL]*8)
    
    obw = np.load('./work/OrientBandwithEst_Half_Sinus.npy')

    print(np.shape(obw))
    obw_grad = obw[:,contLVL]*8
    y = np.ones(len(orient_Exc_grad))#np.random.rand(len(orient_Exc_grad))*4
    y = np.reshape(y,(108,3))
    y *= np.array((0.25,0.5,0.75))
    y = np.reshape(y,324)

    orient_Exc_rad = orient_Exc_grad*(np.pi/180)
    obw_rad = obw_grad*(np.pi/180)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    c = ax.scatter(orient_Exc_rad,y,c=orient_Exc_rad,cmap='viridis',alpha=0.75,s =obw_grad )
    ax.set_yticklabels([])
    plt.savefig('./Output/scatter_TC')


if __name__ == "__main__":
    main()
