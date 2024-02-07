import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import numpy as np

def setSubplotDimension(a):
    x = 0
    y = 0
    if ( (a % 1) == 0.0):
        x = a
        y =a
    elif((a % 1) < 0.5):
        x = round(a) +1
        y = round(a)
    else :
        x = round(a)
        y = round(a)
    return (x,y)   

def run(it):
    #print(it)
    for nd,im in enumerate(ims):
        im.set_data(list_data[nd,it])    
    return(ims)

print('Create STRF movie')

name_l = ['E1','LGN','IL1','E2','IL2']#'LGN',

for name in name_l:

    list_data = np.load('./work/STRF_data_'+name+'.npy')

    print(np.shape(list_data))

    list_data = list_data[0:49]

    maxV = np.max(np.max(np.max(list_data,axis=1),axis=1),axis=1)
    minV = np.min(np.min(np.min(list_data,axis=1),axis=1),axis=1)
    print(np.shape(maxV))
    list_data = np.asarray(list_data)

    n_cells,n_t,w,h = np.shape(list_data)
    r_x,c_y = setSubplotDimension(np.sqrt(n_cells))

    r_x,c_y = setSubplotDimension(np.sqrt(n_cells))

    fig, axes = plt.subplots(int(r_x),int(c_y),figsize=(6,6)) #plt.subplots( nrows= int(r_x),ncols = int(c_y) )   
    ims = []

    flatten = lambda l: [item for sublist in l for item in sublist]
    axes = flatten(axes)
        
    cnt = 0
    for ax in axes:    
        ax.axis('off')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1,hspace=0.3,wspace=0.3)
        data = list_data[cnt,0,:,:]
        im = ax.imshow(data,cmap=plt.get_cmap('gray'),aspect='auto',interpolation='none',vmax=maxV[cnt],vmin=minV[cnt])
        #ax.set_title('%i'%(cnt),fontsize=5)
        cnt+=1   
        ims.append(im)




    print('Start to create movie for '+name)

    ani = animation.FuncAnimation(fig,run,frames=np.arange(0,n_t),interval=220)#180)
    writer = PillowWriter(fps=20)
    ani.save('./Output/STRF/STRF_'+name+'/animation_Noise.gif', dpi=60, writer=writer)
print('Finish with creating!')

