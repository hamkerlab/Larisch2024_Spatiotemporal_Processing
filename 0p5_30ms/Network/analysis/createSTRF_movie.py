import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
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

name_l = ['E1']#,'LGN','IL1','E2','IL2']#'LGN',

for name in name_l:

    list_data = np.load('./work/STRF_data_'+name+'.npy')

    print(np.shape(list_data))

    list_data = list_data[0:49]


    list_data = np.asarray(list_data)   
    n_cells,n_t,w,h = np.shape(list_data)
 
    for c in range(n_cells):
        list_data[c] = list_data[c] - np.mean(list_data[c])
        list_data[c] = list_data[c]/np.max(np.abs(list_data[c]))


    maxV = np.max(np.max(np.max(list_data,axis=1),axis=1),axis=1)
    minV = np.min(np.min(np.min(list_data,axis=1),axis=1),axis=1)

    print(maxV,minV)

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
        im = ax.imshow(data,cmap=plt.get_cmap('RdBu',9),aspect='auto',interpolation='none',vmax=np.max(maxV),vmin=np.min(minV))#,vmax=maxV[cnt],vmin=minV[cnt])
        #ax.set_title('%i'%(cnt),fontsize=5)
        cnt+=1   
        ims.append(im)




    print('Start to create movie for '+name)

    ani = animation.FuncAnimation(fig,run,frames=np.arange(0,n_t),interval=220)#180)
    writer = PillowWriter(fps=20)
    ani.save('./Output/STRF/STRF_'+name+'/animation_Noise.gif', dpi=50, writer=writer)


    ## make some additional image plots ##
    c = 1
    plt.figure(figsize=(14,20))
    for i in range(20): # 5 cells
        for t in range(170,n_t,20):
            #print(t)
            plt.subplot(20,7,c)
            plt.imshow(list_data[i,t], cmap='RdBu', interpolation='none', vmin = np.min(list_data[i]), vmax=np.max(list_data[i]))
            plt.axis('off')
            c +=1
    plt.savefig('./Output/STRF/STRF_'+name+'/STRF_imshows',bbox_inches='tight',dpi=300)
    plt.close()

    c = 1
    x = np.linspace(0,len(list_data[6,t,:, 24])-1,len(list_data[6,t,:, 24]),dtype='int32')
    
    if name == 'E1':

        c = 1
        plt.figure(figsize=(26,10))
        norm = mp.colors.Normalize(vmax=np.max(np.abs(list_data[6,:])), vmin=-np.max(np.abs(list_data[6,:])))
        for t in range(n_t-10,160,-25):
            plt.subplot(2,6,c)
            plt.title('t = %i ms'%int(300-t), fontsize=18 )
            plt.imshow(list_data[6,t], cmap = plt.get_cmap('RdBu',9),norm=norm, interpolation='none')#, vmin = np.min(list_data[6,:]), vmax=np.max(list_data[6,:]) )
            plt.gca().add_patch(Rectangle((22,0),1,47, linewidth=1, edgecolor='r', facecolor='none',alpha=0.5))
            #plt.axis('off')
            if c == 1:
                plt.axis('on')
                plt.ylabel('y', fontsize=18)
                plt.xlabel('x', fontsize=18)
                plt.xticks([0,w],[0,w],fontsize=12)
                plt.yticks([0,w],[0,w],fontsize=12)
            else:
                plt.xticks([])
                plt.yticks([])
            plt.subplot(2,6,c+6)
            y_data = list_data[6,t,:, 22]   
            plt.plot(x, y_data ,'-o', color='black')
            plt.fill_between(x, y_data ,0, where=y_data >0 , color='steelblue',alpha=0.5, label='ON' )
            plt.fill_between(x, y_data ,0, where=y_data <=0 , color='tomato',alpha=0.5, label='OFF' )
            plt.hlines(0,0,len(x), colors='gray')
            plt.ylim(-1, 1)
            if c == 1:
                plt.ylabel('normalized amplitude', fontsize=18)
                plt.xlabel('spatial extention [px]', fontsize=18)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)     
                plt.legend()
            else:
                plt.xticks([])
                plt.yticks([])
            c +=1
        plt.savefig('./Output/STRF/STRF_'+name+'/STRF_Cell6_filled',bbox_inches='tight',dpi=300)

        c = 1
        plt.figure(figsize=(26,10))
        norm = mp.colors.Normalize(vmax=np.max(np.abs(list_data[11,:])), vmin=-np.max(np.abs(list_data[11,:])))
        for t in range(n_t-10,160,-25):
            plt.subplot(2,6,c)
            plt.title('t = %i ms'%int(300-t) , fontsize=18)
            plt.imshow(list_data[11,t], cmap=plt.get_cmap('RdBu',9),norm=norm, interpolation='none')#, vmin = np.min(list_data[11,:]), vmax=np.max(list_data[11,:]))
            plt.gca().add_patch(Rectangle((17,0),1,47, linewidth=1, edgecolor='r', facecolor='none',alpha=0.5))
            #plt.axis('off')
            if c == 1:
                plt.axis('on')
                plt.ylabel('y', fontsize=18)
                plt.xlabel('x', fontsize=18)
                plt.xticks([0,w],[0,w],fontsize=12)
                plt.yticks([0,w],[0,w],fontsize=12)
            else:
                plt.xticks([])
                plt.yticks([])
            plt.subplot(2,6,c+6)
            y_data = list_data[11,t,:, 17]   
            plt.plot(x, y_data ,'-o', color='black')
            plt.fill_between(x, y_data ,0, where=y_data >0 , color='steelblue',alpha=0.5, label='ON' )
            plt.fill_between(x, y_data ,0, where=y_data <=0 , color='tomato',alpha=0.5, label='OFF' )
            plt.hlines(0,0,len(x), colors='gray')
            plt.ylim(-1, 1)
            if c == 1:
                plt.ylabel('normalized amplitude', fontsize=18)
                plt.xlabel('spatial extention [px]', fontsize=18)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)     
                plt.legend()
            else:
                plt.xticks([])
                plt.yticks([])
            c +=1
        plt.savefig('./Output/STRF/STRF_'+name+'/STRF_Cell11_filled',bbox_inches='tight',dpi=300)

        c = 1
        plt.figure(figsize=(26,10))
        norm = mp.colors.Normalize(vmax=np.max(np.abs(list_data[12,:])), vmin=-np.max(np.abs(list_data[12,:])))
        for t in range(n_t-10,160,-25):
            plt.subplot(2,6,c)
            plt.title('t = %i ms'%int(300-t) , fontsize=18)
            plt.imshow(list_data[12,t], cmap=plt.get_cmap('RdBu',9),norm=norm, interpolation='none')#, vmin = np.min(list_data[12,:]), vmax=np.max(list_data[12,:]))
            plt.gca().add_patch(Rectangle((27,0),1,47, linewidth=1, edgecolor='r', facecolor='none',alpha=0.5))
            #plt.axis('off')
            if c == 1:
                #plt.axis('on')
                plt.ylabel('y', fontsize=18)
                plt.xlabel('x', fontsize=18)
                plt.xticks([0,w],[0,w],fontsize=12)
                plt.yticks([0,w],[0,w],fontsize=12)
            else:
                plt.xticks([])
                plt.yticks([])

            plt.subplot(2,6,c+6)
            y_data = list_data[12,t,:, 27]   
            plt.plot(x, y_data ,'-o', color='black')
            plt.fill_between(x, y_data ,0, where=y_data >0 , color='steelblue',alpha=0.5, label='ON' )
            plt.fill_between(x, y_data ,0, where=y_data <=0 , color='tomato',alpha=0.5, label='OFF' )
            plt.hlines(0,0,len(x), colors='gray')
            plt.ylim(-1, 1)
            if c == 1:
                plt.ylabel('normalized amplitude', fontsize=18)
                plt.xlabel('spatial extention [px]', fontsize=18)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)     
                plt.legend()
            else:
                plt.xticks([])
                plt.yticks([])
            c +=1
        plt.savefig('./Output/STRF/STRF_'+name+'/STRF_Cell12_filled',bbox_inches='tight',dpi=300)

    if name == 'IL1':
        c = 1
        plt.figure(figsize=(25,4))
        for t in range(n_t-10,170,-20):
            plt.subplot(1,7,c)
            y_data = list_data[5,t,:, 24]   
            plt.plot(x, y_data ,'-o', color='black')
            plt.fill_between(x, y_data ,0, where=y_data >0 , color='steelblue',alpha=0.5 )
            plt.fill_between(x, y_data ,0, where=y_data <=0 , color='tomato',alpha=0.5 )
            #plt.fill_between(x[off_part], list_data[6,t,off_part, 24]+((1-list_data[6,t,off_part, 24])),1 , color='tomato',alpha=0.5 )
            #plt.hlines(1,0,len(list_data[6,t,:, 24]), colors='gray')

            #plt.ylim(0.985, 1.02)
            c +=1
        plt.savefig('./Output/STRF/STRF_'+name+'/STRF_Cell5_filled',bbox_inches='tight',dpi=300)

        c = 1
        plt.figure(figsize=(25,4))
        for t in range(n_t-10,170,-20):
            plt.subplot(1,7,c)
            y_data = list_data[7,t,:, 23]   
            plt.plot(x, y_data ,'-o', color='black')
            plt.fill_between(x, y_data ,0, where=y_data >0 , color='steelblue',alpha=0.5 )
            plt.fill_between(x, y_data ,0, where=y_data <=0 , color='tomato',alpha=0.5 )
            #plt.fill_between(x[off_part], list_data[6,t,off_part, 24]+((1-list_data[6,t,off_part, 24])),1 , color='tomato',alpha=0.5 )
            #plt.hlines(1,0,len(list_data[6,t,:, 24]), colors='gray')

            #plt.ylim(0.98, 1.025)
            c +=1
        plt.savefig('./Output/STRF/STRF_'+name+'/STRF_Cell7_filled',bbox_inches='tight',dpi=300)

print('Finish with creating!')



