import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os.path
#------------------------------------------------------------------------------
def calcTemplateMatch(X,Y):
    tm = 0
    normX = np.linalg.norm(X)
    normY = np.linalg.norm(Y)
    if (normX != 0 and normY !=0):
        tm = (np.dot(X,Y) / (normX*normY))
    return(tm)
#------------------------------------------------------------------------------
def calcTM_input(inpt):
    print(np.shape(inpt))
    n_patches,w,h,d = np.shape(inpt)
    tm = np.zeros((n_patches,n_patches))
    for i in range(n_patches):
        img_i = inpt[i,:,:,0] - inpt[i,:,:,1] 
        v_inpt_i = np.reshape(img_i,w*h)
        for j in range(n_patches):
            img_j = inpt[j,:,:,0] - inpt[j,:,:,1]
            v_inpt_j = np.reshape(img_j,w*h)
            tm[i,j] = calcTemplateMatch(v_inpt_i,v_inpt_j)
    np.save('./work/fluctuation_tm_input',tm)
    return(tm) 
#------------------------------------------------------------------------------
def estimateD_prime_old(spkCV1,inpt):
    print(np.shape(spkCV1))
    n_cells,n_patches,n_trails =np.shape(spkCV1)
    
    tmInpt = calcTM_input(inpt)
    #tmInpt = np.load('./work/fluctuation_tm_input.npy')

    n_patches = n_patches-1
    d_prime = np.zeros(n_patches)
    sort_idx = np.argsort(tmInpt[0,:]*-1)#sort index decreasing over tm
    for i_patch in range(1,n_patches+1):
        resp_proj1 = np.zeros(n_trails)
        resp_proj2 = np.zeros(n_trails)

        #mean over the acticity of the choosen patches
        mu1 = np.mean(spkCV1[:,0,:],1)
        mu2 = np.mean(spkCV1[:,sort_idx[i_patch],:],1)
    
        diffvec = mu1 - mu2

        for i_trails in range(n_trails):
            resp_proj1[i_trails] = np.dot(spkCV1[:,0,i_trails],diffvec)
            resp_proj2[i_trails] = np.dot(spkCV1[:,sort_idx[i_patch],i_trails],diffvec)

        mu_proj1 = np.mean(resp_proj1)
        var_proj1 = np.var(resp_proj1)   

        mu_proj2 = np.mean(resp_proj2)
        var_proj2 = np.var(resp_proj2)

        d_prime[i_patch-1] = np.abs(mu_proj1- mu_proj2) / np.sqrt( 0.5*(var_proj1 + var_proj2) )
    return(d_prime)
#------------------------------------------------------------------------------
def estimateD_prime(spkCV1):
    n_cells,n_patches,n_trails =np.shape(spkCV1)

    n_pairs = int(n_patches/2)
    d_prime = np.zeros((n_pairs,n_pairs))
    
    for i_patch in range(n_pairs):
        for j_patch in range(n_pairs):
            resp_proj1 = np.zeros(n_trails)
            resp_proj2 = np.zeros(n_trails)

            #mean over the acticity of the choosen patches
            mu1 = np.mean(spkCV1[:,i_patch,:],1)
            mu2 = np.mean(spkCV1[:,j_patch,:],1)
        
            diffvec = mu1 - mu2
            
            for i_trails in range(n_trails):
                resp_proj1[i_trails] = np.dot(spkCV1[:,i_patch,i_trails],diffvec)
                resp_proj2[i_trails] = np.dot(spkCV1[:,j_patch,i_trails],diffvec)

            mu_proj1 = np.mean(resp_proj1)
            var_proj1 = np.var(resp_proj1)   

            mu_proj2 = np.mean(resp_proj2)
            var_proj2 = np.var(resp_proj2)

            #if ((var_proj1 == 0.0 ) and (var_proj2 == 0.0)):
            #    print(i1,i2)
                
            if (i_patch == j_patch):
                d_prime[i_patch,j_patch] = 0.0
            else:
                d_prime[i_patch,j_patch] = np.abs(mu_proj1- mu_proj2) / np.sqrt( 0.5*(var_proj1 + var_proj2) )
    return(d_prime)
#------------------------------------------------------------------------------
def estimateD_primeSwitch(spkCV1):
    print(np.shape(spkCV1))
    n_cells,n_places,n_patches,n_trails =np.shape(spkCV1)

    d_prime = np.zeros((n_places,n_patches,n_patches))

    for p in range(n_places):    
        for i_patch in range(n_patches):
            for j_patch in range(n_patches):
                resp_proj1 = np.zeros(n_trails)
                resp_proj2 = np.zeros(n_trails)

                #mean over the acticity of the choosen patches
                mu1 = np.mean(spkCV1[:,p,i_patch,:],1)
                mu2 = np.mean(spkCV1[:,p,j_patch,:],1)
            
                diffvec = mu1 - mu2
                
                for i_trails in range(n_trails):
                    resp_proj1[i_trails] = np.dot(spkCV1[:,p,i_patch,i_trails],diffvec)
                    resp_proj2[i_trails] = np.dot(spkCV1[:,p,j_patch,i_trails],diffvec)

                mu_proj1 = np.mean(resp_proj1)
                var_proj1 = np.var(resp_proj1)   

                mu_proj2 = np.mean(resp_proj2)
                var_proj2 = np.var(resp_proj2)

                #if ((var_proj1 == 0.0 ) and (var_proj2 == 0.0)):
                #    print(i1,i2)
                    
                if (i_patch == j_patch):
                    d_prime[p,i_patch,j_patch] = 0.0
                else:
                    d_prime[p,i_patch,j_patch] = np.abs(mu_proj1- mu_proj2) / np.sqrt( 0.5*(var_proj1 + var_proj2) )
    return(d_prime)
#------------------------------------------------------------------------------
def startAnalysis():

    if not os.path.exists('./Output/Fluctuation'):
        os.mkdir('./Output/Fluctuation')

    #####################################
    #           load data               #
    #####################################

    spkCLGN = np.load('./work/fluctuation_frLGN.npy')
    spkCV1 = np.load('./work/fluctuation_frExc.npy')
    inpt = np.load('./work/fluctuation_Input.npy')

    gExcV1 = np.load('./work/fluctuation_V1_gExc.npy')
    gInhV1 = np.load('./work/fluctuation_V1_gInh.npy')

    inpt_switch = np.load('./work/fluctuationSwitch_Input.npy')

    spkV1_switch = np.load('./work/fluctuationSwitch_frExc.npy')
    d_prime_switch = estimateD_primeSwitch(spkV1_switch)
    np.save('./work/fluctuation_dPrime_switch',d_prime_switch)

    print('Finish with d-prime')
    tmInpt = calcTM_input(inpt)
    tmInpt = np.load('./work/fluctuation_tm_input.npy')

    #plt.figure()
    #plt.plot(d_prime_switch,'o-')
    #plt.savefig('./Output/d_prime.png')

    #i_sort = np.argsort(tmInpt[0,:]*-1)
    
#    plt.figure()
#    plt.scatter(tmInpt[0,i_sort[1:len(i_sort)]],d_prime)
#    plt.xlabel('cosine')
#    plt.ylabel('d prime')    
#    plt.savefig('./Output/dprime_TM.png')
    
    #spkCLGN /=8.0
    #spkCV1 /=8.0

    nbrLGNCells,nbrPatches,nbrSamples = np.shape(spkCLGN)
    nbrV1Cells = np.shape(spkCV1)[0]

    meanPerPatchLGN= np.zeros((nbrLGNCells,nbrPatches))
    stdPerPatchLGN = np.zeros((nbrLGNCells,nbrPatches))

    meanPerPatchV1= np.zeros((nbrV1Cells,nbrPatches))
    stdPerPatchV1 = np.zeros((nbrV1Cells,nbrPatches))

    meanPerPatchLGN= np.mean(spkCLGN,axis=2)
    stdPerPatchLGN = np.std(spkCLGN,axis=2)
    varPerPatchLGN = np.var(spkCLGN,axis=2)
    meanPerPatchV1 = np.mean(spkCV1,axis=2)
    stdPerPatchV1  = np.std(spkCV1,axis=2)
    varPerPatchV1  = np.var(spkCV1,axis=2)

    meangExcV1 = np.mean(gExcV1,axis=2)
    meangInhV1 = np.mean(gInhV1,axis=2)

    plt.figure()
    plt.scatter(meangExcV1,meangInhV1)
    plt.xlabel('g_Exc')
    plt.ylabel('g_Inh')
    plt.xlim(0.0,10.0)
    plt.ylim(0.0,10.0)
    plt.savefig('./Output/Fluctuation/gExc_to_gInh_scatter.png',bbox_inches='tight', pad_inches = 0.1,dpi=300)

    binSize = 125#75
    min_v=0.0#np.min(meangExcV1)
    max_v=np.max(meangExcV1)
    heatAll = np.zeros((binSize,binSize))
    extent = []
    xedg = np.zeros(binSize)
    yedg = np.zeros(binSize)
    for i in range(nbrV1Cells):
        idx = np.argsort(meangExcV1[i])
        heat,xedges,yedges = np.histogram2d(meangExcV1[i],meangInhV1[i],bins = binSize,range=([min_v,max_v],[min_v,max_v]))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        xedg = xedges
        yedg = yedges
        heatAll +=heat

    my_cmap = plt.get_cmap('jet')
    my_cmap.set_under('w')
    plt.figure()
    plt.imshow(heatAll.T,extent=extent,origin='lower',interpolation='none',aspect='auto',cmap=my_cmap,vmin=0.001)#,norm=LogNorm())
    cbar = mp.pyplot.colorbar()
    cbar.set_label('# of samples',fontsize=22)
    plt.xlabel('g_Exc',fontsize=22)
    plt.ylabel('g_Inh',fontsize=22)
    plt.xlim(0.0,10.0)
    plt.ylim(0.0,10.0)
    plt.savefig('./Output/Fluctuation/gExc_to_gInh_HIST.png',dpi=300,bbox_inches='tight', pad_inches = 0.1)

    gExc_bins = np.linspace(min_v,max_v,binSize)
    gInh_v = np.zeros(binSize-1)
    gExc_v = np.zeros(binSize-1)
    n_ele = np.zeros(binSize-1)
    for i in range(binSize-1):
        idx = np.where ((meangExcV1 > gExc_bins[i]) & (meangExcV1 <= gExc_bins[i+1]))
        n_ele[i] = np.shape(idx)[1]
        gExc_v[i]= np.mean(meangExcV1[idx])
        gInh_v[i]= np.mean(meangInhV1[idx])


    plt.figure()
    plt.scatter(gExc_v,gInh_v)
    plt.xlabel('g_Exc')
    plt.ylabel('g_Inh')
    plt.xlim(0.0)
    plt.ylim(0.0)
    plt.savefig('./Output/Fluctuation/gExc_to_gInh_MEAN.png',dpi=300,bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.scatter(gExc_v,gInh_v,c=n_ele)
    plt.xlabel('g_Exc')
    plt.ylabel('g_Inh')
    plt.xlim(0.0)
    plt.ylim(0.0)
    plt.colorbar()
    plt.savefig('./Output/Fluctuation/gExc_to_gInh_MEAN_NBRELE.png',dpi=300,bbox_inches='tight', pad_inches = 0.1)

    print(np.shape(varPerPatchLGN))
    
    varCLNG = stdPerPatchLGN /meanPerPatchLGN
    varCV1 = stdPerPatchV1 /meanPerPatchV1

    plt.figure()
    plt.imshow(stdPerPatchV1,interpolation='none')
    plt.colorbar()
    plt.ylabel('neuron Index')
    plt.xlabel('patch Index')
    plt.savefig('./Output/Fluctuation/fluctIMG.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.imshow(varCV1,interpolation='none')
    plt.colorbar()
    plt.ylabel('neuron Index')
    plt.xlabel('patch Index')
    plt.savefig('./Output/Fluctuation/fluctIMG_VarC.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.hist(np.mean(stdPerPatchLGN,axis=0))
    plt.xlabel('average '+r'$\sigma$ ')
    plt.savefig('./Output/Fluctuation/fluctPerPatch_STD_LGN.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.hist(np.mean(varPerPatchLGN,axis=0))
    plt.xlabel('average '+r'$var$ ')
    plt.savefig('./Output/Fluctuation/fluctPerPatch_VAR_LGN.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.hist(np.mean(stdPerPatchLGN,axis=1))
    plt.xlabel('average '+r'$\sigma$ ')
    plt.savefig('./Output/Fluctuation/fluctPerCell_STD_LGN.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.hist(np.mean(varPerPatchLGN,axis=1))
    plt.xlabel('average '+r'$var$ ')
    plt.savefig('./Output/Fluctuation/fluctPerCell_VAR_LGN.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.hist(np.mean(stdPerPatchV1,axis=0))
    plt.xlabel('average '+r'$\sigma$ ')
    plt.savefig('./Output/Fluctuation/fluctPerPatch_STD_V1.png',bbox_inches='tight', pad_inches = 0.1)

    #meanVarC =np.mean(varCV1,axis=0)
    #plt.figure()
    #plt.hist(meanVarC[~np.isnan(meanVarC)])
    #plt.xlabel('average '+r'$\frac{\sigma}{\overline{FR}}$ ')
    #plt.savefig('./Output/Fluctuation/fluctPerPatch_VarC_V1.png',bbox_inches='tight', pad_inches = 0.1)


    plt.figure()
    plt.hist(np.mean(varPerPatchV1,axis=0))
    plt.xlabel('average '+r'$var$ ')
    plt.savefig('./Output/Fluctuation/fluctPerPatch_VAR_V1.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.hist(np.mean(stdPerPatchV1,axis=1))
    plt.xlabel('average '+r'$\sigma$ ')
    plt.savefig('./Output/Fluctuation/fluctPerCell_STD_V1.png',bbox_inches='tight', pad_inches = 0.1)

    #meanVarC =np.mean(varCV1,axis=1)
    #plt.figure()
    #plt.hist(meanVarC[~np.isnan(meanVarC)])
    #plt.xlabel('average '+r'$\frac{\sigma}{\overline{FR}}$ ')
    #plt.savefig('./Output/Fluctuation/fluctPerCell_VarC_V1.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    plt.hist(np.mean(varPerPatchV1,axis=1))
    plt.xlabel('average '+r'$VAR$ ')
    plt.savefig('./Output/Fluctuation/fluctPerCell_VAR_V1.png',bbox_inches='tight', pad_inches = 0.1)


    slopeLGN,offset = np.polyfit(np.mean(meanPerPatchLGN,axis=1),np.mean(varPerPatchLGN,axis=1),1)
    print(slopeLGN)
    x = np.linspace(np.min(np.mean(meanPerPatchLGN,axis=1)),np.max(np.mean(meanPerPatchLGN,axis=1)))
    line = slopeLGN*x + offset

    fig = plt.figure()
    ax = plt.gca()
    #plt.loglog(x,line,'k-')
    ax.scatter(np.mean(meanPerPatchLGN,axis=1),np.mean(varPerPatchLGN,axis=1))
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    plt.title('mean(mean(FR)) to mean(var(FR)) per Cell')
    plt.savefig('./Output/Fluctuation/MeanFR_to_MeanVAR_LGN.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    for i in range(nbrLGNCells):
        plt.scatter(meanPerPatchLGN[i,:],varPerPatchLGN[i,:])
    plt.savefig('./Output/Fluctuation/MeanFR_to_MeanVAR_LGN_ALL.png',bbox_inches='tight', pad_inches = 0.1)


    slopeV1,offset = np.polyfit(np.mean(meanPerPatchV1,axis=1),np.mean(varPerPatchV1,axis=1),1)
    print(slopeV1)

    x = np.linspace(np.min(np.mean(meanPerPatchV1,axis=1)),np.max(np.mean(meanPerPatchV1,axis=1)))
    line = slopeV1*x + offset

    print(np.shape(np.mean(meanPerPatchV1,axis=1)))
    plt.figure()
    #plt.plot(x,line,'k-')
    plt.scatter(np.mean(meanPerPatchV1,axis=1),np.mean(varPerPatchV1,axis=1))
    plt.plot()
    plt.title('mean(mean(FR)) to mean(var(FR)) per Patch')
    #plt.xlim(1.0,2.8)
    #plt.ylim(0.0,0.8)
    plt.savefig('./Output/Fluctuation/MeanFR_to_MeanVAR_V1.png',bbox_inches='tight', pad_inches = 0.1)

    meanFR = np.zeros((nbrV1Cells,nbrPatches))
    varFR  = np.zeros((nbrV1Cells,nbrPatches))
    for i in range(nbrV1Cells):
        idx = np.argsort(meanPerPatchV1[i,:])
        meanFR[i] = meanPerPatchV1[i,idx]
        varFR[i] = varPerPatchV1[i,idx]
   
        
    patchesMean,intervallMean = np.histogram(meanFR[0],20,range=(np.min(meanFR),np.max(meanFR)))
    patchesVar,intervallVar = np.histogram(varFR[0],20,range=(np.min(varFR),np.max(varFR)))

    plt.figure()
    plt.scatter(intervallMean,intervallVar)
    plt.savefig('./Output/Fluctuation/MeanFR_to_MeanVAR_V1_TEST.png',bbox_inches='tight', pad_inches = 0.1)


    #calc the SNR after http://scholarpedia.org/article/Signal-to-noise_ratio_in_neuroscience
    snr =np.zeros(nbrV1Cells)
    for i in range(nbrV1Cells):
        snr[i] = np.sum(meanPerPatchV1[i]**2)/np.sum(varPerPatchV1[i,:])
    
    plt.figure()
    plt.hist(snr)
    plt.title('mean(SNR) = ' +str(np.round(np.mean(snr),2)))
    plt.xlabel('SNR')
    plt.ylabel('# of neurons')
    plt.savefig('./Output/Fluctuation/SNR_Hist.png',bbox_inches='tight', pad_inches = 0.1)

    plt.figure()
    #plt.plot(x,line,'k-')
    for i in range(nbrV1Cells):    
        plt.scatter(meanFR[i,:],varFR[i,:])
    #plt.xlim(1.0,2.8)
    #plt.ylim(0.0,0.8)
    plt.savefig('./Output/Fluctuation/MeanFR_to_MeanVAR_V1_ALL.png',bbox_inches='tight', pad_inches = 0.1)


    x = np.linspace(0,1)
    lineLGN = slopeLGN*x
    lineV1 = slopeV1*x
    plt.figure()
    plt.plot(x,lineLGN,'b-',label='LGN;slope: '+str(np.round(slopeLGN,3)))
    plt.plot(x,lineV1,'g--',label='V1;slope: '+str(np.round(slopeV1,3)))
    plt.legend()
    plt.savefig('./Output/Fluctuation/Slopes.png',bbox_inches='tight', pad_inches = 0.1)

#------------------------------------------------------------------------------
if __name__ == "__main__":
    startAnalysis()
