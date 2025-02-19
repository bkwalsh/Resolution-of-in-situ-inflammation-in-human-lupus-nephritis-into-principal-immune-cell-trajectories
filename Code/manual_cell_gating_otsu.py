###the packages we use
import time
start_time = time.perf_counter()
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import skimage.filters as filters
import seaborn as sns 

#%%

import matplotlib.font_manager as font_manager

# List all available fonts
# Check if Arial is available
fonts = [f.name for f in font_manager.fontManager.ttflist]
if "Arial" in fonts:
    plt.rcParams['font.family'] = 'Arial'
else:
    print("Arial font is not available. Using default font.")
#%%


def convert_mixed_dtype_to_string(df):
    """
    Convert columns with mixed data types to string data type.

    Parameters:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: Dataframe with mixed datatype columns converted to string.
    """
    
    # Identify columns with object dtype
    mixed_dtype_columns = df.columns[df.dtypes == 'object']
    
    # Convert those columns to string
    for col in mixed_dtype_columns:
        print('Mixed datatype in:',col)
        df[col] = df[col].astype(str)

    return df


def scatter_plot_save(saveName,col1,col2,plotTitle,thresh1Lim,thresh2Lim,df,alpha,s,columnsKeep,scaleColor,xLim,yLim,text,returnExport,showPlot,timed,literalLim,quantileThresholding,nPoints,hue,hueOrder,palette):
    
    matplotlib.rcParams.update({'font.size': 14,
                               'font.weight':'extra bold'})
    lineSize = 4
    ratio = 5 #for #x,y joint plots
    levels=5

    col1Temp = col1    
    col2Temp = col2
    col1 = [s for s in columnsKeep if col1 in s]
    col2 = [s for s in columnsKeep if col2 in s]
    

    if literalLim: ##use the literal range given versus the quartile
        x = xLim
        y = yLim

    else:
        x = (np.quantile(np.array(df[col1]),xLim[0]),np.quantile(np.array(df[col1]),xLim[1])) ##cd45
        y = (np.quantile(np.array(df[col2]),yLim[0]),np.quantile(np.array(df[col2]),yLim[1])) ##dapi


    
    if nPoints > df.shape[0]:
        nPoints = df.shape[0]
    
    
    matplotlib.rcParams['lines.linewidth'] = 9
    if hue != False:
        g = sns.JointGrid(data=df.sample(nPoints,random_state=42),x=col1[0], y=col2[0],ratio=ratio, space=.5,hue=hue,hue_order=hueOrder,palette=palette)
    else:
        g = sns.JointGrid(data=df.sample(nPoints,random_state=42),x=col1[0], y=col2[0],ratio=ratio, space=.5)
    g.plot_joint(sns.scatterplot, s=s, alpha=alpha)
    g.plot_joint(sns.kdeplot, zorder=0, levels=levels,fill=False,legend=False)
    g.plot_marginals(sns.histplot, kde=True,legend=False)
    g.ax_marg_y.tick_params(labeltop=True)
    g.ax_marg_y.grid(True, axis='x', ls=':')
    g.ax_marg_y.xaxis.set_major_locator(MaxNLocator(2))

    g.ax_marg_x.tick_params(labelleft=True)
    g.ax_marg_x.grid(True, axis='y', ls=':')
    g.ax_marg_x.yaxis.set_major_locator(MaxNLocator(2))
        # g.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)
    g.ax_joint.set_xscale('log')
    g.ax_joint.set_yscale('log')
    g.ax_joint.set_xlim(xLim) 
    g.ax_joint.set_ylim(yLim) 
    g.fig.set_figheight(15)
    g.fig.set_figwidth(15)
    # Remove the legend
    g.ax_joint.get_legend().remove()
    # plt.show()
    

    xtext=0.02
    if not text == False:
        plt.figtext(xtext ,.90, f"Input # of cells: {df.shape[0]}",fontsize = 16, ha='left', va='top',color='r',weight="extra bold", transform=ax.transAxes)
    if not thresh1Lim == False:
        
        if quantileThresholding: ##do we use quantiles or exact limit
            thresh1 = np.quantile(np.array(df[col1]),thresh1Lim)
            thresh1pos = df[np.array(df[col1])>=thresh1]
            thresh1neg = df[np.array(df[col1])<thresh1]
        else:
            thresh1 = thresh1Lim
            thresh1pos = df[np.array(df[col1])>=thresh1]
            thresh1neg = df[np.array(df[col1])<thresh1]
            
        # ax.vlines(x=thresh1,  linewidth=lineSize, color='black',ymin=y[0],ymax=y[-1])
        
        # plt.axvline(x=thresh1,  linewidth=lineSize, color='black')
        
        g.refline(x=thresh1,  linewidth=lineSize, color='black')
        
        
        if (not thresh1Lim == False) and (thresh2Lim == False) and (not text == False): ##we want to threshold only using xaxis and write text
            p =thresh1pos.shape[0]/df.shape[0]
            plt.figtext(xtext, .85, f"# {col1Temp}+ : {thresh1pos.shape[0]}, {p*100:.2f}%",
                    fontsize = 16, ha='left', va='top',color='r',weight="extra bold", transform=ax.transAxes)
            p=thresh1neg.shape[0]/df.shape[0]
            plt.figtext(xtext, .80, f"# {col1Temp}- : {thresh1neg.shape[0]}, {p*100:.2f}%",
                    fontsize = 16, ha='left', va='top',color='r',weight="extra bold", transform=ax.transAxes)  
        
    if not thresh2Lim == False:
        if quantileThresholding: ##do we use quantiles or exact limit
            thresh2 = np.quantile(np.array(df[col2]),thresh2Lim)
            thresh2pos = df[np.array(df[col2])>=thresh2]
            thresh2neg = df[np.array(df[col2])<thresh2]
        # ax.hlines(y=thresh2,  linewidth=lineSize, color='black',xmin=x[0],xmax=x[-1])
        else:
            thresh2 = thresh2Lim
            thresh2pos = df[np.array(df[col2])>=thresh2]
            thresh2neg = df[np.array(df[col2])<thresh2]
#         
        # plt.axhline(y=thresh2,  linewidth=lineSize, color='black')
        
        g.refline(y=thresh2,  linewidth=lineSize, color='black')
        
        if (thresh1Lim == False) and (not thresh2Lim == False) and (not text == False): ##we want to threshold only using xaxis and write text
            p =thresh2pos.shape[0]/df.shape[0]
            plt.figtext(xtext, .85, f"# {col2Temp}+ : {thresh2pos.shape[0]}, {p*100:.2f}%",
                    fontsize = 16, ha='left', va='top',color='r',weight="extra bold", transform=ax.transAxes)
            p=thresh2neg.shape[0]/df.shape[0]
            plt.figtext(xtext, .80, f"# {col2Temp}- : {thresh2neg.shape[0]}, {p*100:.2f}%",
                    fontsize = 16, ha='left', va='top',color='r',weight="extra bold", transform=ax.transAxes) 
        
    if (not thresh1Lim == False) and (not thresh2Lim == False): ##we want to apply both thresholds
            
        thresh1pos_thresh2pos= thresh1pos[np.array(thresh1pos[col2])>=thresh2]
        thresh1pos_thresh2neg= thresh1pos[np.array(thresh1pos[col2])<thresh2]
        thresh1neg_thresh2pos= thresh1neg[np.array(thresh1neg[col2])>=thresh2]
        thresh1neg_thresh2neg= thresh1neg[np.array(thresh1neg[col2])<thresh2]
        
        
        g.refline(x=thresh1,  linewidth=lineSize, color='black')
        g.refline(y=thresh2,  linewidth=lineSize, color='black')
        
        if (not thresh1Lim == False) and (not thresh2Lim == False) and (not text == False): ##we want to threshold into quadrants and plot text
            
            p =thresh1pos_thresh2pos.shape[0]/df.shape[0]
            plt.figtext(xtext, .85, f"# {col1Temp}+{col2Temp}+ : {thresh1pos_thresh2pos.shape[0]}, {p*100:.2f}%",
                    fontsize = 16, ha='left', va='top',color='r',weight="extra bold", transform=ax.transAxes)
            p=thresh1pos_thresh2neg.shape[0]/df.shape[0]
            plt.figtext(xtext, .80, f"# {col1Temp}+{col2Temp}- : {thresh1pos_thresh2neg.shape[0]}, {p*100:.2f}%",
                    fontsize = 16, ha='left', va='top',color='r',weight="extra bold", transform=ax.transAxes)
            p=thresh1neg_thresh2neg.shape[0]/df.shape[0]
            plt.figtext(xtext, .75, f"# {col1Temp}-{col2Temp}- : {thresh1neg_thresh2neg.shape[0]}, {p*100:.2f}%",
                    fontsize = 16, ha='left', va='top',color='r',weight="extra bold", transform=ax.transAxes)
            p=thresh1neg_thresh2pos.shape[0]/df.shape[0]
            plt.figtext(xtext, .70, f"# {col1Temp}-{col2Temp}+ : {thresh1neg_thresh2pos.shape[0]}, {p*100:.2f}%",
                    fontsize = 16, ha='left', va='top',color='r',weight="extra bold", transform=ax.transAxes)

    
    x = df[col1].values
    y = df[col2].values
    x = [item for sublist in x for item in sublist]
    y = [item for sublist in y for item in sublist]
    


    for axis in ['top','bottom','left','right']:
        g.ax_joint.spines[axis].set_linewidth(6)  # Thicker spines

    if thresh1Lim:
        g.refline(x=thresh1Lim, linewidth=6, color='black')  # Increase `linewidth` for thicker lines

    if thresh2Lim:
        g.refline(y=thresh2Lim, linewidth=6, color='black')
    if showPlot:
        if timed:
            plt.show(block=False)
            plt.pause(5)
            plt.close() 
        else: 
            plt.show()
    
    g.savefig(saveName,dpi=300)
    plt.close()
    
    if (not thresh1Lim == False) and (not thresh2Lim == False) and (returnExport): ##we want to threshold into quadrants and return exports
        return(thresh1pos_thresh2pos,thresh1neg_thresh2pos,thresh1pos_thresh2neg,thresh1neg_thresh2neg)

    if (not thresh1Lim == False) and (thresh2Lim == False) and (returnExport): ##we want to threshold  xaxis  and return exports
        return(thresh1pos,thresh1neg)
        
    if (thresh1Lim == False) and (not thresh2Lim == False) and (returnExport): ##we want to threshold  xaxis  and return exports
        return(thresh2pos,thresh2neg)
    
    

def scatter_density_plot_save(saveName,col1,col2,plotTitle,thresh1Lim,thresh2Lim,df,vmax,plotDPI,columnsKeep,scaleColor,xLim,yLim,text,returnExport,showPlot,timed,normalizedDisp,literalLim,quantileThresholding):
    
    matplotlib.rcParams.update({'font.size': 16,
                               'font.weight':'extra bold'})
    lineSize = 3

    col1Temp = col1    
    col2Temp = col2
    col1 = [s for s in columnsKeep if col1 in s]
    col2 = [s for s in columnsKeep if col2 in s]
    

    if literalLim: ##use the literal range given versus the quartile
        x = xLim
        y = yLim

    else:
        x = (np.quantile(np.array(df[col1]),xLim[0]),np.quantile(np.array(df[col1]),xLim[1])) ##cd45
        y = (np.quantile(np.array(df[col2]),yLim[0]),np.quantile(np.array(df[col2]),yLim[1])) ##dapi

    fig = plt.figure(figsize=(12,12))

    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    ax.set_title(plotTitle,weight="extra bold",fontsize = 16)
    ax.set_xlabel(col1[0],weight="extra bold",fontsize = 14)
    ax.set_ylabel(col2[0],weight="extra bold",fontsize = 14)

    xtext=0.02
    if not text == False:
        plt.figtext(xtext ,.90, f"Input # of cells: {df.shape[0]}",fontsize = 16, ha='left', va='top',color='r',weight="extra bold", transform=ax.transAxes)
    if not thresh1Lim == False:

        if quantileThresholding: ##do we use quantiles or exact limit
            thresh1 = np.quantile(np.array(df[col1]),thresh1Lim)
            thresh1pos = df[np.array(df[col1])>=thresh1]
            thresh1neg = df[np.array(df[col1])<thresh1]
        else:
            thresh1 = thresh1Lim
            thresh1pos = df[np.array(df[col1])>=thresh1]
            thresh1neg = df[np.array(df[col1])<thresh1]
        ax.vlines(x=thresh1,  linewidth=lineSize, color='black',ymin=y[0],ymax=y[-1])
        
        if (not thresh1Lim == False) and (thresh2Lim == False) and (not text == False): ##we want to threshold only using xaxis and write text
            p =thresh1pos.shape[0]/df.shape[0]
            plt.figtext(xtext, .85, f"# {col1Temp}+ : {thresh1pos.shape[0]}, {p*100:.2f}%",
                    fontsize = 16, ha='left', va='top',color='r',weight="extra bold", transform=ax.transAxes)
            p=thresh1neg.shape[0]/df.shape[0]
            plt.figtext(xtext, .80, f"# {col1Temp}- : {thresh1neg.shape[0]}, {p*100:.2f}%",
                    fontsize = 16, ha='left', va='top',color='r',weight="extra bold", transform=ax.transAxes)  
        
    if not thresh2Lim == False:
        if quantileThresholding: ##do we use quantiles or exact limit
            thresh2 = np.quantile(np.array(df[col2]),thresh2Lim)
            thresh2pos = df[np.array(df[col2])>=thresh2]
            thresh2neg = df[np.array(df[col2])<thresh2]
        else:
            thresh2 = thresh2Lim
            thresh2pos = df[np.array(df[col2])>=thresh2]
            thresh2neg = df[np.array(df[col2])<thresh2]
            
        ax.hlines(y=thresh2,  linewidth=lineSize, color='black',xmin=x[0],xmax=x[-1])
        
        if (thresh1Lim == False) and (not thresh2Lim == False) and (not text == False): ##we want to threshold only using xaxis and write text
            p =thresh2pos.shape[0]/df.shape[0]
            plt.figtext(xtext, .85, f"# {col2Temp}+ : {thresh2pos.shape[0]}, {p*100:.2f}%",
                    fontsize = 16, ha='left', va='top',color='r',weight="extra bold", transform=ax.transAxes)
            p=thresh2neg.shape[0]/df.shape[0]
            plt.figtext(xtext, .80, f"# {col2Temp}- : {thresh2neg.shape[0]}, {p*100:.2f}%",
                    fontsize = 16, ha='left', va='top',color='r',weight="extra bold", transform=ax.transAxes) 
        
    if (not thresh1Lim == False) and (not thresh2Lim == False): ##we want to apply both thresholds
            
        thresh1pos_thresh2pos= thresh1pos[np.array(thresh1pos[col2])>=thresh2]
        thresh1pos_thresh2neg= thresh1pos[np.array(thresh1pos[col2])<thresh2]
        thresh1neg_thresh2pos= thresh1neg[np.array(thresh1neg[col2])>=thresh2]
        thresh1neg_thresh2neg= thresh1neg[np.array(thresh1neg[col2])<thresh2]
        
        if (not thresh1Lim == False) and (not thresh2Lim == False) and (not text == False): ##we want to threshold into quadrants and plot text
            
            p =thresh1pos_thresh2pos.shape[0]/df.shape[0]
            plt.figtext(xtext, .85, f"# {col1Temp}+{col2Temp}+ : {thresh1pos_thresh2pos.shape[0]}, {p*100:.2f}%",
                    fontsize = 16, ha='left', va='top',color='r',weight="extra bold", transform=ax.transAxes)
            p=thresh1pos_thresh2neg.shape[0]/df.shape[0]
            plt.figtext(xtext, .80, f"# {col1Temp}+{col2Temp}- : {thresh1pos_thresh2neg.shape[0]}, {p*100:.2f}%",
                    fontsize = 16, ha='left', va='top',color='r',weight="extra bold", transform=ax.transAxes)
            p=thresh1neg_thresh2neg.shape[0]/df.shape[0]
            plt.figtext(xtext, .75, f"# {col1Temp}-{col2Temp}- : {thresh1neg_thresh2neg.shape[0]}, {p*100:.2f}%",
                    fontsize = 16, ha='left', va='top',color='r',weight="extra bold", transform=ax.transAxes)
            p=thresh1neg_thresh2pos.shape[0]/df.shape[0]
            plt.figtext(xtext, .70, f"# {col1Temp}-{col2Temp}+ : {thresh1neg_thresh2pos.shape[0]}, {p*100:.2f}%",
                    fontsize = 16, ha='left', va='top',color='r',weight="extra bold", transform=ax.transAxes)
    if (not normalizedDisp == False): 
        
        norm = ImageNormalize(vmin=0., vmax=vmax, stretch=LogStretch(normalizedDisp))
        density = ax.scatter_density(x=np.array(df[col1]), y=np.array(df[col2]),cmap=scaleColor,vmax=vmax,dpi=plotDPI,norm=norm)
        
    else:
        density = ax.scatter_density(x=np.array(df[col1]), y=np.array(df[col2]),cmap=scaleColor,vmax=vmax,dpi=plotDPI)
        
        
    ax.set_xlim(x) ##cd45

    ax.set_ylim(y) #dapi
        

    fig.colorbar(density, label='Number of points per pixel')
    if showPlot:
        if timed:
            plt.show(block=False)
            plt.pause(2)
            plt.close() 
        else: 
            plt.show()
        
    fig.savefig(saveName,dpi=100)
    plt.close()
    
    if (not thresh1Lim == False) and (not thresh2Lim == False) and (returnExport): ##we want to threshold into quadrants and return exports
        return(thresh1pos_thresh2pos,thresh1neg_thresh2pos,thresh1pos_thresh2neg,thresh1neg_thresh2neg)

    if (not thresh1Lim == False) and (thresh2Lim == False) and (returnExport): ##we want to threshold  xaxis  and return exports
        return(thresh1pos,thresh1neg)
        
    if (thresh1Lim == False) and (not thresh2Lim == False) and (returnExport): ##we want to threshold  xaxis  and return exports
        return(thresh2pos,thresh2neg)
    
def otsu_thresholding(df,feature,classes):
   
    thresholds2 = filters.threshold_multiotsu(df[feature].to_numpy(), classes=2)
    thresholds3 = filters.threshold_multiotsu(df[feature].to_numpy(), classes=3)
    thresholds4 = filters.threshold_multiotsu(df[feature].to_numpy(), classes=4)
    thresholds5 = filters.threshold_multiotsu(df[feature].to_numpy(), classes=5)

    
    if classes==2:
        thresh = thresholds2
    if classes==3:
        thresh = thresholds3
    if classes==4:
        thresh = thresholds4
    if classes==5:
        thresh = thresholds5
        
    if classes==6:    
        
        thresh = filters.threshold_multiotsu(df[feature].to_numpy(), classes=6)

    return(thresh)
    
    
#%%        
        
def main():
    
#%%    
    
# if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r',
            '--read_dir',
            type=str,
            default='cell_patches_1um',
            help=''
            )
    parser.add_argument('-rc',
            '--cell_info_dir',
            type=str
            # default='S09-24464',
            # help='sample to calculate expression for'
            )
    parser.add_argument('-w',
            '--write_dir',
            type=str,
            default='channel_expression_csvs',
            help=''
            )
 

    args,unparsed = parser.parse_known_args()

    
    
    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)
    if not os.path.exists(args.write_dir+"manual_gating_QC/"):
        os.mkdir(args.write_dir+"manual_gating_QC/")
    if not os.path.exists(args.write_dir+"gating_plots/"):
        os.mkdir(args.write_dir+"gating_plots/")

    df = convert_mixed_dtype_to_string(pd.DataFrame(pd.read_csv(args.read_dir,index_col=False)))
    df =df.fillna(0)

    dfInfo = convert_mixed_dtype_to_string(pd.DataFrame(pd.read_csv(args.cell_info_dir,index_col=False)))
    dfInfo = dfInfo.loc[df.index]

    
    df[[ 'disease_cohort', 'AccessionNumber', 'TissueType']] = dfInfo[[ 'disease_cohort', 'AccessionNumber', 'TissueType']]
    
  
    
    columnsKeep = list(df.columns)
    
    cellLabelDict = {}
    
   
    ##based on https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    # (0.01, '#440053'),
    (0.05, '#404388'),
    (0.1, '#2a788e'),
    (0.3, '#21a784'),
    (0.4, '#78d151'),
    (0.5, '#fde624'),
    (0.7,'#d60f1c'),
    (0.9,'#e30bdc'),
    (1,'#ffd6fe'),
], N=150)  #256
    
   
   

      #%%
    showPlot = True
    
    nPoints = 40000
    s=2
    alpha=1
    hueOrder = np.unique(df['disease_cohort'])
    hueDict = {}
    thresholdDict ={}

    hueDict = {'Normal_Kidney':[0,1,0,1],'Lupus_Nephritis':[0,0,1,1], 'Renal_Allograft':[1,0,1,1]}
    
   
    
    NucleusAreapDAPIp,NucleusAreanDAPIp,NucleusAreapDAPIn,NucleusAreanDAPIn = scatter_plot_save(saveName=args.write_dir+"manual_gating_QC/"+'DAPI_Nucleus-area.tif',
                              col1="Nucleus-area",
                              col2="DAPI_Int-mean",
                              plotTitle="DAPI v. Nucleus",
                              thresh1Lim=50,
                              thresh2Lim=20,
                              df=df,
                              alpha=alpha,
                                s=s,
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                              xLim=[1,2500],
                              yLim=[10,250],
                              text=False,
                              returnExport=True,showPlot=showPlot,timed=True,literalLim=True,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict) 
                                              # hue='disease_cohort')  
                                              
    thresholdDict['Nucleus-area'] = 50                                          
    thresholdDict['DAPI_Int-mean'] = 20       


 #%%
    gating_df = pd.DataFrame(index=df.index)
    gating_df['Nucleus-area_DAPI_Int-mean'] ='none' 
    
    gating_df.loc[NucleusAreapDAPIp.index,'Nucleus-area_DAPI_Int-mean'] = "NucleusAreapDAPIp"
    gating_df.loc[NucleusAreanDAPIp.index,'Nucleus-area_DAPI_Int-mean'] = "NucleusAreanDAPIp"
    
    gating_df.loc[NucleusAreapDAPIn.index,'Nucleus-area_DAPI_Int-mean'] = "NucleusAreapDAPIn"
    gating_df.loc[NucleusAreanDAPIn.index,'Nucleus-area_DAPI_Int-mean'] = "NucleusAreanDAPIn"
    
                               
                                                  
#%%
      
    
    #4 due to large spread
    #1st beacause we want any positivity 
    CD45Thresh = otsu_thresholding(df = NucleusAreapDAPIp,feature="CD45_Int-mean",classes=5) ##what if we apply otsu's method to the CD45?

    #%%
    CD45pDAPIp,CD45nDAPIp = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'CD45_DAPI.tif',
                              col1="CD45_Int-mean",
                              col2="DAPI_Int-mean",
                              plotTitle="DAPI v. CD45, Major axis+ gated",
                              thresh1Lim=CD45Thresh[1],
                              thresh2Lim=False,
                              df=NucleusAreapDAPIp,
                              alpha=alpha,
                                s=s,
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                              xLim=[0.01,300],
                              yLim=[30,275],
                              text=False,
                              returnExport=True,showPlot=showPlot,timed=True,literalLim=True,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict) 
                                              # hue='disease_cohort')   
                                              
                                                                                
    thresholdDict['CD45_Int'] = CD45Thresh[1]       
    
    #%%
    
    gating_df['CD45_Int-mean_DAPI_Int-mean'] ='none' 
    gating_df.loc[CD45pDAPIp.index,'CD45_Int-mean_DAPI_Int-mean'] = "CD45pDAPIp"
    gating_df.loc[CD45nDAPIp.index,'CD45_Int-mean_DAPI_Int-mean'] = "CD45nDAPIp"
    
    ##this is a set of additional immune gates to account for the low expression of CD45
   
    CD3Thresh = otsu_thresholding(df = CD45nDAPIp,feature="CD3_Int-mean",classes=2) ##large spread
    HLAIIThresh = otsu_thresholding(df = CD45nDAPIp,feature="HLAII_Int-mean",classes=4) ##large spread

    
    CD45nHLAIIpCD3p,CD45nHLAIInCD3p,CD45nHLAIIpCD3n,CD45nHLAIInCD3n = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45-_CD3_HLAII.tif',
                              col1="HLAII_Int-mean",
                              col2="CD3_Int-mean",
                              # col2="Nucleus-area",
                              plotTitle="CD3 v. HLAII ;CD45-DAPI+ gated ",
                              thresh1Lim=HLAIIThresh[0],
                              # thresh1Quantile=0.99,
                              thresh2Lim=CD3Thresh[0],
                              df=CD45nDAPIp,
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                              # xLim=[0.00,0.0055],                      
                              xLim=[.1,250],
                            yLim=[.1,250],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict) 

    thresholdDict['CD45n_CD3_Int'] = CD3Thresh[0]
    thresholdDict['CD45n_HLAII_Int'] = HLAIIThresh[0]
    
    #%%
    
    
    
    gating_df['HLAII_Int-mean_CD3_Int-mean'] ='none' 
    
    gating_df.loc[CD45nHLAIIpCD3p.index,'HLAII_Int-mean_CD3_Int-mean'] = "CD45nHLAIIpCD3p"
    gating_df.loc[CD45nHLAIInCD3p.index,'HLAII_Int-mean_CD3_Int-mean'] = "CD45nHLAIInCD3p"
    
    gating_df.loc[CD45nHLAIIpCD3n.index,'HLAII_Int-mean_CD3_Int-mean'] = "CD45nHLAIIpCD3n"
    gating_df.loc[CD45nHLAIInCD3n.index,'HLAII_Int-mean_CD3_Int-mean'] = "CD45nHLAIInCD3n"
    
    
    
   #%%
    CD14Thresh = otsu_thresholding(df = CD45nHLAIIpCD3n,feature="CD14_Int-mean",classes=4) ##large spread
    CD16Thresh = otsu_thresholding(df = CD45nHLAIIpCD3n,feature="CD16_Int-mean",classes=2) ##large spread
    
   
    CD45nHLAIIpCD14pCD16p,CD45nHLAIIpCD14nCD16p,CD45nHLAIIpCD14pCD16n,CD45nHLAIIpCD14nCD16n = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45-CD3-HLAII+_CD14_CD16.tif',
                             col1="CD14_Int-mean",
                             col2="CD16_Int-mean",
                             # col2="Nucleus-area",
                             plotTitle="CD14 v. CD16 ;CD45-DAPI+HLAII+CD3- gated ",
                             thresh1Lim=CD14Thresh[0],
                             # thresh1Quantile=0.99,
                             thresh2Lim=CD16Thresh[0],
                             df=CD45nHLAIIpCD3n,
                             columnsKeep=columnsKeep,
                             scaleColor=white_viridis,
                             # xLim=[0.00,0.0055],                      
                             xLim=[1,500],
                             yLim=[1,500],
                             text=False,
                             returnExport=True,
                             showPlot=showPlot,
                             timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict) 

  
    thresholdDict['CD45nCD3nHLAIIp_CD14_Int'] = CD14Thresh[0]
    thresholdDict['CD45nCD3nHLAIIp_CD16_Int'] = CD16Thresh[0]
   #%%
   
    gating_df['CD45nCD3nHLAIIp_CD14_Int-mean_CD16_Int-mean'] ='none' 
    
    gating_df.loc[CD45nHLAIIpCD14pCD16p.index,'CD45nCD3nHLAIIp_CD14_Int-mean_CD16_Int-mean'] = "CD45nHLAIIpCD14pCD16p"
    gating_df.loc[CD45nHLAIIpCD14nCD16p.index,'CD45nCD3nHLAIIp_CD14_Int-mean_CD16_Int-mean'] = "CD45nHLAIIpCD14nCD16p"
    
    gating_df.loc[CD45nHLAIIpCD14pCD16n.index,'CD45nCD3nHLAIIp_CD14_Int-mean_CD16_Int-mean'] = "CD45nHLAIIpCD14pCD16n"
    gating_df.loc[CD45nHLAIIpCD14nCD16n.index,'CD45nCD3nHLAIIp_CD14_Int-mean_CD16_Int-mean'] = "CD45nHLAIIpCD14nCD16n"
    
    
    
   #%%
    for idx in list(CD45nHLAIIpCD14pCD16p.index):
       cellLabelDict[idx] = "HLAII+_Monocytes"
    for idx in list(CD45nHLAIIpCD14nCD16p.index):
       cellLabelDict[idx] = "CD16+_macrophages"
   
    for idx in list(CD45nHLAIIpCD14pCD16n.index):
       cellLabelDict[idx] = "CD14+_macrophages"
   
   
   #%%
    CD14Thresh = otsu_thresholding(df = CD45nHLAIInCD3n,feature="CD14_Int-mean",classes=3) ##large spread
    CD16Thresh = otsu_thresholding(df = CD45nHLAIInCD3n,feature="CD16_Int-mean",classes=4) ##large spread
    
    CD45nHLAIInCD14pCD16p,CD45nHLAIInCD14nCD16p,CD45nHLAIInCD14pCD16n,CD45nHLAIInCD14nCD16n = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45-CD3-HLAII-_CD14_CD16.tif',
                              col1="CD14_Int-mean",
                              col2="CD16_Int-mean",
                              # col2="Nucleus-area",
                              plotTitle="CD14 v. CD16 ;CD45-DAPI+HLAII-CD3- gated ",
                              thresh1Lim=CD14Thresh[0],
                              # thresh1Quantile=0.99,
                              thresh2Lim=CD16Thresh[1],
                              df=CD45nHLAIInCD3n,
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                              # xLim=[0.00,0.0055],                      
                              xLim=[1,200],
                            yLim=[0.1,200],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict) 

    thresholdDict['CD45nCD3nHLAIIn_CD14_Int'] = CD14Thresh[0]
    thresholdDict['CD45nCD3nHLAIIn_CD16_Int'] = CD16Thresh[1]
  
   
   #%%  
    gating_df['CD45nCD3nHLAIIn_CD14_Int-mean_CD16_Int-mean'] ='none' 
   
    gating_df.loc[CD45nHLAIInCD14pCD16p.index,'CD45nCD3nHLAIIn_CD14_Int-mean_CD16_Int-mean'] = "CD45nHLAIInCD14pCD16p"
    gating_df.loc[CD45nHLAIInCD14nCD16p.index,'CD45nCD3nHLAIIn_CD14_Int-mean_CD16_Int-mean'] = "CD45nHLAIInCD14nCD16p"
   
    gating_df.loc[CD45nHLAIInCD14pCD16n.index,'CD45nCD3nHLAIIn_CD14_Int-mean_CD16_Int-mean'] = "CD45nHLAIInCD14pCD16n"
    gating_df.loc[CD45nHLAIInCD14nCD16n.index,'CD45nCD3nHLAIIn_CD14_Int-mean_CD16_Int-mean'] = "CD45nHLAIInCD14nCD16n"
   
    
    
   #%%
    for idx in list(CD45nHLAIInCD14pCD16p.index):
        cellLabelDict[idx] = "HLAII-_Monocytes"
    
   #%%
    ##we dont apply otsu's here because we want less stringent thresholds, most cells should be nonimmune 
    
    dfTemp = pd.concat([CD45nHLAIInCD14nCD16n, CD45nHLAIIpCD14nCD16n])
    dfTemp = pd.concat([dfTemp, CD45nHLAIInCD14pCD16n])
    dfTemp = pd.concat([dfTemp, CD45nHLAIInCD14nCD16p])
    
    MUC1Thresh = otsu_thresholding(df = dfTemp,feature="MUC1_Int-mean",classes=5) ##large spread
    CD10Thresh = otsu_thresholding(df = dfTemp,feature="CD10_Int-mean",classes=5) ##large spread
    

    MUC1pCD10p,MUC1nCD10p,MUC1pCD10n,MUC1nCD10n = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45-_MUC1_CD10.tif',
                              col1="MUC1_Int-mean",
                              col2="CD10_Int-mean",
                              # col2="Nucleus-area",
                              plotTitle="MUC1 v. CD10 ;CD45-DAPI+ gated (All Tubules)",
                              thresh1Lim=MUC1Thresh[0],
                              # thresh1Quantile=0.99,
                              thresh2Lim=CD10Thresh[0],
                              df=dfTemp,
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                              # xLim=[0.00,0.0055],                      
                              xLim=[0.0001,500],
                              yLim=[0.0001,500],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict) 


    thresholdDict['CD45n_MUC1_Int'] = MUC1Thresh[0]
    thresholdDict['CD45n_CD10_Int'] = CD10Thresh[0]

#%%

    gating_df['MUC1_Int-mean_CD10_Int-mean'] ='none' 
   
    gating_df.loc[MUC1pCD10p.index,'MUC1_Int-mean_CD10_Int-mean'] = "MUC1pCD10p"
    gating_df.loc[MUC1nCD10p.index,'MUC1_Int-mean_CD10_Int-mean'] = "MUC1nCD10p"
   
    gating_df.loc[MUC1pCD10n.index,'MUC1_Int-mean_CD10_Int-mean'] = "MUC1pCD10n"
    gating_df.loc[MUC1nCD10n.index,'MUC1_Int-mean_CD10_Int-mean'] = "MUC1nCD10n"
   
   

    #%%  
    for idx in list(MUC1pCD10p.index):
        cellLabelDict[idx] = "Proximal_tubules"   
    for idx in list(MUC1nCD10p.index):
        cellLabelDict[idx] = "Proximal_tubules"
    for idx in list(MUC1pCD10n.index):
        cellLabelDict[idx] = "Distal_collecting_tubules"
   
    #%%
    
    CD31Thresh = otsu_thresholding(df = MUC1nCD10n,feature="CD31_Int-mean",classes=5) ##large spread
    Claudin1Thresh  = otsu_thresholding(df = MUC1nCD10n,feature="Claudin1_Int-mean",classes=5) ##large spread
    
    
    CD31pClaudin1p,CD31nClaudin1p,CD31pClaudin1n,CD31nClaudin1n = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+MUC1-CD10-_CD31_Claudin1.tif',
                              col1="CD31_Int-mean",
                              col2="Claudin1_Int-mean",
                              # col2="Nucleus-area",
                              plotTitle="CD31 v. Claudin1 ;CD45-DAPI+MUC1-CD10- gated ",
                              thresh1Lim=CD31Thresh[0],
                              # thresh1Quantile=0.99,
                              thresh2Lim=2,
                              df=MUC1nCD10n,
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                                xLim=[0.01,200],
                              yLim=[0.01,200],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict) 
    
    thresholdDict['CD45n_CD31_Int'] = CD31Thresh[0]
    thresholdDict['CD45n_Claudin1_Int'] = Claudin1Thresh[0]

    
    #%%
    
    gating_df['CD31_Int-mean_Claudin1_Int-mean'] ='none' 
   
    gating_df.loc[CD31pClaudin1p.index,'CD31_Int-mean_Claudin1_Int-mean'] = "CD31pClaudin1p"
    gating_df.loc[CD31nClaudin1p.index,'CD31_Int-mean_Claudin1_Int-mean'] = "CD31nClaudin1p"
    gating_df.loc[CD31pClaudin1n.index,'CD31_Int-mean_Claudin1_Int-mean'] = "CD31pClaudin1n"
    gating_df.loc[CD31nClaudin1n.index,'CD31_Int-mean_Claudin1_Int-mean'] = "CD31nClaudin1n"
   
 
    
    #%%
    
    
    for idx in list(CD31pClaudin1p.index):
        cellLabelDict[idx] = "Endothelial_cells"
    for idx in list(CD31nClaudin1p.index):
        cellLabelDict[idx] = "Inflamed_tubule"
    for idx in list(CD31pClaudin1n.index):
        cellLabelDict[idx] = "Endothelial_cells"
        
    #%%
    ##marcus wants this as a QC check
    currentCells = list(CD45nHLAIInCD14pCD16n.index)
    referenceCells = list(MUC1pCD10p.index) +list(MUC1nCD10p.index) +list(MUC1pCD10n.index)+list(CD31pClaudin1p.index)+list(CD31nClaudin1p.index)+list(CD31pClaudin1n.index)
    notCells = [x for x in currentCells if x not in referenceCells]
    
    
   #%%   

    HLAIIThresh = otsu_thresholding(df = CD45pDAPIp,feature="HLAII_Int-mean",classes=4) ##large spread
    CD3Thresh = otsu_thresholding(df = CD45pDAPIp,feature="CD3_Int-mean",classes=5) ##large spread
    

    
    CD45pHLAIIpCD3p,CD45pHLAIInCD3p,CD45pHLAIIpCD3n,CD45pHLAIInCD3n = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+_HLAII_CD3.tif',
                              col1="HLAII_Int-mean",
                              col2="CD3_Int-mean",
                              plotTitle="CD3 v. HLAII , gated on DAPI+CD45+",
                              thresh1Lim=HLAIIThresh[0],
                              thresh2Lim=CD3Thresh[0],
                              df=CD45pDAPIp,
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                                xLim=[.1,250],
                              yLim=[.1,250],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict) 
    
    

    thresholdDict['CD45p_CD3_Int'] = CD3Thresh[0]
    thresholdDict['CD45p_HLAII_Int'] = HLAIIThresh[0]
    #%%
    
    gating_df['CD45p_HLAII_Int-mean_CD3_Int-mean'] ='none' 
   
    gating_df.loc[CD45pHLAIIpCD3p.index,'CD45p_HLAII_Int-mean_CD3_Int-mean'] = "CD45pHLAIIpCD3p"
    gating_df.loc[CD45pHLAIInCD3p.index,'CD45p_HLAII_Int-mean_CD3_Int-mean'] = "CD45pHLAIInCD3p"
    gating_df.loc[CD45pHLAIIpCD3n.index,'CD45p_HLAII_Int-mean_CD3_Int-mean'] = "CD45pHLAIIpCD3n"
    gating_df.loc[CD45pHLAIInCD3n.index,'CD45p_HLAII_Int-mean_CD3_Int-mean'] = "CD45pHLAIInCD3n"
   
  
    
 #%%
    CD4Thresh = otsu_thresholding(df = pd.concat([CD45pHLAIInCD3p, CD45pHLAIIpCD3p]),feature="CD4_Int-mean",classes=2) ##large spread
    CD8Thresh = otsu_thresholding(df = pd.concat([CD45pHLAIInCD3p, CD45pHLAIIpCD3p]),feature="CD8_Int-mean",classes=5) ##large spread
    

    CD4pCD8p,CD4nCD8p,CD4pCD8n,CD4nCD8n = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3+HLAII-CD4_CD8.tif',
                              col1="CD4_",
                              col2="CD8_",
                              plotTitle="CD4 CD8 gated on CD45+CD3+HLAII-",
                              thresh1Lim=CD4Thresh[0],
                              thresh2Lim=CD8Thresh[1],
                              df=pd.concat([CD45pHLAIInCD3p, CD45pHLAIIpCD3p]), ##HLAII+ T cells are activated DONT IGNORE INDEX
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                               xLim=[1,300],
                              yLim=[0.1,300],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict) 
    
    thresholdDict['CD45pHLAIInpCD3p_CD4_Int'] = CD4Thresh[0]
    thresholdDict['CD45pHLAIInpCD3p_CD8_Int'] = CD8Thresh[1]
    
    
    #%%
    gating_df['CD45pHLAIInpCD3p_CD4_Int-mean_CD8_Int-mean'] ='none' 
    gating_df.loc[CD4pCD8p.index,'CD45pHLAIInpCD3p_CD4_Int-mean_CD8_Int-mean'] = "CD4pCD8p"
    gating_df.loc[CD4nCD8p.index,'CD45pHLAIInpCD3p_CD4_Int-mean_CD8_Int-mean'] = "CD4nCD8p"
    gating_df.loc[CD4pCD8n.index,'CD45pHLAIInpCD3p_CD4_Int-mean_CD8_Int-mean'] = "CD4pCD8n"
    gating_df.loc[CD4nCD8n.index,'CD45pHLAIInpCD3p_CD4_Int-mean_CD8_Int-mean'] = "CD4nCD8n"
    
    
    #%%
    for idx in list(CD4pCD8p.index):
        cellLabelDict[idx] = "CD4+CD8+_Tcell"

        
    for idx in list(CD4nCD8p.index):
        cellLabelDict[idx] = "CD8+_Tcell"
    for idx in list(CD4pCD8n.index):
        cellLabelDict[idx] = "CD4+_Tcell"
    
    
    #%%
    ICOSThresh = otsu_thresholding(df = CD4pCD8n,feature="ICOS_Int-mean",classes=3) ##large spread
    PD1Thresh = otsu_thresholding(df = CD4pCD8n,feature="PD1_Int-mean",classes=3) ##large spread
    
  
    
    CD4pCD8nICOSpPD1p,CD4pCD8nICOSnPD1p,CD4pCD8nICOSpPD1n,CD4pCD8nICOSnPD1n = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3+HLAII-CD4+CD8-ICOS_PD1.tif',
                              col1="ICOS_",
                              col2="PD1_",
                              plotTitle="ICOS PD1 gated on DAPI+CD45+CD3+HLAII-CD4+CD8-ICOS_PD1",
                              thresh1Lim=ICOSThresh[0],
                              thresh2Lim=PD1Thresh[0],
                              df=CD4pCD8n, ##HLAII+ T cells are activated DONT IGNORE INDEX
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                               xLim=[3,260],
                              yLim=[3,260],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict) 
    
    thresholdDict['CD45pHLAIInpCD3pCD4_ICOS_Int'] = ICOSThresh[0]
    thresholdDict['CD45pHLAIInpCD3pCD4_PD1_Int'] = PD1Thresh[1]
    
    #%%
    gating_df['CD45pHLAIInpCD3pCD4_ICOS_Int_PD1_Int'] ='none' 
    gating_df.loc[CD4pCD8nICOSpPD1p.index,'CD45pHLAIInpCD3pCD4_ICOS_Int_PD1_Int'] = "CD4pCD8nICOSpPD1p"
    gating_df.loc[CD4pCD8nICOSnPD1p.index,'CD45pHLAIInpCD3pCD4_ICOS_Int_PD1_Int'] = "CD4pCD8nICOSnPD1p"
    gating_df.loc[CD4pCD8nICOSpPD1n.index,'CD45pHLAIInpCD3pCD4_ICOS_Int_PD1_Int'] = "CD4pCD8nICOSpPD1n"
    gating_df.loc[CD4pCD8nICOSnPD1n.index,'CD45pHLAIInpCD3pCD4_ICOS_Int_PD1_Int'] = "CD4pCD8nICOSnPD1n"
    
    #%%
    
    
    for idx in list(CD4pCD8nICOSpPD1p.index):
        cellLabelDict[idx] = "CD4+ICOS+PD1+_Tcell"

    for idx in list(CD4pCD8nICOSnPD1p.index):
        cellLabelDict[idx] = "CD4+PD1+_Tcell"
    for idx in list(CD4pCD8nICOSpPD1n.index):
        cellLabelDict[idx] = "CD4+ICOS+_Tcell" 
        
        
    
    #%%
    FOXP3Thresh = otsu_thresholding(df = CD4pCD8nICOSnPD1n,feature="Foxp3_Int-mean",classes=2) ##large spread
    CD103Thresh = otsu_thresholding(df = CD4pCD8nICOSnPD1n,feature="CD103_Int-mean",classes=4) ##large spread
  
    
    #%%

    FOXP3pCD103p,FOXP3nCD103p,FOXP3pCD103n,FOXP3nCD103n = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3+HLAIICD4+CD8-ICOS-PD1-_FoxP3_CD103.tif',
                              col1="Foxp3_",
                              col2="CD103_",
                              plotTitle="FOXP3/CD103 gated on DAPI+CD45+CD3+HLAIICD4+CD8-ICOS-PD1-",
                              thresh1Lim=FOXP3Thresh[0],
                              thresh2Lim=CD103Thresh[1],
                              df=CD4pCD8nICOSnPD1n, ##HLAII+ T cells are activated DONT IGNORE INDEX
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                               xLim=[1,230],
                              yLim=[1,230],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict) 
    
    thresholdDict['CD45pHLAIInCD3pCD4_Foxp3_Int'] = FOXP3Thresh[0]
    thresholdDict['CD45pHLAIInCD3pCD4_CD103_Int'] = CD103Thresh[1]
    
    #%%
    for idx in list(FOXP3pCD103p.index):
        cellLabelDict[idx] = "CD4+_intraepithelial_Tcell"
    for idx in list(FOXP3nCD103p.index):
        cellLabelDict[idx] = "CD4+_intraepithelial_Tcell"
    for idx in list(FOXP3pCD103n.index):
        cellLabelDict[idx] = "CD4+Foxp3+_Tcell"
    #%%
    
    gating_df['CD4pCD8n_Foxp3_Int_CD103_Int'] ='none' 
    gating_df.loc[FOXP3pCD103p.index,'CD4pCD8n_Foxp3_Int_CD103_Int'] = "FOXP3pCD103p"
    gating_df.loc[FOXP3nCD103p.index,'CD4pCD8n_Foxp3_Int_CD103_Int'] = "FOXP3nCD103p"
    gating_df.loc[FOXP3pCD103n.index,'CD4pCD8n_Foxp3_Int_CD103_Int'] = "FOXP3pCD103n"
    gating_df.loc[FOXP3nCD103n.index,'CD4pCD8n_Foxp3_Int_CD103_Int'] = "FOXP3nCD103n"
    
    
    #%%
    ICOSThresh = otsu_thresholding(df = CD4nCD8p,feature="ICOS_Int-mean",classes=3) ##large spread
    PD1Thresh = otsu_thresholding(df = CD4nCD8p,feature="PD1_Int-mean",classes=5) ##large spread
    
    
    CD4nCD8pICOSpPD1p,CD4nCD8pICOSnPD1p,CD4nCD8pICOSpPD1n,CD4nCD8pICOSnPD1n = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3+HLAII-CD4-CD8+ICOS_PD1.tif',
                              col1="ICOS_",
                              col2="PD1_",
                              plotTitle="ICOS PD1 gated on DAPI+CD45+CD3+HLAII-CD4-CD8+ICOS_PD1",
                              thresh1Lim=ICOSThresh[1],
                              thresh2Lim=PD1Thresh[2],
                              df=CD4nCD8p, ##HLAII+ T cells are activated DONT IGNORE INDEX
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                               xLim=[3,260],
                              yLim=[3,260],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict) 
    
    thresholdDict['CD45pHLAIInCD3pCD8_ICOS_Int'] = ICOSThresh[1]
    thresholdDict['CD45pHLAIInCD3pCD4_PD1_Int'] = PD1Thresh[2]
    
    #%%
    for idx in list( CD4nCD8pICOSpPD1p.index):
        cellLabelDict[idx] = "CD8+ICOS+PD1+_Tcell"
    for idx in list( CD4nCD8pICOSpPD1n.index):
        cellLabelDict[idx] = "CD8+ICOS+_Tcell"
    for idx in list( CD4nCD8pICOSnPD1p.index):
        cellLabelDict[idx] = "CD8+PD1+_Tcell"
    #%%
    gating_df['CD45pHLAIInpCD3pCD8_ICOS_Int_PD1_Int'] ='none' 
    gating_df.loc[CD4nCD8pICOSpPD1p.index,'CD45pHLAIInpCD3pCD8_ICOS_Int_PD1_Int'] = "CD4nCD8pICOSpPD1p"
    gating_df.loc[CD4nCD8pICOSnPD1p.index,'CD45pHLAIInpCD3pCD8_ICOS_Int_PD1_Int'] = "CD4nCD8pICOSnPD1p"
    gating_df.loc[CD4nCD8pICOSpPD1n.index,'CD45pHLAIInpCD3pCD8_ICOS_Int_PD1_Int'] = "CD4nCD8pICOSpPD1n"
    gating_df.loc[CD4nCD8pICOSnPD1n.index,'CD45pHLAIInpCD3pCD8_ICOS_Int_PD1_Int'] = "CD4nCD8pICOSnPD1n"
    
    
    
    #%%
    
    FOXP3Thresh = otsu_thresholding(df = CD4nCD8pICOSnPD1n,feature="Foxp3_Int-mean",classes=3) ##large spread
    CD103Thresh = otsu_thresholding(df = CD4nCD8pICOSnPD1n,feature="CD103_Int-mean",classes=3) ##large spread
 
    FOXP3pCD103p,FOXP3nCD103p,FOXP3pCD103n,FOXP3nCD103n = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3+HLAII-CD4-CD8+ICOS-PD1-_FoxP3_CD103.tif',
                              col1="Foxp3_",
                              col2="CD103_",
                              plotTitle="FOXP3 gated on DAPI+CD45+CD3+HLAII-CD4-CD8+ICOS-PD1-",
                              thresh1Lim=FOXP3Thresh[1],
                              thresh2Lim=CD103Thresh[0],
                              df=CD4nCD8pICOSnPD1n, ##HLAII+ T cells are activated DONT IGNORE INDEX
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                              xLim=[1,230],
                              yLim=[1,230],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict) 
    
    thresholdDict['CD45pHLAIInCD3pCD8_Foxp3_Int'] = FOXP3Thresh[1]
    thresholdDict['CD45pHLAIInCD3pCD8_CD103_Int'] = CD103Thresh[0]
    #%%
    
    for idx in list(FOXP3pCD103p.index):
        cellLabelDict[idx] = "CD8+_intraepithelial_Tcell"
    for idx in list(FOXP3nCD103p.index):
        cellLabelDict[idx] = "CD8+_intraepithelial_Tcell"
    for idx in list(FOXP3pCD103n.index):
        cellLabelDict[idx] = "CD8+Foxp3+_Tcell"
    #%%
    
    gating_df['CD4nCD8p_Foxp3_Int_CD103_Int'] ='none' 
    gating_df.loc[FOXP3pCD103p.index,'CD4nCD8p_Foxp3_Int_CD103_Int'] = "FOXP3pCD103p"
    gating_df.loc[FOXP3nCD103p.index,'CD4nCD8p_Foxp3_Int_CD103_Int'] = "FOXP3nCD103p"
    gating_df.loc[FOXP3pCD103n.index,'CD4nCD8p_Foxp3_Int_CD103_Int'] = "FOXP3pCD103n"
    gating_df.loc[FOXP3nCD103n.index,'CD4nCD8p_Foxp3_Int_CD103_Int'] = "FOXP3nCD103n"
        
        #%%
    temp =  pd.concat([FOXP3nCD103n,FOXP3nCD103p ,CD4nCD8pICOSnPD1p])  
        
        
        
    GZMKThresh = otsu_thresholding(df = temp,feature="GZMK_Int-mean",classes=3) ##large spread
    GZMBThresh = otsu_thresholding(df = temp,feature="GZMB_Int-mean",classes=3) ##large spread
    scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3+HLAII-CD4-CD8+ICOS-PD1+-FoxP3-_GZMB_GZMK.tif',
                              col1="GZMK_",
                              col2="GZMB_",
                              plotTitle="GZMK v GZMB gated on DAPI+CD45+CD3+HLAII-CD4-CD8+ICOS-PD1-FOXP3+",
                              thresh1Lim=GZMKThresh[0],
                              thresh2Lim=GZMBThresh[0],
                              df=temp, ##HLAII+ T cells are activated DONT IGNORE INDEX
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                               xLim=[2,230],
                              yLim=[2,230],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict) 
    
    
    #%%
        
    GZMKThresh = otsu_thresholding(df = FOXP3nCD103n,feature="GZMK_Int-mean",classes=3) ##large spread
    GZMBThresh = otsu_thresholding(df = FOXP3nCD103n,feature="GZMB_Int-mean",classes=3) ##large spread
    scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3+HLAII-CD4-CD8+ICOS-PD1-FoxP3-CD103-_GZMB_GZMK.tif',
                              col1="GZMK_",
                              col2="GZMB_",
                              plotTitle="GZMK v GZMB gated on DAPI+CD45+CD3+HLAII-CD4-CD8+ICOS-PD1-FOXP3-CD103-",
                              thresh1Lim=GZMKThresh[0],
                              thresh2Lim=GZMBThresh[0],
                              df=FOXP3nCD103n, ##HLAII+ T cells are activated DONT IGNORE INDEX
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                               xLim=[2,230],
                              yLim=[2,230],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict) 
    
    
    #%%
    ##
   
    MUC1Thresh = otsu_thresholding(df = CD4pCD8p,feature="MUC1_Int-mean",classes=6) ##large spread
    CD10Thresh = otsu_thresholding(df = CD4pCD8p,feature="CD10_Int-mean",classes=6) ##large spread
  

    
    MUC1pCD10p,MUC1nCD10p,MUC1pCD10n,MUC1nCD10n = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3+HLAII-CD4+CD8+MUC1_CD10.tif',
                              col1="MUC1_Int-mean",
                              col2="CD10_Int-mean",
                              plotTitle="Muc1gated on CD45+CD3+HLAII-CD4+CD8+ ",
                              thresh1Lim=MUC1Thresh[0],
                              thresh2Lim=CD10Thresh[0],
                              df=CD4pCD8p, ##HLAII+ T cells are activated DONT IGNORE INDEX
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                               xLim=[0.1,300],
                              yLim=[0.1,300],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict) 
  
    
    thresholdDict['CD45pCD4pCD8p_MUC1_Int'] = MUC1Thresh[0]
    thresholdDict['CD45pCD4pCD8p_CD10_Int'] = CD10Thresh[0]
    
    #%%
    
    
    
    for idx in list(MUC1pCD10p.index): ##we only expect the RBCS to be positive for everything
        cellLabelDict[idx] = "RBCs"
    for idx in list(MUC1nCD10p.index):
        cellLabelDict[idx] = "RBCs"
    for idx in list(MUC1pCD10n.index):
        cellLabelDict[idx] = "RBCs"
    # print(asdadasdsaas)
    
    #%%
    
    gating_df['CD45pCD4pCD8p_MUC1_Int_CD10_Int'] ='none' 
    gating_df.loc[MUC1pCD10p.index,'CD45pCD4pCD8p_MUC1_Int_CD10_Int'] = "MUC1pCD10p"
    gating_df.loc[MUC1nCD10p.index,'CD45pCD4pCD8p_MUC1_Int_CD10_Int'] = "MUC1nCD10p"
    gating_df.loc[MUC1pCD10n.index,'CD45pCD4pCD8p_MUC1_Int_CD10_Int'] = "MUC1pCD10n"
    gating_df.loc[MUC1nCD10n.index,'CD45pCD4pCD8p_MUC1_Int_CD10_Int'] = "MUC1nCD10n"
    

    #%%
    CD56Thresh = otsu_thresholding(df = CD4nCD8n,feature="CD56_Int-mean",classes=5) ##large spread
    TCRDThresh = otsu_thresholding(df = CD4nCD8n,feature="TCRD_Int-mean",classes=5) ##large spread
    

    CD56pTCRgdp,CD56nTCRgdp,CD56pTCRgdn,CD56nTCRgdn = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3+HLAII-CD4-CD8-.tif',
                              col1="CD56_",
                              col2="TCRD_Int-mean",
                              plotTitle="CD56 v TCRgd gated on CD3+HLAII-CD4-CD8-",
                              thresh1Lim=CD56Thresh[1],
                              thresh2Lim=TCRDThresh[1],
                              df=CD4nCD8n,
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                               xLim=[0.5,250],
                              yLim=[0.5,250],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict)  
    
    
    thresholdDict['CD45pCD4nCD8n_CD56_Int'] = CD56Thresh[0]
    thresholdDict['CD45pCD4nCD8n_TCRD_Int'] = TCRDThresh[1]
   
#%%
    gating_df['CD45pCD4nCD8n_CD56_Int_TCRD_Int'] ='none' 
    gating_df.loc[CD56pTCRgdp.index,'CD45pCD4nCD8n_CD56_Int_TCRD_Int'] = "CD56pTCRgdp"
    gating_df.loc[CD56nTCRgdp.index,'CD45pCD4nCD8n_CD56_Int_TCRD_Int'] = "CD56nTCRgdp"
    gating_df.loc[CD56pTCRgdn.index,'CD45pCD4nCD8n_CD56_Int_TCRD_Int'] = "CD56pTCRgdn"
    gating_df.loc[CD56nTCRgdn.index,'CD45pCD4nCD8n_CD56_Int_TCRD_Int'] = "CD56nTCRgdn"
    

#%%
    GZMKThresh = otsu_thresholding(df = CD56nTCRgdp,feature="GZMK_Int-mean",classes=3) ##large spread
    GZMBThresh = otsu_thresholding(df = CD56nTCRgdp,feature="GZMB_Int-mean",classes=3) ##large spread
    scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3+HLAII-CD4-CD8-CD56-TCRgd+_GZMB_GZMK.tif',
                              col1="GZMK_",
                              col2="GZMB_",
                              plotTitle="GZMK v GZMB gated on DAPI+CD45+CD3+HLAII-CD4-CD8-CD56-TCRgd+",
                              thresh1Lim=GZMKThresh[0],
                              thresh2Lim=GZMBThresh[0],
                              df=CD56nTCRgdp, ##HLAII+ T cells are activated DONT IGNORE INDEX
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                               xLim=[1,230],
                              yLim=[1,230],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict) 
     
  
   #%% 
    scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3+HLAII-CD4-CD8-CD56-TCRgd-CD3+.tif',
                              col1="CD3_",
                              col2="DAPI",
                              plotTitle="CD3 v DAPI gated on DAPI+CD45+CD3+HLAII-CD4-CD8-CD56-TCRgd-CD3+",
                              thresh1Lim=25,
                              thresh2Lim=40,
                              df=CD56nTCRgdn,
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                               xLim=[5,250],
                              yLim=[5,250],
                              text=False,
                              returnExport=False,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict)  
    
   
    for idx in list(CD56pTCRgdp.index):
        cellLabelDict[idx] = "TCRgd_Tcell"
    
    for idx in list(CD56nTCRgdp.index):
        cellLabelDict[idx] = "TCRgd_Tcell"
    
    for idx in list(CD56pTCRgdn.index):
        cellLabelDict[idx] = "NK_Tcell"
        
    for idx in list(CD56nTCRgdn.index):
        cellLabelDict[idx] = "CD3+_Tcell"    
        
        #%%
    
    CD14Thresh = otsu_thresholding(df = CD45pHLAIInCD3n,feature="CD14_Int-mean",classes=2) ##large spread
    CD16Thresh = otsu_thresholding(df = CD45pHLAIInCD3n,feature="CD16_Int-mean",classes=2) ##large spread
    
  
        
    CD45pHLAIInCD14pCD16p,CD45pHLAIInCD14nCD16p,CD45pHLAIInCD14pCD16n,CD45pHLAIInCD14nCD16n = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3-HLAII-CD14_CD16.tif',
                              col1="CD14_",
                              col2="CD16_",
                              plotTitle="CD14 v CD16 gated on CD45+CD3-HLAII-",
                              thresh1Lim=CD14Thresh[0],
                              thresh2Lim=CD16Thresh[0],
                              df=CD45pHLAIInCD3n,
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                                xLim=[1,200],
                              yLim=[0.1,200],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict)      

    
    thresholdDict['CD45pHLAIInCD3n_CD14_Int'] = CD14Thresh[0]
    thresholdDict['CD45pHLAIInCD3n_CD16_Int'] = CD16Thresh[0]
   
    
    for idx in list(CD45pHLAIInCD14pCD16p.index):
        cellLabelDict[idx] = "HLAII-_Monocytes"
    for idx in list(CD45pHLAIInCD14pCD16n.index):
        cellLabelDict[idx] = "HLAII-_Monocytes"
    for idx in list(CD45pHLAIInCD14nCD16p.index):
        cellLabelDict[idx] = "Neutrophils"
    
    #%%
    gating_df['CD45pHLAIInCD3n_CD14_Int_CD16_Int'] ='none' 
    gating_df.loc[CD45pHLAIInCD14pCD16p.index,'CD45pHLAIInCD3n_CD14_Int_CD16_Int'] = "CD45pHLAIInCD14pCD16p"
    gating_df.loc[CD45pHLAIInCD14nCD16p.index,'CD45pHLAIInCD3n_CD14_Int_CD16_Int'] = "CD45pHLAIInCD14nCD16p"
    gating_df.loc[CD45pHLAIInCD14pCD16n.index,'CD45pHLAIInCD3n_CD14_Int_CD16_Int'] = "CD45pHLAIInCD14pCD16n"
    gating_df.loc[CD45pHLAIInCD14nCD16n.index,'CD45pHLAIInCD3n_CD14_Int_CD16_Int'] = "CD45pHLAIInCD14nCD16n"
    
    
    #%%
    ###maybe lower this one?
    CD56Thresh = otsu_thresholding(df = CD45pHLAIInCD14nCD16n,feature="CD56_Int-mean",classes=5) ##large spread
    CD138Thresh = otsu_thresholding(df = CD45pHLAIInCD14nCD16n,feature="CD138_Int-mean",classes=2) ##large spread
    

  
    CD56pCD138p,CD56nCD138p,CD56pCD138n,CD56nCD138n = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3-HLAII-CD14-CD16-_CD56_CD138.tif',
                              col1="CD56_",
                              col2="CD138_",
                              plotTitle="CD138 v CD56 gated on CD45+CD3-HLAII-CD16-CD16-",
                              thresh1Lim=CD56Thresh[1],
                              thresh2Lim=CD138Thresh[0],
                              df=CD45pHLAIInCD14nCD16n,
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                               xLim=[1,175],
                              yLim=[1,175],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict)            
    


    thresholdDict['CD45pHLAIInCD3nCD14nCD16n_CD56_Int'] = CD56Thresh[1]
    thresholdDict['CD45pHLAIInCD3nCD14nCD16n_CD138_Int'] = CD138Thresh[0]
    
    for idx in list(CD56pCD138p.index):
        cellLabelDict[idx] = "plasma_cells"
    for idx in list(CD56nCD138p.index):
        cellLabelDict[idx] = "plasma_cells"

        
    for idx in list(CD56pCD138n.index):
        cellLabelDict[idx] = "NK_cells"    
        
    #%%
    gating_df['CD45pHLAIInCD3n_CD14_Int_CD16_Int'] ='none' 
    gating_df.loc[CD45pHLAIInCD14pCD16p.index,'CD45pHLAIInCD3n_CD14_Int_CD16_Int'] = "CD45pHLAIInCD14pCD16p"
    gating_df.loc[CD45pHLAIInCD14nCD16p.index,'CD45pHLAIInCD3n_CD14_Int_CD16_Int'] = "CD45pHLAIInCD14nCD16p"
    gating_df.loc[CD45pHLAIInCD14pCD16n.index,'CD45pHLAIInCD3n_CD14_Int_CD16_Int'] = "CD45pHLAIInCD14pCD16n"
    gating_df.loc[CD45pHLAIInCD14nCD16n.index,'CD45pHLAIInCD3n_CD14_Int_CD16_Int'] = "CD45pHLAIInCD14nCD16n"
   


    #%%
    CD14Thresh = otsu_thresholding(df = CD45pHLAIIpCD3n,feature="CD14_Int-mean",classes=5) ##large spread
    CD16Thresh = otsu_thresholding(df = CD45pHLAIIpCD3n,feature="CD16_Int-mean",classes=5) ##large spread
  
    CD45pHLAIIpCD14pCD16p,CD45pHLAIIpCD14nCD16p,CD45pHLAIIpCD14pCD16n,CD45pHLAIIpCD14nCD16n = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3-HLAII+CD14_CD16.tif',
                              col1="CD14_",
                              col2="CD16_",
                              plotTitle="CD14 v CD16 gated on CD45+CD3-HLAII+ (macrophages?)",
                              thresh1Lim=CD14Thresh[1],
                              thresh2Lim=CD16Thresh[1],
                              df=CD45pHLAIIpCD3n,
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                              xLim=[1,500],
                              yLim=[1,500],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict)                       
    
    thresholdDict['CD45pHLAIIpCD3n_CD14_Int'] = CD14Thresh[1]
    thresholdDict['CD45pHLAIIpCD3n_CD16_Int'] = CD16Thresh[1]
    #%%
    for idx in list(CD45pHLAIIpCD14pCD16p.index):
        cellLabelDict[idx] = "HLAII+_Monocytes"
    
    # for idx in list(CD14pCD16p.index):
    #     cellLabelDict[idx] = "HLAII+_monocytes"
    for idx in list(CD45pHLAIIpCD14nCD16p.index):
        cellLabelDict[idx] = "CD16+_macrophages"
    
    for idx in list(CD45pHLAIIpCD14pCD16n.index):
        cellLabelDict[idx] = "CD14+_macrophages"
        
        
        #%%
    gating_df['CD45pHLAIIpCD3n_CD14_Int_CD16_Int'] ='none' 
    gating_df.loc[CD45pHLAIIpCD14pCD16p.index,'CD45pHLAIIpCD3n_CD14_Int_CD16_Int'] = "CD45pHLAIIpCD14pCD16p"
    gating_df.loc[CD45pHLAIIpCD14nCD16p.index,'CD45pHLAIIpCD3n_CD14_Int_CD16_Int'] = "CD45pHLAIIpCD14nCD16p"
    gating_df.loc[CD45pHLAIIpCD14pCD16n.index,'CD45pHLAIIpCD3n_CD14_Int_CD16_Int'] = "CD45pHLAIIpCD14pCD16n"
    gating_df.loc[CD45pHLAIIpCD14nCD16n.index,'CD45pHLAIIpCD3n_CD14_Int_CD16_Int'] = "CD45pHLAIIpCD14nCD16n"
    
        
        
        #%%
        
    MERTKThresh = otsu_thresholding(df = pd.concat([CD45nHLAIIpCD14pCD16n, CD45pHLAIIpCD14pCD16n]) ,feature="MERTK_Int-mean",classes=3) ##large spread
   
        
    CD163Thresh = otsu_thresholding(df = pd.concat([CD45nHLAIIpCD14pCD16n, CD45pHLAIIpCD14pCD16n]),feature="CD163_Int-mean",classes=3) ##large spread
   
    HLAIIpCD14pCD16nMERTKpCD163p,HLAIIpCD14pCD16nMERTKnCD163p,HLAIIpCD14pCD16nMERTKpCD163n,HLAIIpCD14pCD16nMERTKnCD163n = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3-HLAII+CD14+CD16-_MERTK_CD163.tif',
                            col1="MERTK_Int-mean",
                            col2="CD163_Int-mean",
                            # col2="Nucleus-area",
                            plotTitle="MERTK v. CD163 ;DAPI+CD45+CD3-HLAII+CD14+CD16 gated ",
                            thresh1Lim=MERTKThresh[0],
                            # thresh1Quantile=0.99,
                            thresh2Lim=CD163Thresh[0],
                            df=pd.concat([CD45nHLAIIpCD14pCD16n, CD45pHLAIIpCD14pCD16n]),
                            columnsKeep=columnsKeep,
                            scaleColor=white_viridis,
                            # xLim=[0.00,0.0055],                      
                            xLim=[0.1,250],
                            yLim=[0.1,250],
                            text=False,
                            returnExport=True,
                            showPlot=showPlot,
                            timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict) 
    thresholdDict['CD45pHLAIIpCD3nCD14p_MERTK_Int'] = MERTKThresh[0]
    thresholdDict['CD45pHLAIIpCD3nCD14p_CD163_Int'] = CD163Thresh[0]
       
   #%%
    for idx in list(HLAIIpCD14pCD16nMERTKnCD163p.index):
        cellLabelDict[idx] = "CD14+CD163+_macrophages"
    for idx in list(HLAIIpCD14pCD16nMERTKpCD163n.index):
        cellLabelDict[idx] = "CD14+MERTK+_macrophages"    
    for idx in list(HLAIIpCD14pCD16nMERTKpCD163p.index):
        cellLabelDict[idx] = "CD14+CD163+MERTK+_macrophages"  
   #%%
    gating_df['CD45pHLAIIpCD3nCD14p_MERTK_Int_CD163_Int'] ='none' 
    gating_df.loc[HLAIIpCD14pCD16nMERTKpCD163p.index,'CD45pHLAIIpCD3nCD14p_MERTK_Int_CD163_Int'] = "HLAIIpCD14pCD16nMERTKpCD163p"
    gating_df.loc[HLAIIpCD14pCD16nMERTKnCD163p.index,'CD45pHLAIIpCD3nCD14p_MERTK_Int_CD163_Int'] = "HLAIIpCD14pCD16nMERTKnCD163p"
    gating_df.loc[HLAIIpCD14pCD16nMERTKpCD163n.index,'CD45pHLAIIpCD3nCD14p_MERTK_Int_CD163_Int'] = "HLAIIpCD14pCD16nMERTKpCD163n"
    gating_df.loc[HLAIIpCD14pCD16nMERTKnCD163n.index,'CD45pHLAIIpCD3nCD14p_MERTK_Int_CD163_Int'] = "HLAIIpCD14pCD16nMERTKnCD163n"
    
        
        #%%
    MERTKThresh = otsu_thresholding(df = pd.concat([CD45nHLAIIpCD14nCD16p, CD45pHLAIIpCD14nCD16p]) ,feature="MERTK_Int-mean",classes=3) ##large spread
    CD163Thresh = otsu_thresholding(df = pd.concat([CD45nHLAIIpCD14nCD16p, CD45pHLAIIpCD14nCD16p]),feature="CD163_Int-mean",classes=3) ##large spread
    
   
    scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3-HLAII+CD14-CD16+_MERTK_CD163.tif',
                            col1="MERTK_Int-mean",
                            col2="CD163_Int-mean",
                            # col2="Nucleus-area",
                            plotTitle="MERTK v. CD163 ;DAPI+CD45+CD3-HLAII+CD14-CD16+ gated ",
                            thresh1Lim=MERTKThresh[0],
                            # thresh1Quantile=0.99,
                            thresh2Lim=CD163Thresh[0],
                            df=pd.concat([CD45nHLAIIpCD14nCD16p, CD45pHLAIIpCD14nCD16p]),
                            columnsKeep=columnsKeep,
                            scaleColor=white_viridis,
                            # xLim=[0.00,0.0055],                      
                            xLim=[0.1,250],
                            yLim=[0.1,250],
                            text=False,
                            returnExport=True,
                            showPlot=showPlot,
                            timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict) 

  
        
        #%%
        
        
    CD20Thresh = otsu_thresholding(df = CD45pHLAIIpCD14nCD16n,feature="CD20_Int-mean",classes=3) ##large spread
    CD138Thresh = otsu_thresholding(df = CD45pHLAIIpCD14nCD16n,feature="CD138_Int-mean",classes=3) ##large spread
    
    
    CD20pCD138p,CD20nCD138p,CD20pCD138n,CD20nCD138n = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3-HLAII+CD14-CD16-_CD20_CD138.tif',
                              col1="CD20_",
                              col2="CD138_",
                              plotTitle="CD20 v CD138 gated on CD45+CD3-HLAII+CD14-CD16- ",
                              thresh1Lim=CD20Thresh[0],
                              thresh2Lim=CD138Thresh[0],
                              df=CD45pHLAIIpCD14nCD16n,
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                              xLim=[.1,300],
                              yLim=[.1,300],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict)                   
    
    thresholdDict['CD45pHLAIIpCD14nCD16n_CD20_Int'] = CD20Thresh[0]
    thresholdDict['CD45pHLAIIpCD14nCD16n_CD138_Int'] = CD138Thresh[0]
        
    for idx in list(CD20pCD138p.index):
        cellLabelDict[idx] = "Plasmablasts"
    for idx in list(CD20nCD138p.index):
        cellLabelDict[idx] = "Plasmablasts"
    
    for idx in list(CD20pCD138n.index):
        cellLabelDict[idx] = "Bcell"
        
        
        #%%
        
    gating_df['CD45pHLAIIpCD14nCD16n_CD20_Int_CD138_Int'] ='none' 
    gating_df.loc[CD20pCD138p.index,'CD45pHLAIIpCD14nCD16n_CD20_Int_CD138_Int'] = "CD20pCD138p"
    gating_df.loc[CD20nCD138p.index,'CD45pHLAIIpCD14nCD16n_CD20_Int_CD138_Int'] = "CD20nCD138p"
    gating_df.loc[CD20pCD138n.index,'CD45pHLAIIpCD14nCD16n_CD20_Int_CD138_Int'] = "CD20pCD138n"
    gating_df.loc[CD20nCD138n.index,'CD45pHLAIIpCD14nCD16n_CD20_Int_CD138_Int'] = "CD20nCD138n"
     
        
        
    
    #%%
    
    BDCA2Thresh = otsu_thresholding(df = CD20nCD138n,feature="BDCA2_Int-mean",classes=3) ##large spread
    CD103Thresh = otsu_thresholding(df = CD20nCD138n,feature="CD103_Int-mean",classes=5) ##large spread


    
    BDCA2pCD103p,BDCA2nCD103p,BDCA2pCD103n,BDCA2nCD103n = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3-HLAII+CD14-CD16-CD20-CD138-_BDCA2_CD103.tif',
                              col1="BDCA2_",
                              col2="CD103_",
                              plotTitle="BDCA2 v CD103 gated on HLAII+CD14-CD16-CD20-CD138- (pDCs) ",
                              thresh1Lim=BDCA2Thresh[0],
                              thresh2Lim=CD103Thresh[1],
                              df=CD20nCD138n,
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                              xLim=[1,150],
                              yLim=[1,150],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict)                
    
    
    thresholdDict['CD45pHLAIIpCD14nCD16nCD20nCD138n_BDCA2_Int'] = BDCA2Thresh[0]
    thresholdDict['CD45pHLAIIpCD14nCD16nCD20nCD138n_CD103_Int'] = CD103Thresh[0]
    #%%
    for idx in list(BDCA2pCD103n.index):
        cellLabelDict[idx] = "pDC"
    for idx in list(BDCA2nCD103p.index):
        cellLabelDict[idx] = "cDC2"
        for idx in list(BDCA2pCD103p.index):
            cellLabelDict[idx] = "BDCA2pCD103p_DCs"
            #%%
            
    gating_df['CD45pHLAIIpCD14nCD16nCD20nCD138n_BDCA2_Int_CD103_Int'] ='none' 
    gating_df.loc[BDCA2pCD103p.index,'CD45pHLAIIpCD14nCD16nCD20nCD138n_BDCA2_Int_CD103_Int'] = "BDCA2pCD103p"
    gating_df.loc[BDCA2nCD103p.index,'CD45pHLAIIpCD14nCD16nCD20nCD138n_BDCA2_Int_CD103_Int'] = "BDCA2nCD103p"
    gating_df.loc[BDCA2pCD103n.index,'CD45pHLAIIpCD14nCD16nCD20nCD138n_BDCA2_Int_CD103_Int'] = "BDCA2pCD103n"
    gating_df.loc[BDCA2nCD103n.index,'CD45pHLAIIpCD14nCD16nCD20nCD138n_BDCA2_Int_CD103_Int'] = "BDCA2nCD103n"
           
            
        #%%
    BDCA1Thresh = otsu_thresholding(df = BDCA2nCD103n,feature="BDCA1_Int-mean",classes=3) ##large spread
    CD11cThresh = otsu_thresholding(df = BDCA2nCD103n,feature="CD11c_Int-mean",classes=3) ##large spread
    
    
    BDCA1pCD11cp,BDCA1nCD11cp,BDCA1pCD11cn,BDCA1nCD11cn = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3-HLAII+CD14-CD16-CD20-CD138-BDCA2-CD103-_BDCA1_CD11c.tif',
                              col1="BDCA1_",
                              col2="CD11c_",
                              plotTitle="BDCA1 v CD11c gated on CD20-CD138-BDCA2-CD4- (cDCs) ",
                              thresh1Lim=BDCA1Thresh[0],
                              thresh2Lim=CD11cThresh[0],
                              df=BDCA2nCD103n,
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                              xLim=[1,300],
                              yLim=[1,300],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict)       
    
    
    
    #%%

    
    gating_df['CD45pHLAIIpCD14nCD16nCD20nCD138nBDCA2n_BDCA1_Int_CD103_Int'] ='none' 
    gating_df.loc[BDCA1pCD11cp.index,'CD45pHLAIIpCD14nCD16nCD20nCD138nBDCA2n_BDCA1_Int_CD103_Int'] = "BDCA1pCD11cp"
    gating_df.loc[BDCA1nCD11cp.index,'CD45pHLAIIpCD14nCD16nCD20nCD138nBDCA2n_BDCA1_Int_CD103_Int'] = "BDCA1nCD11cp"
    gating_df.loc[BDCA1pCD11cn.index,'CD45pHLAIIpCD14nCD16nCD20nCD138nBDCA2n_BDCA1_Int_CD103_Int'] = "BDCA1pCD11cn"
    gating_df.loc[BDCA1nCD11cn.index,'CD45pHLAIIpCD14nCD16nCD20nCD138nBDCA2n_BDCA1_Int_CD103_Int'] = "BDCA1nCD11cn"
    
    
    thresholdDict['CD45pHLAIIpCD14nCD16nCD20nCD138nBDCA2n_BDCA1_Int'] = BDCA1Thresh[0]
    thresholdDict['CD45pHLAIIpCD14nCD16nCD20nCD138nBDCA2n_CD11c_Int'] = CD11cThresh[0]

    
    thresholdDict =pd.DataFrame(thresholdDict,index=[0])
    thresholdDict.to_csv(args.write_dir+'otsu_thresholds.csv')
    
    
    for idx in list(BDCA1pCD11cp.index):
        cellLabelDict[idx] = "cDC1"
    for idx in list(BDCA1pCD11cn.index):
        cellLabelDict[idx] = "cDC1"
    for idx in list(BDCA1nCD11cp.index):
        cellLabelDict[idx] = "cDC1"
    
    #%%
    dfTemp = pd.concat([BDCA1pCD11cp, BDCA1nCD11cp])
    dfTemp = pd.concat([dfTemp, BDCA1pCD11cn])
    
    CD14Thresh = otsu_thresholding(df = dfTemp,feature="CD14_Int-mean",classes=3) ##large spread
    
   
    scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+CD45+CD3-HLAII+CD14-CD16-CD20-CD138-BDCA2-CD103-CD11c+.tif',
                              col1="CD14_",
                              col2="CD11c_",
                              plotTitle="CD14 v CD11c gated on CD20-CD138-BDCA2-CD103-CD11c+ (mo-DCs) ",
                              thresh1Lim=CD14Thresh[0],
                              thresh2Lim=False,
                              df=dfTemp,
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                              xLim=[1,300],
                              yLim=[1,300],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict)       
    ##
    
    #%%

    dfTemp = pd.concat([CD45pHLAIInCD14pCD16p, CD45pHLAIInCD14pCD16n])
    dfTemp = pd.concat([dfTemp, CD45nHLAIInCD14pCD16p])
    CD68Thresh = otsu_thresholding(df = dfTemp,feature="CD68_Int-mean",classes=2) ##large spread
   
    CD68p,CD68n = scatter_plot_save(saveName=args.write_dir+"gating_plots/"+'DAPI+HLAII-_monocytes_CD68.tif',
                              col1="CD14_",
                              col2="CD68_",
                              plotTitle="CD14 v CD68 gated on DAPI+HLAII- monocytes CD68 ",
                              thresh1Lim=False,
                              thresh2Lim=CD68Thresh[0],
                              df=dfTemp,
                              columnsKeep=columnsKeep,
                              scaleColor=white_viridis,
                              xLim=[1,300],
                              yLim=[1,300],
                              text=False,
                              returnExport=True,
                              showPlot=showPlot,
                              timed=True,literalLim=True,alpha=alpha,s=s,quantileThresholding=False,nPoints=nPoints,hue='disease_cohort',hueOrder=hueOrder,palette=hueDict)       
    ##
    #%%
    for idx in list(CD68p.index):
        cellLabelDict[idx] = "Mo-Macrophage"
    
    #%%
    
    gating_df['CD45pHLAIInCD14pCD16p_CD68_Int'] ='none' 
    gating_df.loc[CD68p.index,'CD45pHLAIInCD14pCD16p_CD68_Int'] = "CD68p"
    gating_df.loc[CD68n.index,'CD45pHLAIInCD14pCD16p_CD68_Int'] = "CD68n"
    
    
    #%%
    gating_df.to_csv(args.write_dir+'cell_gating_DT_results.csv',index=False)
    
    
    
    #%%
    df['class_label'] = 'general_unknown'
    colIdx = df.shape[1] -1
    for key in sorted(cellLabelDict.keys()):
        print(f"Key:{key}",cellLabelDict[key])
        df.iloc[key,colIdx] = cellLabelDict[key]
        
        #%%
    
    
    cellCounts = df['class_label'].value_counts()
    
    cellCounts.to_csv(args.write_dir+'per_cell_type_total_counts_all_cohorts.csv',index=True)
   
    
    cellProp = df['class_label'].value_counts()/df['class_label'].value_counts().sum()
    cellProp.to_csv(args.write_dir+'per_cell_type_proportion_all_cohorts.csv',index=True)
    
    for cohort in np.unique(dfInfo['disease_cohort']):
        cellCounts = df[dfInfo['disease_cohort']==cohort]['class_label'].value_counts()
        cellCounts.to_csv(args.write_dir+f'per_cell_type_total_counts_{cohort}.csv',index=True)
        cellProp = df[dfInfo['disease_cohort']==cohort]['class_label'].value_counts()/df[dfInfo['disease_cohort']==cohort]['class_label'].value_counts().sum()
        cellProp.to_csv(args.write_dir+f'per_cell_type_proportion_{cohort}.csv',index=True)

    df.to_csv(args.write_dir+'cell_expression_classified_manual_gating.csv',index=False)

 

   #%%
    

if __name__=='__main__':
    main()
end_time = time.perf_counter()
print(f"Time script took to run: {end_time-start_time} seconds ({(end_time-start_time)/60} minutes) .")











