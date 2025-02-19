###the packages we use
import time
start_time = time.perf_counter()
import os
import argparse
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import collections
from skimage import io
import seaborn as sns
import scipy.stats as stats
import matplotlib.colors as mcolors
from PIL import ImageColor
from statsmodels.stats.multitest import multipletests

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

def convert_pvalue_to_asterisks(pvalue):
    print('pvalue',pvalue)
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return " "

def CompName_area_dict_create(tissueMaskDir,df):
    grpArea =  {}
    for comp in np.unique(df['CompName']):
    
        print("comp",comp)
        files = [x for x in os.listdir(tissueMaskDir) if comp in x]
        if len(files) >0:
            file = files[0]
            print("file",file)

            img = io.imread(tissueMaskDir+file)
            grpArea[comp] = np.sum(img)/1000000 ##million pixels
        else:
            # print(asdasdasda)
            print("FILENOT_Found!"*10)
        print("*******"*5)
    print("grp1AreaDict",grpArea) 
    
    return(grpArea)


    


def cell_count_area_normalized(grpArea,df,cellClass):
    
    ##lets make a dictionary with the area by composite
    print("grp1AreaDict",grpArea) 
    
    
    exportDict = {}
    for comp in np.unique(df['CompName']):
        print("comp ",comp)
        print("Area: ",grpArea[comp])
        print("Current Cell: ",cellClass)
        dfComp = df[df['CompName']==comp]
        print("Cell Classes Present:", np.unique(dfComp['class_label']))
        print("dfComp.shape ",dfComp.shape)
        dfComp = dfComp[dfComp['class_label'] == cellClass]
        print("dfComp.shape post cell filter: ",dfComp.shape[0])
        
        if dfComp.shape[0] == 0:
            print("Normalized count: 0")
            exportDict[comp] = 0
        else:
            
            if comp in list(grpArea.keys()): ##some CompName areas do not have all tissue mask
            
                print("Normalized count: ",dfComp.shape[0]/grpArea[comp]) ##normalize by the area
                exportDict[comp] = dfComp.shape[0]/grpArea[comp]
            # exportList.append(dfComp.shape[0]/grpArea[comp])
        print("_"*20)
    
    
    exportDict = pd.DataFrame.from_dict(exportDict,orient='index')
    print(exportDict)
    print('df.columns',df.columns)
    print('df.shape',df.shape)
    exportList = []
    
    print(df['AccessionNumber'])
    print(df['AccessionNumber'].astype(str).value_counts())
    for acc in np.unique(df['AccessionNumber'].astype(str)):
        # print(exportDict)
        comps = np.unique(df[df['AccessionNumber']==acc]['CompName'])
        comps = [x for x in comps if x in list(exportDict.index)]
        
        print(acc,comps)
        
        if len(comps) > 0:
            
        # print('exportDict',exportDict.loc[comps])
            print("mean",np.mean(exportDict.loc[comps],axis=0)[0])
            exportList.append(np.mean(exportDict.loc[comps],axis=0)[0]) ###mean for that accession number
            print("*"*20)
    # 'AccessionNumber'
    print("exportList",exportList)
    # print(qweqweqw)
    
    return(exportList)
    
    
    

def stats_test_per_patient(df,infoDf,colorDict,group1,group2,feat,groupCol,writeDir,grp1Area,grp2Area):
    ## based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    
  
    dfTemp =  pd.concat([df,infoDf],axis = 1,ignore_index=True)
    dfTemp.columns = list(df.columns)+list(infoDf.columns)

    dfTemp = dfTemp.loc[:,~dfTemp.columns.duplicated(keep='first')]
#
    df1 = dfTemp[dfTemp[groupCol]==group1]
    df2 = dfTemp[dfTemp[groupCol]==group2]
    

    
    toExport = collections.defaultdict(list)

    for fe in np.unique(dfTemp[feat]):
        
        grp1 = cell_count_area_normalized(grp1Area,df1,fe)
        grp2 = cell_count_area_normalized(grp2Area,df2,fe)

        U1, p = stats.mannwhitneyu(grp1, grp2)
        
        toExport[feat].append(fe)
        toExport['n_group1'].append(len(grp1))
        toExport['n_group2'].append(len(grp2))
        toExport['mean_group1'].append(np.mean(grp1,axis=0))
        toExport['mean_group2'].append(np.mean(grp2,axis=0))
        toExport['MannU_p_value'].append(p)
        toExport['log2FC'].append(np.log2(np.mean(grp2,axis=0)/np.mean(grp1,axis=0)))
        
        ks, p = stats.kstest(grp1, grp2)
        toExport['KS_p_value'].append(p)

        
    toExport = pd.DataFrame(toExport)
    
    _ , pVal ,_ ,_ = multipletests(list(toExport['MannU_p_value']), method ='fdr_bh')
    toExport['MannU_p_value_corrected'] = pVal
    
    _ , pVal ,_ ,_ = multipletests(list(toExport['KS_p_value']), method ='fdr_bh')
    toExport['KS_p_value_corrected'] = pVal
    toExport.to_csv(writeDir,index=False)
    return(toExport)
        

    

def plot_cell_label_by(feature,df,infoDf,colorDict,saveDir,showPlot,timed,grp1Area,grp2Area,grp1AreaCohort,grp2AreaCohort):
    print("feature",feature)
    
    if not os.path.exists(saveDir):
            os.mkdir(saveDir)
    
    
    cellIdxes=np.unique(df[['class_label']])
    for feat in sorted(np.unique(infoDf[feature])):
        tempDf = df[infoDf[feature]==feat]
        tempInfoDf =  infoDf[infoDf[feature]==feat]
        
        areaList = []
        for compTempArea in np.unique(tempInfoDf['CompName']):
            if np.unique(tempInfoDf[tempInfoDf['CompName']==compTempArea]["disease_cohort"])[0] == grp1AreaCohort:
                areaList.append(grp1Area[compTempArea])
            if np.unique(tempInfoDf[tempInfoDf['CompName']==compTempArea]["disease_cohort"])[0] == grp2AreaCohort:

                areaList.append(grp2Area[compTempArea])

        
        totalArea = np.sum(areaList)

        
        
        areaList = []
        keepForThis = []
        

        cellCount = tempDf['class_label'].value_counts()/totalArea ###normalized by area captured
        
        color_list =[]    
        for lab in cellCount.index:

            cellColor = mcolors.CSS4_COLORS[colorDict[lab]]

            cellRGB = ImageColor.getcolor(cellColor, "RGB")
            color_list.append(cellRGB)
        

        newDict = {}
        for key in colorDict.keys(): #should have all our cell classes, we do this in case the sample doesnt have the cell class nn present
            if key in cellCount.keys():
                newDict[key] = cellCount[key]
            
            else:
                newDict[key] = 0

        counterDict = {key: value for key, value in sorted(newDict.items())} ##sort just in case

        
        cellCount = pd.Series(data = list(counterDict.values()),index=list(counterDict.keys()))
        
        cellCount.to_csv(saveDir+"class_frequency/"+str(feat)+'.csv',index=True)
        
        plt.figure(figsize=(10,8))
        sns.barplot(x = cellCount.index, y = cellCount.values, alpha=0.8,palette=colorDict)
        plt.title(f'Class frequency in {feat}', fontsize=16,fontweight='extra bold')
        plt.ylabel('Number of Occurrences',fontsize=14,fontweight='extra bold')
        plt.xlabel('Cell class', fontsize=14,fontweight='extra bold')
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=90)
        plt.tight_layout()
        
        if not os.path.exists(saveDir+"class_frequency/"):
            os.mkdir(saveDir+"class_frequency/")
        
        if showPlot:
            if timed:
                plt.show(block=False)
                plt.pause(2)
                plt.close() 
            else: 
                plt.show()
                
                
        plt.savefig(saveDir+"class_frequency/"+str(feat)+'.tiff',dpi=75)
        plt.close()
      

        #https://stackoverflow.com/questions/68154123/matplotlib-grouped-bar-chart-with-individual-data-points


        
        cellCount = pd.Series(data = list(cellCount.values/cellCount.values.sum()),index=list(counterDict.keys())) ##make into proportion
        
        
        plt.figure(figsize=(10,8))
        sns.barplot(x = cellCount.index, y = cellCount.values, alpha=0.8,palette=colorDict)
        plt.title(f'Class proportion in {feat}', fontsize=16,fontweight='extra bold')
        plt.ylabel('Proportion', fontsize=14,fontweight='extra bold')
        plt.xlabel('Cell class', fontsize=14,fontweight='extra bold')
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=90)
        plt.tight_layout()
        
        if not os.path.exists(saveDir+"class_proportion/"):
            os.mkdir(saveDir+"class_proportion/")
        
        
        if showPlot:
            if timed:
                plt.show(block=False)
                plt.pause(2)
                plt.close() 
            else: 
                plt.show()
        plt.savefig(saveDir+"class_proportion/"+str(feat)+'.tiff',dpi=75)
        plt.close()
        
        cellCount.to_csv(saveDir+"class_proportion/"+str(feat)+'.csv',index=True)

        
        
def flourescence_expression_by_cell_class(df,saveDir,showPlot,timed):     
    
    if not os.path.exists(saveDir+"class_flourescence_expression/"):
            os.mkdir(saveDir+"class_flourescence_expression/")
    saveDir=saveDir+"class_flourescence_expression/"
    
    
    cols = [ 'CD86_Int-mean', 'Claudin1_Int-mean', 'RORgt_Int-mean', 'HLAII_Int-mean', 'COLIII_Int-mean', 'CD14_Int-mean', 'Foxp3_Int-mean', 'CD56_Int-mean', 'GZMB_Int-mean', 'IL10_Int-mean', 'CD69_Int-mean', 'CD163_Int-mean', 'CD21_Int-mean', 'MERTK_Int-mean', 'CD11c_Int-mean', 'SLAMF7_Int-mean', 'CD27_Int-mean', 'CD10_Int-mean', 'IFNG_Int-mean', 'CD43_Int-mean', 'CD31_Int-mean', 'TCRD_Int-mean', 'ICOS_Int-mean', 'iNOS_Int-mean', 'CD68_Int-mean', 'BDCA2_Int-mean', 'GZMK_Int-mean', 'Ki67_Int-mean', 'BDCA1_Int-mean', 'CD3_Int-mean', 'GZMA_Int-mean', 'MUC1_Int-mean', 'CD16_Int-mean', 'MXA_Int-mean', 'PD1_Int-mean', 'CD20_Int-mean', 'CD8_Int-mean', 'Tbet_Int-mean', 'CD103_Int-mean', 'CD4_Int-mean', 'CD45_Int-mean', 'CD138_Int-mean', 'mTOC_Int-mean']
    
    cols = sorted(cols)
    
    
    colors = list(mcolors.CSS4_COLORS.keys())

    toRemove = ['navajowhite','slategray','ghostwhite','lightyellow','ivory','linen','cornsilk','gray','moccasin','papayawhip','lavenderblush','lightgoldenrodyellow','silver',
               'whitesmoke','mintcream','honeydew','dimgrey','floralwhite','lightslategray','darkgrey' ,'blanchedalmond','gainsboro','oldlace','darkolivegreen',
               'dimgray','seashell', 'mistyrose','aliceblue','azure']
    colors = [x for x in colors if x not in toRemove]
    
    for cellLab in np.unique(df['class_label']):

        classDf = df[df['class_label']==cellLab]

        expressionMeans = classDf[cols].mean()
        expressionSD = classDf[cols].std()

        plt.figure(figsize=(10,8))
        plt.bar(expressionMeans.index, expressionMeans.values,color=colors,yerr=expressionSD,capsize=5)

        # set the plot title and axis labels
        plt.title(f'Mean values of each column {cellLab}', fontsize=16,fontweight='extra bold')
        plt.xlabel('Columns', fontsize=14,fontweight='extra bold')
        plt.ylabel('Mean value', fontsize=14,fontweight='extra bold')
        plt.xticks(expressionMeans.index, rotation=90, fontsize=14,fontweight='extra bold')
        plt.tight_layout()
        plt.ylim(0,100)
        plt.savefig(saveDir+cellLab+'.tiff',dpi=100)
        # show the plot
        if showPlot:
            if timed:
                plt.show(block=False)
                plt.pause(10)
                plt.close() 
            else: 
                plt.show()
        
 
        
    #%%
        
def main():
    
    #%%

    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
            '--cell_expression_dir',
            type=str,
            # default='cell_patches_1um',
            help=''
            )

    parser.add_argument('-w',
            '--write_dir',
            type=str,
            default='channel_expression_csvs',
            help=''
            )
    parser.add_argument('-i',
            '--cell_info_dir',
            type=str,
            default='channel_expression_csvs',
            help=''
            )
    parser.add_argument('-s',
            '--WSI_dir',
            type=str,
            default='',
            help=''
            )
    
    parser.add_argument('-l',
            '--color_dict',
            type=str,
            # default='cell_patches_1um',
            help=''
            )
    parser.add_argument('-tm',
            '--area_dict',
            type=str,
            # default='cell_patches_1um',
            help=''
            )
    

    args,unparsed = parser.parse_known_args()


    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)
    if not os.path.exists(args.write_dir+'density_analysis/'):
        os.mkdir(args.write_dir+'density_analysis/')
    

    df = convert_mixed_dtype_to_string(pd.DataFrame(pd.read_csv(args.cell_expression_dir,index_col=False)))
    infoDf = convert_mixed_dtype_to_string(pd.DataFrame(pd.read_csv(args.cell_info_dir,index_col=False)))
    #%%
    infoDf = infoDf.loc[df.index] ##so they necessarily match

    
    with open(args.color_dict,'rb') as f:
        colorDict = pickle.load(f)

    classLabelsList = np.unique(df['class_label'])
    
    with open(args.area_dict+'NKAreaDict.pkl','rb') as f:
        NKAreaDict = pickle.load(f)
    with open(args.area_dict+'LNAreaDict.pkl','rb') as f:
        LNAreaDict = pickle.load(f)
    with open(args.area_dict+'ARAreaDict.pkl','rb') as f:
        ARAreaDict = pickle.load(f)


    
    
    #%%
    
   
    
    NK_LN = stats_test_per_patient(df,infoDf,colorDict,'Normal_Kidney',"Lupus_Nephritis",'class_label',"disease_cohort",args.write_dir+'NK_LN.csv',
                                   NKAreaDict,LNAreaDict)
    
    #%%
    NK_AR = stats_test_per_patient(df,infoDf,colorDict,'Normal_Kidney','Renal_Allograft','class_label',"disease_cohort",args.write_dir+'NK_AR.csv',
                                   NKAreaDict,ARAreaDict)
    
    
    #%%
    NK_TCMR = stats_test_per_patient(df,infoDf,colorDict,'NK',"TCMR",'class_label',"TissueType",args.write_dir+'NK_TCMR.csv',
                                   NKAreaDict,ARAreaDict)
    #%%
    NK_MR = stats_test_per_patient(df,infoDf,colorDict,'NK',"MR",'class_label',"TissueType",args.write_dir+'NK_MR.csv',
                                   NKAreaDict,ARAreaDict)
    #['LuN' 'MR' 'NK' 'TCMR']
    #%%
    LN_AR = stats_test_per_patient(df,infoDf,colorDict,"Lupus_Nephritis",'Renal_Allograft','class_label',"disease_cohort",args.write_dir+'LN_AR.csv',
                                   LNAreaDict,ARAreaDict)
    
    #%%
    
    LN_MR = stats_test_per_patient(df,infoDf,colorDict,"LuN",'MR','class_label',"TissueType",args.write_dir+'LN_MR.csv',
                                  LNAreaDict,ARAreaDict)
    LN_TCMR = stats_test_per_patient(df,infoDf,colorDict,"LuN",'TCMR','class_label',"TissueType",args.write_dir+'LN_TCMR.csv',
                                   LNAreaDict,ARAreaDict)
    MR_TCMR = stats_test_per_patient(df,infoDf,colorDict,"MR",'TCMR','class_label',"TissueType",args.write_dir+'MR_TCMR.csv',
                                   ARAreaDict,ARAreaDict)
    #%%
    
    
if __name__=='__main__':
    main()

end_time = time.perf_counter()
print(f"Time script took to run: {end_time-start_time} seconds ({(end_time-start_time)/60} minutes) .")











