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
import seaborn as sns
import scipy.stats as stats

# import umap
#function
import matplotlib.colors as mcolors
from statsmodels.stats.multitest import multipletests


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

def cohort_heatmap_plot(classLabelsList,colNames,col1,col2,col3,col4,pVal,writeDir):
    heatmapSignificance = pd.DataFrame(index=classLabelsList,columns=colNames)
    heatmapLog2FC = pd.DataFrame(index=classLabelsList,columns=colNames)
    
    
    
    heatmapLog2FC[colNames[0]] = col1['log2FC'].values
    heatmapLog2FC[colNames[1]] = col2['log2FC'].values
    heatmapLog2FC[colNames[2]] = col3['log2FC'].values
    heatmapLog2FC[colNames[3]] = col4['log2FC'].values
    
    heatmapSignificance[colNames[0]] = [convert_pvalue_to_asterisks(x) for x in col1[pVal].values]
    heatmapSignificance[colNames[1]] = [convert_pvalue_to_asterisks(x) for x in col2[pVal].values]
    heatmapSignificance[colNames[2]] = [convert_pvalue_to_asterisks(x) for x in col3[pVal].values]
    heatmapSignificance[colNames[3]] = [convert_pvalue_to_asterisks(x) for x in col4[pVal].values]
    

    cmap = sns.color_palette("RdBu", as_cmap=True)
    linewidths =1
    fig, ax = plt.subplots(figsize=(10, 10))
    ax=sns.heatmap(heatmapLog2FC,cmap=cmap.reversed(),annot=heatmapSignificance, fmt='', annot_kws={"size": 14, "weight": "extra bold"},vmin=-3, vmax=3, center=0, linewidths=linewidths,linecolor='black')
    
    ax.set_aspect('equal')  # Set the aspect of the axis to be "equal"
   
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(linewidths)  # Set the thickness of the border
    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  # Rotate x-axis labels by 45 degrees

    

    plt.yticks(fontsize=14, weight='extra bold')  # You can adjust the fontsize as needed
    plt.title('Heatmap of log2FC Cell Count',size=14, weight="extra bold")
    plt.xlabel('Biopsy Cohorts Compared',size=14, weight="extra bold")
    plt.ylabel('Cell Types',size=14, weight="extra bold")
    plt.tight_layout()
    plt.savefig(writeDir,dpi=300)
    plt.show()
    plt.close()
    
def cohort_heatmap_plot_new(classLabelsList,colNames,cols,pVal,writeDir,heatmapColor):
    heatmapSignificance = pd.DataFrame(index=classLabelsList,columns=colNames)
    heatmapLog2FC = pd.DataFrame(index=classLabelsList,columns=colNames)
    
    
    for i,col in enumerate(cols):
    
        heatmapLog2FC[colNames[i]] = col['log2FC'].values

    
        heatmapSignificance[colNames[i]] = [convert_pvalue_to_asterisks(x) for x in col[pVal].values]

    cmap = sns.color_palette(heatmapColor, as_cmap=True)
    linewidths =1
    fig, ax = plt.subplots(figsize=(10, 10))
    ax=sns.heatmap(heatmapLog2FC,cmap=cmap.reversed(),annot=heatmapSignificance, fmt='', annot_kws={"size": 12, "weight": "extra bold"},vmin=-3, vmax=3, center=0, linewidths=linewidths,linecolor='black')
       
    ax.set_aspect('equal')  # Set the aspect of the axis to be "equal"
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(linewidths)  # Set the thickness of the border
    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  # Rotate x-axis labels by 45 degrees
    
    plt.yticks(fontsize=14, weight='extra bold')  # You can adjust the fontsize as needed
    plt.title('Heatmap of log2FC Cell Count',size=14, weight="extra bold")
    plt.xlabel('Biopsy Cohorts Compared',size=14, weight="extra bold")
    plt.ylabel('Cell Types',size=14, weight="extra bold")
    plt.tight_layout()
    plt.savefig(writeDir,dpi=300)
    plt.show()
    plt.close()


def cell_count_area_normalized(grpArea,df,cellClass):
    
    ##lets make a dictionary with the area by composite
    print("grp1AreaDict",grpArea) 
    
    exportDict = {}
    for comp in np.unique(df['CompName']):
        
        dfComp = df[df['CompName']==comp]
        dfComp = dfComp[dfComp['class_label'] == cellClass]
        
        if dfComp.shape[0] == 0:
            exportDict[comp] = 0
        else:
            
            if comp in list(grpArea.keys()): ##some CompName areas do not have all tissue mask
   
                exportDict[comp] = dfComp.shape[0]/grpArea[comp]
 
        print("_"*20)
    
    
    exportDict = pd.DataFrame.from_dict(exportDict,orient='index')
    exportList = []
    

    for acc in np.unique(df['AccessionNumber'].astype(str)):
        # print(exportDict)
        comps = np.unique(df[df['AccessionNumber']==acc]['CompName'])
        comps = [x for x in comps if x in list(exportDict.index)]
        
        
        if len(comps) > 0:
            

            exportList.append(np.mean(exportDict.loc[comps],axis=0)[0]) ###mean for that accession number

    
    return(exportList)
    
    
    

def stats_test_per_patient(df,infoDf,colorDict,group1,group2,feat,groupCol,writeDir,grp1Area,grp2Area):
    ## based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    
    
    dfTemp =  pd.concat([df,infoDf],axis = 1,ignore_index=True)
    dfTemp.columns = list(df.columns)+list(infoDf.columns)
    

    dfTemp = dfTemp.loc[:,~dfTemp.columns.duplicated(keep='first')]

    df1 = dfTemp[dfTemp[groupCol]==group1]
    df2 = dfTemp[dfTemp[groupCol]==group2]
    
    
    toExport = collections.defaultdict(list)

    for fe in np.unique(dfTemp[feat]):
        
        grp1 = cell_count_area_normalized(grp1Area,df1,fe)
        grp2 = cell_count_area_normalized(grp2Area,df2,fe)

  
        U1, p = stats.mannwhitneyu(grp1, grp2)

        
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


def plot_cell_label_by(feature,df,infoDf,colorDict,saveDir,showPlot,timed,grp1Area,grp2Area,grp3Area,grp1AreaCohort,grp2AreaCohort,grp3AreaCohort):
    
    if not os.path.exists(saveDir):
            os.mkdir(saveDir)
    
    cellIdxes=np.unique(df[['class_label']])
    for feat in sorted(np.unique(infoDf[feature].tolist())):
        tempDf = df[infoDf[feature]==feat]
        tempInfoDf =  infoDf[infoDf[feature]==feat]

        

        areaList = []
        for compTempArea in np.unique(tempInfoDf['CompName']):
            if np.unique(tempInfoDf[tempInfoDf['CompName']==compTempArea]["disease_cohort"])[0] == grp1AreaCohort:
                areaList.append(grp1Area[compTempArea])
            if np.unique(tempInfoDf[tempInfoDf['CompName']==compTempArea]["disease_cohort"])[0] == grp2AreaCohort:
                areaList.append(grp2Area[compTempArea])
            if np.unique(tempInfoDf[tempInfoDf['CompName']==compTempArea]["disease_cohort"])[0] == grp3AreaCohort:
                areaList.append(grp3Area[compTempArea])
        
        totalArea = np.sum(areaList)

        areaList = []
        keepForThis = []
         

        cellCount = tempDf['class_label'].value_counts()/totalArea ###normalized by area captured
        
        color_list =[]    
        for lab in cellCount.index:


            cellColor = colorDict[lab]
            cellRGB = cellColor[:4]
            color_list.append(cellRGB)
        
        newDict = {}
        for key in colorDict.keys(): #should have all our cell classes, we do this in case the sample doesnt have the cell class nn present
            if key in cellCount.keys():
                newDict[key] = cellCount[key]
            
            else:
                newDict[key] = 0

        counterDict = {key: value for key, value in sorted(newDict.items())} ##sort just in case
        
        cellCount = pd.Series(data = list(counterDict.values()),index=list(counterDict.keys()))
        
        if not os.path.exists(saveDir+"class_frequency/"):
            os.mkdir(saveDir+"class_frequency/")
        cellCount.to_csv(saveDir+"class_frequency/"+str(feat)+'.csv',index=True)
        
        plt.figure(figsize=(10,8))
        sns.barplot(x = cellCount.index, y = cellCount.values, alpha=0.8,palette=colorDict)
        plt.title(f'Class frequency in {feat}', fontsize=16,fontweight='extra bold')
        plt.ylabel('Number of Occurrences',fontsize=14,fontweight='extra bold')
        plt.xlabel('Cell class', fontsize=14,fontweight='extra bold')
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=90)
        plt.tight_layout()
         
        
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
        #Should I add the actual dots?
        
        print("Total cells",cellCount.values.sum())
        
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
        ##proportion
        
        
def flourescence_expression_by_cell_class(df,saveDir,showPlot,timed,filename):     
    
    if not os.path.exists(saveDir+"class_flourescence_expression/"):
            os.mkdir(saveDir+"class_flourescence_expression/")
    saveDir=saveDir+"class_flourescence_expression/"
    
    
    cols = [ 'CD86_Int-mean', 'Claudin1_Int-mean', 'RORgt_Int-mean', 'HLAII_Int-mean', 'COLIII_Int-mean', 'CD14_Int-mean', 'Foxp3_Int-mean', 'CD56_Int-mean', 'GZMB_Int-mean', 'CD69_Int-mean', 'CD163_Int-mean', 'CD21_Int-mean', 'MERTK_Int-mean', 'CD11c_Int-mean', 'SLAMF7_Int-mean', 'CD27_Int-mean', 'CD10_Int-mean', 'IFNG_Int-mean', 'CD43_Int-mean', 'CD31_Int-mean', 'TCRD_Int-mean', 'ICOS_Int-mean', 'iNOS_Int-mean', 'CD68_Int-mean', 'BDCA2_Int-mean', 'GZMK_Int-mean', 'Ki67_Int-mean', 'BDCA1_Int-mean', 'CD3_Int-mean', 'GZMA_Int-mean', 'MUC1_Int-mean', 'CD16_Int-mean', 'MXA_Int-mean', 'PD1_Int-mean', 'CD20_Int-mean', 'CD8_Int-mean', 'Tbet_Int-mean', 'CD103_Int-mean', 'CD4_Int-mean', 'CD45_Int-mean', 'CD138_Int-mean', 'mTOC_Int-mean']
    
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
        # Set the thickness of the spines
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(5)  # Set the thickness here

        plt.bar(expressionMeans.index, expressionMeans.values,color=colors,yerr=expressionSD,capsize=5, edgecolor='black', linewidth=1)

        # set the plot title and axis labels
        plt.title(f'Mean values of each column {cellLab}', fontsize=16,fontweight='extra bold')
        plt.xlabel('Columns', fontsize=14,fontweight='extra bold')
        plt.ylabel('Mean value', fontsize=14,fontweight='extra bold')
        plt.xticks(expressionMeans.index, rotation=90, fontsize=14,fontweight='extra bold')
        plt.tight_layout()
        plt.ylim(0,100)
        plt.savefig(saveDir+cellLab+filename,dpi=100)
        # show the plot
        if showPlot:
            if timed:
                plt.show(block=False)
                plt.pause(10)
                plt.close() 
            else: 
                plt.show()

        
    #%%
def reorder_dataframe(df, class_order):
    """Reorder dataframe based on a given class order."""
    # Create a dictionary to map class labels to their order
    class_order_dict = {class_label: index for index, class_label in enumerate(class_order)}
    
    # Sort the dataframe based on the order in class_order_dict
    df['order'] = df['class_label'].map(class_order_dict)
    df = df.sort_values('order').drop(columns='order')
    
    return df        
    
    
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
            '--cohort_stats_dir',
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
    

    df = pd.DataFrame(pd.read_csv(args.cell_expression_dir,index_col=False))
    infoDf = pd.DataFrame(pd.read_csv(args.cell_info_dir,index_col=False))
    
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
    
    for cohort in df['disease_cohort'].unique():
        dfTemp = df.copy()
        dfTemp = df[df['disease_cohort']==cohort]
        counts = dfTemp['class_label'].value_counts()
        cell_class = counts.index
        proportions = dfTemp['class_label'].value_counts(normalize=True)
        percentages = (proportions * 100).round(2)  # Convert to percentage and round to 2 significant digits
   
        
        results = pd.DataFrame({'Class_label':cell_class,'Count': counts,'Percentage':percentages ,'Proportion': proportions})


    NK_LN = pd.DataFrame(pd.read_csv(args.write_dir+'NK_LN.csv',index_col=False))
    NK_AR = pd.DataFrame(pd.read_csv(args.write_dir+'NK_AR.csv',index_col=False))
    NK_TCMR =  pd.DataFrame(pd.read_csv(args.write_dir+'NK_TCMR.csv',index_col=False))
    NK_MR =  pd.DataFrame(pd.read_csv(args.write_dir+'NK_MR.csv',index_col=False))
    
    
    #%%
    class_order_plot=['Bcell','Plasmablasts','plasma_cells','CD4+Foxp3+_Tcell',
                      'CD4+ICOS+PD1+_Tcell','CD4+ICOS+_Tcell','CD4+PD1+_Tcell','CD4+_Tcell','CD4+_intraepithelial_Tcell',
                      'CD8+Foxp3+_Tcell','CD8+ICOS+PD1+_Tcell','CD8+ICOS+_Tcell','CD8+PD1+_Tcell',
                      'CD8+_Tcell','CD8+_intraepithelial_Tcell','CD3+_Tcell','CD4+CD8+_Tcell','TCRgd_Tcell','NK_Tcell','NK_cells','CD14+CD163+MERTK+_macrophages',
                     'CD14+CD163+_macrophages','CD14+MERTK+_macrophages','CD14+_macrophages','CD16+_macrophages','HLAII+_Monocytes',
                     'HLAII-_Monocytes','Mo-Macrophage','Neutrophils','BDCA2pCD103p_DCs','cDC1','cDC2','pDC', 'Distal_collecting_tubules',
                     'Endothelial_cells','Inflamed_tubule','Proximal_tubules','RBCs','general_unknown']
    
    
    NK_LN = reorder_dataframe(NK_LN,  class_order_plot)
    NK_AR = reorder_dataframe(NK_AR,  class_order_plot)
    NK_TCMR = reorder_dataframe(NK_TCMR,  class_order_plot)
    NK_MR = reorder_dataframe(NK_MR,  class_order_plot)
    
    #%%
    LN_AR = pd.DataFrame(pd.read_csv(args.write_dir+'LN_AR.csv',index_col=False))
    LN_MR = pd.DataFrame(pd.read_csv(args.write_dir+'LN_MR.csv',index_col=False))
    LN_TCMR = pd.DataFrame(pd.read_csv(args.write_dir+'LN_TCMR.csv',index_col=False))
    MR_TCMR =  pd.DataFrame(pd.read_csv(args.write_dir+'MR_TCMR.csv',index_col=False))
    
    
    
    #%%
    
    LN_AR = reorder_dataframe(LN_AR,  class_order_plot)
    LN_MR = reorder_dataframe(LN_MR,  class_order_plot)
    LN_TCMR = reorder_dataframe(LN_TCMR,  class_order_plot)
    MR_TCMR = reorder_dataframe(MR_TCMR,  class_order_plot)
    

#%%
    ####comparing the Normal Kidney control to the pathologic cohorts
    
    cohort_heatmap_plot(classLabelsList=class_order_plot,
                        colNames=['NK_LN','NK_AR','NK_TCMR','NK_MR'],
                        col1=NK_LN,
                        col2=NK_AR,
                        col3=NK_TCMR,
                        col4=NK_MR,
                        pVal='MannU_p_value',
                        writeDir=args.write_dir+'control_pathologic_density_uncorrected.tif')
    
  
    #%%
    cohort_heatmap_plot_new(classLabelsList=class_order_plot,
                        colNames=['LN_AR','NK_AR','NK_LN'],
                        cols=[LN_AR,NK_AR,NK_LN],
                        pVal='MannU_p_value',
                        writeDir=args.write_dir+'main_figure_density_uncorrected.png',
                        heatmapColor='RdBu')
    cohort_heatmap_plot_new(classLabelsList=class_order_plot,
                        colNames=['LN_AR','NK_AR','NK_LN'],
                        cols=[LN_AR,NK_AR,NK_LN],
                        pVal='MannU_p_value_corrected',
                        writeDir=args.write_dir+'main_figure_density_corrected.png',
                        heatmapColor='RdBu')
    #%%
    
    
    cohort_heatmap_plot_new(classLabelsList=class_order_plot,
                        colNames= ['MR_TCMR','LN_TCMR','LN_MR','NK_MR','NK_TCMR'],
                        cols= [MR_TCMR,LN_TCMR,LN_MR,NK_MR,NK_TCMR,LN_AR,NK_AR,NK_LN],
                        pVal='MannU_p_value',
                        writeDir=args.write_dir+'supplemental_figure_density_uncorrected.png',
                        heatmapColor='RdBu')
    cohort_heatmap_plot_new(classLabelsList=class_order_plot,
                        colNames= ['MR_TCMR','LN_TCMR','LN_MR','NK_MR','NK_TCMR'],
                        cols= [MR_TCMR,LN_TCMR,LN_MR,NK_MR,NK_TCMR],
                        pVal='MannU_p_value',
                        writeDir=args.write_dir+'supplemental_figure_density_corrected.png',
                        heatmapColor='RdBu')
    
    #
    #%%
    cohort_heatmap_plot_new(classLabelsList=class_order_plot,
                        colNames= ['MR_TCMR','LN_TCMR','LN_MR','LN_AR','NK_MR','NK_TCMR','NK_AR','NK_LN'],
                        cols= [MR_TCMR,LN_TCMR,LN_MR,LN_AR,NK_MR,NK_TCMR,NK_AR,NK_LN],
                        pVal='MannU_p_value',
                        writeDir=args.write_dir+'Medical_imaging_uncorrected.png',
                        heatmapColor='PRGn')
    cohort_heatmap_plot_new(classLabelsList=class_order_plot,
                        colNames= ['MR_TCMR','LN_TCMR','LN_MR','LN_AR','NK_MR','NK_TCMR','NK_AR','NK_LN'],
                        cols= [MR_TCMR,LN_TCMR,LN_MR,LN_AR,NK_MR,NK_TCMR,NK_AR,NK_LN],
                        pVal='MannU_p_value_corrected',
                        writeDir=args.write_dir+'Medical_imaging_corrected.png',
                        heatmapColor='PRGn')
    
    
    #%%
    cohort_heatmap_plot(classLabelsList=class_order_plot,
                        colNames=['NK_LN','NK_AR','NK_TCMR','NK_MR'],
                        col1=NK_LN,
                        col2=NK_AR,
                        col3=NK_TCMR,
                        col4=NK_MR,
                        pVal='MannU_p_value_corrected',
                        writeDir=args.write_dir+'control_pathologic_density_corrected.tif')
    
    #%%
    cohort_heatmap_plot(classLabelsList=class_order_plot,
                        colNames=['LN_AR','LN_MR','LN_TCMR','MR_TCMR'],
                        col1=LN_AR,
                        col2=LN_MR,
                        col3=LN_TCMR,
                        col4=MR_TCMR,
                        pVal='MannU_p_value',
                        writeDir=args.write_dir+'only_pathologic_density_uncorrected.tif')
    
    #%%
    cohort_heatmap_plot(classLabelsList=class_order_plot,
                        colNames=['LN_AR','LN_MR','LN_TCMR','MR_TCMR'],
                        col1=LN_AR,
                        col2=LN_MR,
                        col3=LN_TCMR,
                        col4=MR_TCMR,
                        pVal='MannU_p_value_corrected',
                        writeDir=args.write_dir+'only_pathologic_density_corrected.tif')
    
    #%%
    
    flourescence_expression_by_cell_class(df,args.write_dir,showPlot=True,timed=True,filename='.tiff')
    #%%
    for cohort in np.unique(df['disease_cohort']):
        flourescence_expression_by_cell_class(df[df['disease_cohort']==cohort],args.write_dir,showPlot=True,timed=True,filename=f'_{cohort}.tiff')
    
    #%%
    plot_cell_label_by("CompName",df,infoDf,colorDict,args.write_dir+'CompName/',False,False,LNAreaDict,ARAreaDict,NKAreaDict,"Lupus_Nephritis","Renal_Allograft","Normal_Kidney")
    
    #%%

    plot_cell_label_by("disease_cohort",df,infoDf,colorDict,args.write_dir+'disease_cohort/',False,False,LNAreaDict,ARAreaDict,NKAreaDict,"Lupus_Nephritis","Allograft_Rejection","Normal_Kidney")
    plot_cell_label_by("AccessionNumber",df,infoDf,colorDict,args.write_dir+'AccessionNumber/',False,False,LNAreaDict,ARAreaDict,NKAreaDict,"Lupus_Nephritis","Allograft_Rejection","Normal_Kidney")
    #%%
    plot_cell_label_by("TissueType",df,infoDf,colorDict,args.write_dir+'TissueType/',False,False,LNAreaDict,ARAreaDict,NKAreaDict,"Lupus_Nephritis","Allograft_Rejection","Normal_Kidney")

    #%%
    #Normalize by Area, make sure code stops dropping the 0 counts
    
if __name__=='__main__':
    main()

end_time = time.perf_counter()
print(f"Time script took to run: {end_time-start_time} seconds ({(end_time-start_time)/60} minutes) .")











