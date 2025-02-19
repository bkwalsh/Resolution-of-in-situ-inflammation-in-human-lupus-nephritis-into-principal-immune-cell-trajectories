###the packages we use
import time
start_time = time.perf_counter()
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
      

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
def DBSCAN_spatial_segmentation(epsilon,cells_to_include,dfInfo,writeDir,df):        
    
   
    if cells_to_include == 'All':

        idxList = list(dfInfo.index)
    else:
        idxList = []
        for cellClass in cells_to_include:
            idxList.append(df[df['class_label']==cellClass].index)

        idxList = [item for sublist in idxList for item in sublist]
        
        
    dfInfo = dfInfo.loc[idxList]
    
    if dfInfo.shape[0]>0:
        dfToExport = pd.DataFrame(data = np.repeat(np.nan,dfInfo.shape[0]),columns=['DBSCAN_label'],index=list(dfInfo.index))

        clustering = DBSCAN(eps=epsilon, min_samples=2).fit(dfInfo[['GlobalMaskCentroidRow', 'GlobalMaskCentroidCol']])

        for lab,idxTemp in zip(list(clustering.labels_),list(dfInfo.index)):

            dfToExport.loc[idxTemp] = lab

        dfToExport.to_csv(writeDir,index=True)
    print("____"*10)

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
    
    
    
    parser.add_argument('-co',
            '--cohort',
            type=str,
            # default='cell_patches_1um',
            help=''
            )
    parser.add_argument('-px',
            '--pixels',
            type=str,
            # default='cell_patches_1um',
            help=''
            )
        
    args,unparsed = parser.parse_known_args()




    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)
        
    if not os.path.exists(args.write_dir+'plots/'):
        os.mkdir(args.write_dir+'plots/')
    if not os.path.exists(args.write_dir+'plots/'+args.pixels+'/'):
        os.mkdir(args.write_dir+'plots/'+args.pixels+'/')

    
    df = convert_mixed_dtype_to_string(pd.DataFrame(pd.read_csv(args.cell_expression_dir,index_col=False)))
    
    
    dfInfo = convert_mixed_dtype_to_string(pd.DataFrame(pd.read_csv(args.cell_info_dir,index_col=False))) #['Sample', 'Area', 'StainBatch', 'ImagingBatch', 'CompName', 'TileNum', 'TileCellID', 'GlobalTileRow', 'GlobalTileCol', 'GlobalPatchRow', 'GlobalPatchCol', 'PatchRows', 'PatchCols', 'GlobalMaskRow', 'GlobalMaskCol', 'GlobalMaskCentroidRow', 'GlobalMaskCentroidCol', 'disease_cohort']
    
    
    df = df[dfInfo['disease_cohort']==args.cohort]
    dfInfo = dfInfo[dfInfo['disease_cohort']==args.cohort]
    
    dfToExport = pd.DataFrame(data = np.repeat(np.nan,dfInfo.shape[0]),columns=['DBSCAN_label'],index=list(dfInfo.index))

    #%%
    
    for comp in np.unique(dfInfo['CompName']):
        
        compName = comp.split('_')

        if not os.path.exists(args.write_dir+compName[0]):
            os.mkdir(args.write_dir+compName[0])

        if not os.path.exists(args.write_dir+compName[0]+'/'+compName[1]+'/'):
            os.mkdir(args.write_dir+compName[0]+'/'+compName[1]+'/')
           
        epsilon=float(args.pixels)
        tempPath = args.write_dir+compName[0]+'/'+compName[1]+'/'
        DBSCAN_spatial_segmentation(epsilon=epsilon,cells_to_include='All',dfInfo=dfInfo[dfInfo['CompName']==comp],writeDir=f'{tempPath}DBSCAN_all.csv',df=df[dfInfo['CompName']==comp])
        
       #all immune cells
        DBSCAN_spatial_segmentation(epsilon=epsilon,cells_to_include=['Neutrophils', 'CD3+_Tcell', 
                                                                      'CD4+_Tcell',
        'NK_Tcell', 'CD14+CD163+_macrophages', 'CD14+_macrophages',
        'CD8+_Tcell', 'CD8+Foxp3+_Tcell', 'TCRgd_Tcell',
        'HLAII+_Monocytes', 'CD4+PD1+_Tcell', 'CD8+PD1+_Tcell',
        'CD4+Foxp3+_Tcell', 'Bcell', 'CD14+MERTK+_macrophages','CD14+CD163+MERTK+_macrophages',
        'CD4+CD8+_Tcell', 'Mo-Macrophage', 'plasma_cells',
        'CD16+_macrophages', 'Plasmablasts', 'HLAII-_Monocytes',
        'CD4+ICOS+PD1+_Tcell', 'CD8+ICOS+PD1+_Tcell', 'cDC1',
        'CD4+ICOS+_Tcell', 'NK_cells', 'CD8+ICOS+_Tcell', 'pDC', 'cDC2'],dfInfo=dfInfo[dfInfo['CompName']==comp],writeDir=f'{tempPath}DBSCAN_all_immune.csv',df=df[dfInfo['CompName']==comp])

        ##T-cell
        DBSCAN_spatial_segmentation(epsilon=epsilon,cells_to_include=['CD3+_Tcell', 'CD4+_Tcell',
        'NK_Tcell',
        'CD8+_Tcell', 'CD8+Foxp3+_Tcell', 'TCRgd_Tcell',
         'CD4+PD1+_Tcell', 'CD8+PD1+_Tcell',
        'CD4+Foxp3+_Tcell', 
        'CD4+CD8+_Tcell',
        'CD4+ICOS+PD1+_Tcell', 'CD8+ICOS+PD1+_Tcell', 
        'CD4+ICOS+_Tcell',  'CD8+ICOS+_Tcell'],
                                    dfInfo=dfInfo[dfInfo['CompName']==comp],writeDir=f'{tempPath}DBSCAN_Tcells.csv',df=df[dfInfo['CompName']==comp])
        ##B-cell
        DBSCAN_spatial_segmentation(epsilon=epsilon,cells_to_include=['Bcell','Plasmablasts','plasma_cells'],dfInfo=dfInfo[dfInfo['CompName']==comp],writeDir=f'{tempPath}DBSCAN_Bcells.csv',df=df[dfInfo['CompName']==comp])
        ##Myeloid Cells
        DBSCAN_spatial_segmentation(epsilon=epsilon,cells_to_include=['Neutrophils','CD14+CD163+_macrophages', 'CD14+_macrophages',
        'HLAII+_Monocytes','CD14+MERTK+_macrophages','CD14+CD163+MERTK+_macrophages'
         'Mo-Macrophage', 
        'CD16+_macrophages',  'HLAII-_Monocytes',
        'cDC1',
         'NK_cells',  'pDC', 'cDC2'],dfInfo=dfInfo[dfInfo['CompName']==comp],writeDir=f'{tempPath}DBSCAN_Myeloidcells.csv',df=df[dfInfo['CompName']==comp])
        ##Adaptive
        DBSCAN_spatial_segmentation(epsilon=epsilon,cells_to_include=['CD3+_Tcell', 'CD4+_Tcell',
        'CD8+_Tcell', 'CD8+Foxp3+_Tcell', 'CD4+PD1+_Tcell', 'CD8+PD1+_Tcell',
        'CD4+Foxp3+_Tcell', 'Bcell', 
        'CD4+CD8+_Tcell', 'plasma_cells',
       'Plasmablasts',
        'CD4+ICOS+PD1+_Tcell', 'CD8+ICOS+PD1+_Tcell', 
        'CD4+ICOS+_Tcell',  'CD8+ICOS+_Tcell'],
                                    dfInfo=dfInfo[dfInfo['CompName']==comp],writeDir=f'{tempPath}DBSCAN_AdaptiveImmunecells.csv',df=df[dfInfo['CompName']==comp])
        ##innate immune
        DBSCAN_spatial_segmentation(epsilon=epsilon,cells_to_include=['Neutrophils', 'NK_Tcell', 'CD14+CD163+_macrophages', 'CD14+_macrophages',
         'TCRgd_Tcell',
        'HLAII+_Monocytes', 'CD14+MERTK+_macrophages',
        'Mo-Macrophage','CD14+CD163+MERTK+_macrophages'
        'CD16+_macrophages', 'HLAII-_Monocytes',
         'cDC1',
         'NK_cells', 'pDC', 'cDC2'],
                                    dfInfo=dfInfo[dfInfo['CompName']==comp],writeDir=f'{tempPath}DBSCAN_InnateImmunecells.csv',df=df[dfInfo['CompName']==comp])
        ##cytotoxic?
        DBSCAN_spatial_segmentation(epsilon=epsilon,cells_to_include=['NK_Tcell', 'CD8+_Tcell', 'TCRgd_Tcell',
         'NK_cells'],
                                    dfInfo=dfInfo[dfInfo['CompName']==comp],writeDir=f'{tempPath}DBSCAN_Cytotoxiccells.csv',df=df[dfInfo['CompName']==comp])
        ##APCs
        DBSCAN_spatial_segmentation(epsilon=epsilon,cells_to_include=['pDC','cDC1','cDC2','Bcell'],dfInfo=dfInfo[dfInfo['CompName']==comp],writeDir=f'{tempPath}DBSCAN_APCcells.csv',df=df[dfInfo['CompName']==comp])
        ##dendritic
        DBSCAN_spatial_segmentation(epsilon=epsilon,cells_to_include=['pDC','cDC1','cDC2'],dfInfo=dfInfo[dfInfo['CompName']==comp],writeDir=f'{tempPath}DBSCAN_Dendriticcells.csv',df=df[dfInfo['CompName']==comp])
        
       
        ###T-mertk hypothesis cluster
        DBSCAN_spatial_segmentation(epsilon=epsilon,cells_to_include=['CD3+_Tcell', 'CD4+_Tcell',
        'NK_Tcell',
        'CD8+_Tcell', 'CD8+Foxp3+_Tcell', 'TCRgd_Tcell',
         'CD4+PD1+_Tcell', 'CD8+PD1+_Tcell',
        'CD4+Foxp3+_Tcell', 
        'CD4+CD8+_Tcell',
        'CD4+ICOS+PD1+_Tcell', 'CD8+ICOS+PD1+_Tcell', 
        'CD4+ICOS+_Tcell',  'CD8+ICOS+_Tcell','CD14+CD163+MERTK+_macrophages','CD14+MERTK+_macrophages'],
                                    dfInfo=dfInfo[dfInfo['CompName']==comp],writeDir=f'{tempPath}DBSCAN_Tcells_MerTk.csv',df=df[dfInfo['CompName']==comp])
        
      
        print("#########")

#%%
if __name__=='__main__':
    main()

end_time = time.perf_counter()
print(f"Time script took to run: {end_time-start_time} seconds ({(end_time-start_time)/60} minutes) .")











