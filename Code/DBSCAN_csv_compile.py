###the packages we use
import time
start_time = time.perf_counter()
import os
import argparse
import pandas as pd
import numpy as np
import collections



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
def DBSCAN_cohort_csv_compile(DBSCANDir,infoDf,csvName,df,writeDir):
    
   classLabs =  np.unique(df['class_label'])
   exportDict = collections.defaultdict(list) 
   for cohort in np.unique(infoDf['disease_cohort']):


       infoDfCohort = infoDf[infoDf['disease_cohort']==cohort]
       for compName in np.unique(infoDfCohort['CompName']):

           tempPath = DBSCANDir+cohort+f'/{compName.split("_")[0]}/{compName.split("_")[1]}/'
           dfDBSCAN = pd.read_csv(tempPath+csvName,index_col=0)
           infoDfComp = infoDfCohort[infoDfCohort['CompName']==compName]
          
           for dLab in np.unique(dfDBSCAN['DBSCAN_label']):

               
               
               tileNums = infoDfComp.loc[dfDBSCAN[dfDBSCAN['DBSCAN_label']==dLab].index]['TileNum']
               TileCells =  infoDfComp.loc[dfDBSCAN[dfDBSCAN['DBSCAN_label']==dLab].index]['TileCellID']

               

               exportDict['disease_cohort'].append(cohort)
               exportDict['CompName'].append(compName)
               exportDict['AccessionNumber'].append(infoDfComp['AccessionNumber'].values[0])
               exportDict['TissueType'].append(infoDfComp['TissueType'].values[0])
               exportDict['TileNum'].append(tileNums)
               exportDict['TileCellID'].append(TileCells)
               exportDict['idxs'].append(dfDBSCAN[dfDBSCAN['DBSCAN_label']==dLab].index)
               
               exportDict['DBSCAN_label'].append(dLab)
               

               
               sumTotal = np.sum(df.loc[dfDBSCAN[dfDBSCAN['DBSCAN_label']==dLab].index]['class_label'].value_counts())
               for cLab in classLabs:
                   
                   cCount = np.sum(df.loc[dfDBSCAN[dfDBSCAN['DBSCAN_label']==dLab].index]['class_label'] ==cLab)

                   exportDict[cLab+'_count'].append(cCount)
                   exportDict[cLab+'_proportion'].append(cCount/sumTotal)

   exportDict = pd.DataFrame(exportDict)  
   exportDict.to_csv(writeDir+csvName.replace('.csv','_combined.csv'),index=True)
   return(exportDict)
#%%        
def main():
#%%
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
            '--DBSCAN_dir',
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

    parser.add_argument('-cd',
            '--cell_expression_dir',
            type=str,
            # default='cell_patches_1um',
            help=''
            )


    args,unparsed = parser.parse_known_args()




#%%


    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)
        
    if not os.path.exists(args.write_dir+'plots/'):
        os.mkdir(args.write_dir+'plots/')

    df = convert_mixed_dtype_to_string(pd.DataFrame(pd.read_csv(args.cell_expression_dir,index_col=False)))

    infoDf = convert_mixed_dtype_to_string(pd.DataFrame(pd.read_csv(args.cell_info_dir,index_col=False))) #['Sample', 'Area', 'StainBatch', 'ImagingBatch', 'CompName', 'TileNum', 'TileCellID', 'GlobalTileRow', 'GlobalTileCol', 'GlobalPatchRow', 'GlobalPatchCol', 'PatchRows', 'PatchCols', 'GlobalMaskRow', 'GlobalMaskCol', 'GlobalMaskCentroidRow', 'GlobalMaskCentroidCol', 'disease_cohort']
    
    infoDf = infoDf.loc[df.index]
    infoDf["TissueType"] = df["TissueType"] 
    
    
    #%%
    #compiles all the DBSCAN aggregates into a single csv for cluster estimation
    DBSCANDf= DBSCAN_cohort_csv_compile(args.DBSCAN_dir,infoDf,'DBSCAN_all_immune.csv',df,args.write_dir)
    #%%
 
    

if __name__=='__main__':
    main()

end_time = time.perf_counter()
print(f"Time script took to run: {end_time-start_time} seconds ({(end_time-start_time)/60} minutes) .")











