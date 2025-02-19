###the packages we use
import time
start_time = time.perf_counter()
import os
import argparse
import pandas as pd

import numpy as np

import collections
from skimage import io
import multiprocessing as mp
import joblib
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
def image_process(img,writeDir,compName,check_exists,combined_df,lab):
    
    img = np.squeeze(img)#[:20000,:20000]
    
    output_filename = writeDir + f'{compName}_{lab}.csv'
    if check_exists and os.path.exists(output_filename):
        print(f'File {output_filename} already exists. Skipping... {img.sum()}')
        
        
        return(compName,img.sum()) 
    
    ind = np.where(img)
    all_pts = [[x,y] for x,y in zip(ind[0],ind[1])]

        # Convert all_pts into a DataFrame
    pts_df = pd.DataFrame(all_pts, columns=['GlobalMaskCentroidRow', 'GlobalMaskCentroidCol'])
    
    # Convert the columns to integer type for proper matching
    pts_df['GlobalMaskCentroidRow'] = pts_df['GlobalMaskCentroidRow'].astype(int)
    pts_df['GlobalMaskCentroidCol'] = pts_df['GlobalMaskCentroidCol'].astype(int)
    
    # Merge with the original DataFrame to find matching rows
    matching_rows = combined_df.merge(pts_df, on=['GlobalMaskCentroidRow', 'GlobalMaskCentroidCol'])

    matching_rows.to_csv(output_filename,index=False)
    return(compName,img.sum()) 
    
    
def stain_feature_extract(filename,structurePath,dfCohort,infoDfCohort,cohort,writeDir,classLabelsList,classIgnore,jobNum,thresholdMask,start_time, check_exists=False):

    props_dict = collections.defaultdict(list)
    compName = filename.replace('.tif','')
    img = io.imread(structurePath+filename)
    
    dfComp = dfCohort[infoDfCohort['CompName']==compName]
    infoDfComp = infoDfCohort[infoDfCohort['CompName']==compName]
    combined_df = pd.concat([dfComp, infoDfComp], axis=1)
    

    area_dict = {}
    for lab in np.unique(img):
        print('lab',lab)
        if lab >0:
            imgTemp =img.copy()

            imgTemp[imgTemp!=lab] = 0
            imgTemp[imgTemp==lab] = 1

            
            compName_,area = image_process(imgTemp,writeDir,compName,check_exists,combined_df,lab)
            area_dict[lab]=area
            end_time = time.perf_counter()
            print(f"{compName}, lab: {lab} {end_time-start_time} seconds ({(end_time-start_time)/60} minutes) .")

    
    
    saveDf = pd.DataFrame(list(area_dict.items()), columns=['Key', 'Value'])
    saveDf.to_csv(writeDir + f'{compName}_areas.csv', index=False)

            
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
            '--structure_dir',
            type=str,
            default='',
            help=''
            )
    parser.add_argument('-tm',
            '--cohort',
            type=str,
            # default='cell_patches_1um',
            help=''
            )
    parser.add_argument('-p',
            '--processors',
            type=int,
            # default='cell_patches_1um',
            help=''
            )

    
    args,unparsed = parser.parse_known_args()
    
    
    
    
    
    
    #%%
    

    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)
    # if not os.path.exists(args.write_dir+args.cohort+'/'):
    #     os.mkdir(args.write_dir+args.cohort+'/')

    df = pd.DataFrame(pd.read_csv(args.cell_expression_dir,index_col=False))
    infoDf = pd.DataFrame(pd.read_csv(args.cell_info_dir,index_col=False))
    
    df = convert_mixed_dtype_to_string(df)
    infoDf =convert_mixed_dtype_to_string(infoDf)
    
    infoDf = infoDf.loc[df.index] ##so they necessarily match

    classLabelsList = np.unique(df['class_label'])

    files = os.listdir(args.structure_dir)

    dfCohort = df[infoDf['disease_cohort']==args.cohort]
    infoDfCohort = infoDf[infoDf['disease_cohort']==args.cohort]
    
    
    #%%
    for i,acc in enumerate(np.unique(list(dfCohort['AccessionNumber']))):
        print(i,'acc ',acc)
        
        dfAcc = dfCohort[infoDfCohort['AccessionNumber']==acc]
        infoDfAcc = infoDfCohort[infoDfCohort['AccessionNumber']==acc]

    #%%

    if args.processors > joblib.cpu_count():
        args.processors = joblib.cpu_count()
    
   # Prepare arguments for each process
    process_args = [(filename, args.structure_dir, dfCohort, infoDfCohort, args.cohort, args.write_dir, classLabelsList, ['Distal_collecting_tubules', 'Inflamed_tubule', 'Proximal_tubules', 'other'], jobNum, 1,start_time) for jobNum, filename in enumerate(files)]
    
    # Create a pool of workers
    with mp.Pool(processes=args.processors) as pool:
        # Use pool.starmap to apply function to arguments
        results = pool.starmap(stain_feature_extract, process_args)
   #%%
    for jobNum, filename in enumerate(files):
   
       stain_feature_extract(filename, args.structure_dir, dfCohort, infoDfCohort, args.cohort, args.write_dir, classLabelsList, ['Distal_collecting_tubules', 'Inflamed_tubule', 'Proximal_tubules', 'other'], jobNum, 1,start_time)



if __name__=='__main__':
    main()

end_time = time.perf_counter()
print(f"Time script took to run: {end_time-start_time} seconds ({(end_time-start_time)/60} minutes) .")











