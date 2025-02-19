###the packages we use
import time
start_time = time.perf_counter()
import argparse
import pandas as pd
import neuroCombat
import numpy as np

#function
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


def correct_batch(df,covarCols,dfInfo):
    
    covars = pd.DataFrame()
    covars['batch'] = dfInfo[covarCols]
    
    covars['batch'] = dfInfo['CompName'].str.cat(dfInfo['disease_cohort'], sep='_').astype('category').cat.codes.astype(np.int64)
    
    a = list(dfInfo['CompName'])
    b = list(dfInfo['disease_cohort'])
    
    combo = [x+'_'+y for x,y in zip(a,b)]
    uCombo = list(np.unique(combo))
    batch_combo = [uCombo.index(x) for x in combo]
    
    covars['batch'] =  batch_combo
    covars['disease_cohort'] = dfInfo['disease_cohort']
    
    categorical_cols = []

    # To specify the name of the variable that encodes for the scanner/batch covariate:
    batch_col = 'batch'
    
    toKeep = [x for x in df.columns if '-mean' in x]
    toKeep = [x for x in toKeep if 'DIC' not in x]
    print(toKeep)
    
    
    data_combat = neuroCombat.neuroCombat(dat=df[toKeep].T,
                covars=covars,
                batch_col=batch_col,
                categorical_cols=categorical_cols)["data"]

    df[toKeep] = data_combat.T
    return(df)

    
def main():
# if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r',
            '--expression_dir',
            type=str,
            default='cell_patches_1um',
            help=''
            )
    parser.add_argument('-t',
            '--info_dir',
            type=str
            
            )
    parser.add_argument('-w',
            '--write_dir',
            type=str,
            default='',
            help=''
            )
    parser.add_argument('-c',
            '--covariates',
            type=str,
            help=''
            )
    

    args,unparsed = parser.parse_known_args()

    

    df = convert_mixed_dtype_to_string(pd.DataFrame(pd.read_csv(args.expression_dir,index_col=False)))
    dfInfo = convert_mixed_dtype_to_string(pd.DataFrame(pd.read_csv(args.info_dir,index_col=False)))
    
  

    
    ###just as a check
    set1 = set(list(df.index))
    set2 = set(list(dfInfo.index))
    non_matching = set1.symmetric_difference(set2)
    print('non_matching ',non_matching)
    print('    ',np.unique(dfInfo.loc[non_matching]['CompName'].values))
    
  
    
    dfInfo = dfInfo.loc[df.index] ##so they necessarily match
    

    for col in df.columns:
        print(col)
 
    diseaseTemp = dfInfo['disease_cohort'] ##to store

    dfToExport = df.copy()
    for cohort in np.unique(diseaseTemp):
        print(f'Cohort: {cohort}')
        

        dfToExport[dfInfo['disease_cohort']==cohort] =   correct_batch(df = df[dfInfo['disease_cohort']==cohort],
                      covarCols='CompName',
                      dfInfo=dfInfo[dfInfo['disease_cohort']==cohort]
                     )
        

    df.to_csv(args.write_dir,index=False) 
   
    print("COMBATDONE")
   
 
    
    
if __name__=='__main__':
    main()
end_time = time.perf_counter()
print(f"Time script took to run: {end_time-start_time} seconds ({(end_time-start_time)/60} minutes) .")











