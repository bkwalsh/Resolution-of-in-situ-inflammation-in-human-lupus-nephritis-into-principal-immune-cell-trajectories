###the packages we use
import time
start_time = time.perf_counter()
import argparse
import pandas as pd
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
        
def main():
# if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r',
            '--read_dir',
            type=str,
            default='cell_patches_1um',
            help=''
            )

    parser.add_argument('-w',
            '--write_dir',
            type=str,
            default='channel_expression_csvs',
            help=''
            )


    args,unparsed = parser.parse_known_args()

    df = convert_mixed_dtype_to_string(pd.DataFrame(pd.read_csv(args.read_dir,index_col=False)))


    
    columnsKeep = [s for s in list(df.columns) if "Int-mean" in s]
    columnsKeep2 = ['Nucleus-area','Nucleus-axis_major_length','Nucleus-axis_minor_length','Nucleus-eccentricity','Nucleus-extent','Nucleus-perimeter','Nucleus-solidity'] ##for QC gating on the segmentations
    
    columnsKeep = columnsKeep + columnsKeep2
    
    df=df[columnsKeep]
    
    df.to_csv(args.write_dir,index=False) ##stupid but seems more compatible with the scapy package to do this
    

       
    
if __name__=='__main__':
    main()
end_time = time.perf_counter()
print(f"Time script took to run: {end_time-start_time} seconds ({(end_time-start_time)/60} minutes) .")











