###the packages we use
import time
start_time = time.perf_counter()
import os

# Set the environment variable
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import argparse
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import decomposition

import random
from sklearn.cluster import KMeans
from sklearn import metrics
import ast

import multiprocessing as mp


#%%
def K_means_ncluster(df,exportPath,i,seed,percentSample):
    
    random.seed(seed)
    sampleIndexes = random.sample(range(df.shape[0]),round(df.shape[0]*percentSample))
    data_transformed = df[sampleIndexes]
    
    SilhouetteCoefficient = []
    Sum_of_squared_distances = []
    
    K = range(2,25)
    

    for k in K:

        km = KMeans(n_clusters=k,n_init=1)
        km = km.fit(data_transformed)
        SilhouetteCoefficient.append(metrics.silhouette_score(data_transformed, km.labels_, metric='euclidean'))

        Sum_of_squared_distances.append(km.inertia_)

        with open(exportPath+'Run_'+str(i)+".pkl",'wb') as f:
            pickle.dump([SilhouetteCoefficient,Sum_of_squared_distances],f)
        
#%%     

# Define the process_function at the top level of the module
def process_function(inputs):
    X, write_dir,i, seed, percent_sample = inputs[0],inputs[1],inputs[2],inputs[3],inputs[4]
    K_means_ncluster(X, write_dir, i, seed, percent_sample)

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
            '--percent_sample',
            type=str,
            default='channel_expression_csvs',
            help=''
            )

    parser.add_argument('-cd',
            '--processors',
            type=int,
            # default='cell_patches_1um',
            help=''
            )
        
    parser.add_argument('-tm',
            '--repeats',
            type=int,
            # default='cell_patches_1um',
            help=''
            )
    
    parser.add_argument('-rmd',
            '--remove_doublets',
            type=str,
            # default='cell_patches_1um',
            help=''
            )
    
    # Add the argument for removing doublets
    parser.add_argument('--remove_type',
                        type=str, 
                        default=None,
                        help='Threshold for removing doublets')
    
    parser.add_argument('--reference_value',
                        type=str, 
                        default=None,
                        help='Threshold for removing doublets')
    parser.add_argument('--DBSCAN_filename',
                        type=str, 
                        )

    args,unparsed = parser.parse_known_args()



    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)

    
    #%%
    #compiles all the DBSCAN aggregates into a single csv for cluster estimation
    DBSCANDf= pd.read_csv(args.DBSCAN_dir)
    # Identify columns that end with '_count'
    count_columns = [col for col in DBSCANDf.columns if col.endswith('_count')]
    
    # Calculate the total cell count per row
    DBSCANDf['total_cell_count'] = DBSCANDf[count_columns].sum(axis=1)

        # Create the histogram
    plt.hist(DBSCANDf['total_cell_count'], bins=50, color='blue', edgecolor='black')
    
    # Add labels and title for clarity
    plt.xlabel('Total Cell Count')
    plt.ylabel('Frequency')
    plt.title('Histogram of Total Cell Count pre filter')
    
    # Display the histogram
    plt.show()
    
    
    DBSCANDf = DBSCANDf[DBSCANDf['total_cell_count'] <= 3000]
 
    DBSCANDf = DBSCANDf[DBSCANDf['DBSCAN_label'] != -1]
    
    
    # if ast.literal_eval(args.remove_doublets):
    
    if args.remove_doublets =='True':
        DBSCANDf = DBSCANDf[DBSCANDf['total_cell_count'] != 2]
        
    if args.remove_type == 'keep_above':
        remove_threshold = int(args.reference_value)
        DBSCANDf = DBSCANDf[DBSCANDf['total_cell_count'] > remove_threshold]    
        
    if args.remove_type == 'keep_below':
         remove_threshold = int(args.reference_value)
         DBSCANDf = DBSCANDf[DBSCANDf['total_cell_count'] < remove_threshold]      
   
    if args.remove_type == 'keep_range':
        bounds = ast.literal_eval(args.reference_value)
        lower_bound = int(bounds[0])
        upper_bound = int(bounds[1])
        DBSCANDf = DBSCANDf[(DBSCANDf['total_cell_count'] >= lower_bound) & (DBSCANDf['total_cell_count'] <= upper_bound)]

   
        
    plt.hist(DBSCANDf['total_cell_count'], bins=50, color='blue', edgecolor='black')
    
    # Add labels and title for clarity
    plt.xlabel('Total Cell Count')
    plt.ylabel('Frequency')
    plt.title('Histogram of Total Cell Count post filter')
    
    # Display the histogram
    plt.show()
    #%%
    
    # Find columns with NaN values
    columns_with_nans = DBSCANDf.columns[DBSCANDf.isna().any()].tolist()

    DBSCANDf[columns_with_nans] = DBSCANDf[columns_with_nans].fillna(0)
    #%%
       # List to store columns with std = 0
    columns_to_drop = []
    
    # Loop through each column and print descriptive statistics if std != 0
    for column in DBSCANDf.columns:
        if DBSCANDf[column].dtype in ['int64', 'float64']:  # Only consider numerical columns
            std = DBSCANDf[column].std()
            if std != 0:
                print(f"Descriptive statistics for column: {column}")
                print(DBSCANDf[column].describe(), "\n")
            else:
                columns_to_drop.append(column)

    
    #%%
    
    
    # Save the filtered DataFrame to a CSV file
    DBSCANDf.drop(columns=columns_to_drop).to_csv(args.DBSCAN_dir.replace('DBSCAN_all_immune_combined.csv',args.DBSCAN_filename), index=False)  # Save without the index column
    
    

    # List of columns to drop
    columns_to_drop = columns_to_drop + ['Unnamed: 0', 'disease_cohort', 'CompName', 'AccessionNumber', 'TissueType','TileNum', 'TileCellID', 'idxs', 'DBSCAN_label']

#%%
    X = StandardScaler().fit_transform(DBSCANDf.drop(columns=columns_to_drop))
    components = 7
    pca = decomposition.PCA(n_components=components,svd_solver ='full')
    pca.fit(X)
    X = pca.transform(X)
    
    #%%
    # Use multiprocessing to parallelize the execution
    if args.processors > mp.cpu_count():
        args.processors = mp.cpu_count()
    random.seed(42)
    seeds = random.sample(range(1000000000),args.repeats)
    
   
        
        #%%

    # Create a multiprocessing pool and execute the processes
    
    
    # Assuming 'process_function' is already defined and 'args' contains necessary arguments
    inputs = [(X, args.write_dir,i, seeds[i], float(args.percent_sample)) for i in range(args.repeats)]
    
    mp.set_start_method('spawn')
    pool = mp.Pool(args.processors)
    pool.map(process_function, inputs)
    pool.close()

    
    #%%

    
    

if __name__=='__main__':
    main()

end_time = time.perf_counter()
print(f"Time script took to run: {end_time-start_time} seconds ({(end_time-start_time)/60} minutes) .")











