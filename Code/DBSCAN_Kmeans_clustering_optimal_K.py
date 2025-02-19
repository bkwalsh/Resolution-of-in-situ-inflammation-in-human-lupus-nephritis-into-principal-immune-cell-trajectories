###the packages we use
import time
start_time = time.perf_counter()
import os
import argparse
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import decomposition
import random

from sklearn import metrics
from sklearn.cluster import KMeans
#%%


def K_means_ncluster(df,exportPath,i,seed):
    
    random.seed(seed)
    sampleIndexes = random.sample(range(df.shape[0]),round(df.shape[0]*0.5))
    data_transformed = df[sampleIndexes]
    
    SilhouetteCoefficient = []
    Sum_of_squared_distances = []
    
    K = range(2,30)
    # K = range(2,4)
    for k in K:

        km = KMeans(n_clusters=k,n_init=1)
        km = km.fit(data_transformed)
        SilhouetteCoefficient.append(metrics.silhouette_score(data_transformed, km.labels_, metric='euclidean'))
        Sum_of_squared_distances.append(km.inertia_)

        with open(exportPath+'Run_'+str(i)+".pkl",'wb') as f:
            pickle.dump([SilhouetteCoefficient,Sum_of_squared_distances],f)
        print('Run_'+str(i)+".pkl","k=",str(k)," saved.")
   
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

    parser.add_argument('-w',
            '--write_dir',
            type=str,
            default='channel_expression_csvs',
            help=''
            )
    parser.add_argument('-k',
            '--optimal_k',
            type=int,
            help=''
            )
    
    parser.add_argument('-rmd',
        '--remove_doublets',
        type=str,
        # default='cell_patches_1um',
        help=''
        )
    
    # Add the argument for removing doublets
    parser.add_argument('--remove_below',
                        type=int, 
                        default=None,
                        help='Threshold for removing doublets')
    
    args,unparsed = parser.parse_known_args()


    
    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)
    if not os.path.exists(args.write_dir+f'K_{args.optimal_k}/'):
        os.mkdir(args.write_dir+f'K_{args.optimal_k}/')
        #%%
    #compiles all the DBSCAN aggregates into a single csv for cluster estimation
    DBSCANDf= pd.read_csv(args.read_dir)
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

    
    if args.remove_doublets =='True':
        DBSCANDf = DBSCANDf[DBSCANDf['total_cell_count'] != 2]
        
    if args.remove_below is not None:
        remove_threshold = int(args.remove_below)
        DBSCANDf = DBSCANDf[DBSCANDf['total_cell_count'] > remove_threshold]    
        
        
    plt.hist(DBSCANDf['total_cell_count'], bins=50, color='blue', edgecolor='black')
    
    # Add labels and title for clarity
    plt.xlabel('Total Cell Count')
    plt.ylabel('Frequency')
    plt.title('Histogram of Total Cell Count post filter')
    
    # Display the histogram
    plt.show()
    #%%
    
    columns_to_drop = ['Unnamed: 0', 'disease_cohort', 'CompName', 'AccessionNumber', 'TissueType']

    X = StandardScaler().fit_transform(DBSCANDf.drop(columns=columns_to_drop))
    components = 7
    pca = decomposition.PCA(n_components=components,svd_solver ='full')
    pca.fit(X)
    X = pca.transform(X)

    
    random_state = 42
    km = KMeans(n_clusters=args.optimal_k,n_init=10,random_state=random_state)
    
    km = km.fit(X)

    
    with open(args.write_dir+"fitted_km.pkl",'wb') as f:
        pickle.dump(km,f)

 
#%%    
    
if __name__=='__main__':
    main()
end_time = time.perf_counter()
print(f"Time script took to run: {end_time-start_time} seconds ({(end_time-start_time)/60} minutes) .")











