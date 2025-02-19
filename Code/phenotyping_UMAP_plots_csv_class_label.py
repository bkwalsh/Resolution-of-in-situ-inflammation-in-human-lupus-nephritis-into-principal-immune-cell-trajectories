###the packages we use
import time
start_time = time.perf_counter()
import os
import argparse
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
import random
import umap
import matplotlib.patches as mpatches
from sklearn.model_selection import ParameterGrid
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
def color_generate_preset(labels,colorDict):

    legends = []
    for i,lab in enumerate(np.unique(labels)):
        # print("i",i)
        legends.append(mpatches.Patch(color=colorDict[str(lab)], label=lab))
        
    color_list =[]    
    for lab in labels:
        # print("lab",lab)
        color_list.append(colorDict[str(lab)])
    return(color_list,legends)
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
            default='',
            help=''
            )

    parser.add_argument('-n',
            '--n_neighbors',
            type=int,

            )
    parser.add_argument('-d',
            '--min_dist',
            type=float,

            )
    parser.add_argument('-l',
            '--dens_lambda',

            )
    parser.add_argument('-cd',
            '--color_dict',
            type=str,
            default='',
            help=''
            )
    
    parser.add_argument('-i',
            '--immune_only',
            type=str,
            default='',
            help=''
            )
    
    
    parser.add_argument('-b',
            '--balanced_sampling',
            type=str,
            default='',
            help=''
            )

    parser.add_argument('-o',
            '--MFI_only',
            type=str,
            default='False',
            help=''
            )
    parser.add_argument('-jr',
            '--JOINT_ROOT_DIR',
            type=str,
            default='False',
            help=''
            )
    
    args,unparsed = parser.parse_known_args()

    JOINT_ROOT_DIR= args.JOINT_ROOT_DIR

    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)
        
        
    with open(JOINT_ROOT_DIR+'manual_cell_gating/FACS_analogous_plots_corrected/'+"colorDict.pkl",'rb') as f:
         colorDict = pickle.load(f)
    with open(JOINT_ROOT_DIR+'manual_cell_gating/FACS_analogous_plots_corrected/'+"legends.pkl",'rb') as f:
         legends =  pickle.load(f)   
           
#%%


    df = convert_mixed_dtype_to_string(pd.DataFrame(pd.read_csv(args.cell_expression_dir,index_col=False)))
   

    if args.immune_only=='with_other':
        classToRemove = ['Distal_collecting_tubules','Endothelial_cells','Inflamed_tubule', 'Proximal_tubules' ,'RBCs']

        # Find matching rows
        mask = ~df['class_label'].isin(classToRemove)
        df = df.loc[mask]
        
    if args.immune_only=='without_other':
        classToRemove = ['Distal_collecting_tubules','Endothelial_cells','Inflamed_tubule', 'Proximal_tubules' ,'RBCs','other']
    


        # Find matching rows
        mask = ~df['class_label'].isin(classToRemove)
        df = df.loc[mask]

    if args.immune_only=='none':
        print('post-filter',df.shape)
    

    
    #%%
    
    
    X = np.transpose(np.transpose(df)[~(df.dtypes== 'O')])
    
    X = StandardScaler().fit_transform(X)
    components = 7
    pca = decomposition.PCA(n_components=components,svd_solver ='full')
    pca.fit(X)
    X = pca.transform(X)
    
    #%%

    df.loc[X.index]['class_label'].value_counts()
    
    #%%
    ##for balanced samplingfor UMAP
    X = pd.DataFrame(data =X,index=df.index)
    nTotal = 30000 #total number to sample
    sampleSeed = 42
    if args.balanced_sampling=='none':
        
        random.seed(sampleSeed)
        labels  =np.array(df['class_label'])
        indxTemp = random.sample(list(df.index),nTotal)
        X = X.loc[indxTemp,:]
        labels = labels[indxTemp] 
        df = df.iloc[indxTemp,:] 
        
    if args.balanced_sampling=='balanced_completely':
       
       nCohort= nTotal /len(np.unique(df['disease_cohort'])) #3 cohorts 
       indxTemp =[]
       #per cohort
       for cohort in np.unique(df['disease_cohort']): ##balance per cohort
           dfTemp = df[df['disease_cohort']==cohort] ##subset the dataframe
           
           nAcc = nCohort/len(np.unique(list(dfTemp['AccessionNumber']))) ##number of cells to sample per patient
           for acc in np.unique(list(dfTemp['AccessionNumber'])):
               
               dfAcc = dfTemp[dfTemp['AccessionNumber']==acc] ##the current patient
               
               if nAcc>= dfAcc.shape[0]:
                   nAcc = dfAcc.shape[0]

               random.seed(sampleSeed)
               indxTemp.append(random.sample(list(dfAcc.index),int(nAcc)))
        
               
       indxTemp = [item for sublist in indxTemp for item in sublist]
       X = X.loc[indxTemp]
       labels =  df.loc[indxTemp]['class_label']
       df = df.loc[indxTemp]  
       
    if args.balanced_sampling=='balanced_cohort':
       
       nCohort= nTotal /len(np.unique(df['disease_cohort'])) #3 cohorts 
       indxTemp =[]
       #per cohort
       for cohort in np.unique(df['disease_cohort']): ##balance per cohort
           dfTemp = df[df['disease_cohort']==cohort] ##subset the dataframe
         
           random.seed(sampleSeed)
           indxTemp.append(random.sample(list(dfTemp.index),int(nCohort)))
        
               
       indxTemp = [item for sublist in indxTemp for item in sublist]
       X = X.loc[indxTemp]
       labels =  df.loc[indxTemp]['class_label']
       df = df.loc[indxTemp]  
       
    
    if args.MFI_only=='True':
        
        columnsKeep = [x for x in df.columns if 'Nucleus' not in x]
        df = df[columnsKeep]
   
   
#%%
        
    with open(args.color_dict+"colorDict.pkl",'rb') as f:
        colorDict = pickle.load(f)

    
    legends =[] ###have to create anew otherwise crashe if i try to laod a pickle of this
    for key in colorDict.keys():
        print(key)
        legends.append(mpatches.Patch(color=colorDict[key], label=key))

        #%%
    color_list =[]    
    for lab in labels:
        print(lab)
        color_list.append(colorDict[str(lab)])
      #%%  
      
    ###not much use in looping through this  
    seed = 42
    parameters = [{'n_neighbors': np.arange(30,150,10), 
    # 'C' :np.arange(1,300,100),
    'min_dist' :list(np.arange(0,0.3,0.05)),
    'dens_lambda':np.arange(5,50,10)}] 
    
    paramGrid = list(ParameterGrid(parameters))
    print("Number of unique parameter combinations:",len(paramGrid))
      #%%
    s =2.5
    a = 1
    labelS = 18  
      # Data preparation for UMAP
    dfTemp = df.select_dtypes(exclude='object')  # Select only numeric columns for UMAP
    reducer = umap.UMAP(random_state=42, n_neighbors=args.n_neighbors, min_dist=args.min_dist, n_components=2, dens_lambda=args.dens_lambda)
    embedding = reducer.fit_transform(StandardScaler().fit_transform(dfTemp))
    
    # Prepare DataFrame for plotting
    df_subset = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"], index=dfTemp.index)
    df_subset['class_label'] = df.loc[dfTemp.index]['class_label']  # Ensure class labels align with your embedding
    df_subset['disease_cohort'] = df.loc[dfTemp.index]['disease_cohort']  # Ensure class labels align with your embedding
    # Map class labels to colors using colorDict
    color_list = df_subset['class_label'].map(colorDict).tolist()
    
    # Plotting
    plt.figure(figsize=(10, 10))
    plt.scatter(
        x="UMAP1", y="UMAP2",
        color=color_list,  # Use the color list for plotting
        data=df_subset,
        alpha=a, s=s
    )
    
    plt.title('UMAP no PCA n' + str(args.n_neighbors) + '_d' + str(args.min_dist) + '_l' + str(args.dens_lambda), fontweight="bold", fontsize=labelS)
    plt.xlabel("UMAP1", fontweight="bold", fontsize=labelS)
    plt.ylabel("UMAP2", fontweight="bold", fontsize=labelS)
    plt.axis('equal')
    plt.xticks(fontweight="bold", fontsize=12)
    plt.yticks(fontweight="bold", fontsize=12)
    plt.tick_params(axis='both', which='both', length=0)
    plt.savefig(args.write_dir + 'UMAP_no_PCA_n' + str(args.n_neighbors) + '_d' + str(args.min_dist) + '_l' + str(args.dens_lambda) + '_class_label.png', dpi=300)
    plt.show()
    plt.close()
    
    # Save the embedding
    with open(args.write_dir + "UMAP_embeddings_no_PCA_n" + str(args.n_neighbors) + '_d' + str(args.min_dist) + '_l' + str(args.dens_lambda) + '.pkl', "wb") as f:
        pickle.dump(embedding, f)
      
      
      #%%
      
      
    reducer = umap.UMAP(random_state=42,n_neighbors=args.n_neighbors,min_dist=args.min_dist,n_components=2,dens_lambda=args.dens_lambda)
    
    dfTemp = np.transpose(np.transpose(df)[~(df.dtypes== 'O')])
    embedding = reducer.fit_transform(StandardScaler().fit_transform(dfTemp))
    
    df_subset = pd.DataFrame()
    df_subset["UMAP1"] = embedding[:,0]
    df_subset["UMAP2"] = embedding[:,1]
    plt.figure(figsize=(10,10))
    plt.scatter(
        x="UMAP1", y="UMAP2",
        color=color_list,
        data=df_subset,
        alpha=a,s=s)
    # plt.legend(handles=legends,loc='upper left', bbox_to_anchor=(1,1), ncol=2)
    plt.title('UMAP no PCA n'+str(args.n_neighbors)+'_d'+str(args.min_dist)+'_l'+str(args.dens_lambda),fontweight="bold", fontsize=labelS)
    plt.xlabel("UMAP1",fontweight="bold", fontsize=labelS)
    plt.ylabel("UMAP2",fontweight="bold", fontsize=labelS)
    plt.axis('equal')

    plt.xticks(fontweight="bold", fontsize=12)
    plt.yticks(fontweight="bold", fontsize=12)
    plt.tick_params(axis='both', which='both', length=0)
    plt.savefig(args.write_dir+'UMAP_no_PCA_n'+str(args.n_neighbors)+'_d'+str(args.min_dist)+'_l'+str(args.dens_lambda)+'.tif',dpi=300)
    plt.show()
    plt.close()
    
    with open(args.write_dir+"UMAP_embeddings_no_PCA_n"+str(args.n_neighbors)+'_d'+str(args.min_dist)+'_l'+str(args.dens_lambda)+'.pkl',"wb") as f:
        pickle.dump(embedding,f)
        
    
        
        
        
    #%%
    
    legends =[] ###have to create anew otherwise crashe if i try to laod a pickle of this
    cohortColorDict ={'Normal_Kidney':[0,1,0,1],'Lupus_Nephritis':[0,0,1,1], 'Renal_Allograft':[1,0,1,1]}
    
    color_list,legends = color_generate_preset(df['disease_cohort'].tolist(),cohortColorDict)
    #%%
    plt.figure(figsize=(10,10))
    plt.scatter(
        x="UMAP1", y="UMAP2",
        color=color_list,
        data=df_subset,
        alpha=a,s=s)
    # plt.legend(handles=legends,loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    plt.title('UMAP no PCA n'+str(args.n_neighbors)+'_d'+str(args.min_dist)+'_l'+str(args.dens_lambda),fontweight="bold", fontsize=labelS)
    plt.xlabel("UMAP1",fontweight="bold", fontsize=labelS)
    plt.ylabel("UMAP2",fontweight="bold", fontsize=labelS)
    plt.axis('equal')

    plt.xticks(fontweight="bold", fontsize=12)
    plt.yticks(fontweight="bold", fontsize=12)
    plt.tick_params(axis='both', which='both', length=0)

    plt.savefig(args.write_dir+'UMAP_no_PCA_n'+str(args.n_neighbors)+'_d'+str(args.min_dist)+'_l'+str(args.dens_lambda)+'_cohorts.png',dpi=300)
    plt.show()
    plt.close()
    
      
    #%%
    legends = []  # Create a new list for legends, avoiding crash when loading from a pickle
    cohortColorDict = {'Normal_Kidney': [0, 1, 0, 1], 'Lupus_Nephritis': [0, 0, 1, 1], 'Renal_Allograft': [1, 0, 1, 1]}
    cohorts = list(cohortColorDict.keys())  # List of cohorts
    
    for current_cohort in cohorts:
        plt.figure(figsize=(10, 10))
        # First plot the points for all non-current cohorts in gray
        for cohort in cohorts:
            if cohort != current_cohort:
                mask = df['disease_cohort'] == cohort
                plt.scatter(
                    x=df_subset[mask.values]["UMAP1"], 
                    y=df_subset[mask.values]["UMAP2"],
                    color=[0.85, 0.85, 0.85, 1],  # Gray color
                    alpha=a, s=s)
    
        # Then plot the points for the current cohort in its designated color
        mask = df['disease_cohort'] == current_cohort
        plt.scatter(
            x=df_subset[mask.values]["UMAP1"], 
            y=df_subset[mask.values]["UMAP2"],
            color=cohortColorDict[current_cohort],
            alpha=a, s=s)
    
        plt.title(f'UMAP no PCA n{args.n_neighbors}_d{args.min_dist}_l{args.dens_lambda} - {current_cohort}', fontweight="bold", fontsize=labelS)
        plt.xlabel("UMAP1", fontweight="bold", fontsize=labelS)
        plt.ylabel("UMAP2", fontweight="bold", fontsize=labelS)
        plt.axis('equal')
    
    
        plt.xticks(fontweight="bold", fontsize=12)
        plt.yticks(fontweight="bold", fontsize=12)

        plt.tick_params(axis='both', which='both', length=0)
        plt.savefig(f"{args.write_dir}UMAP_no_PCA_n{args.n_neighbors}_d{args.min_dist}_l{args.dens_lambda}_{current_cohort}_cohorts.tif", dpi=300)
        plt.show()
        plt.close()
        
        #%%
        # Loop through each cohort and plot only the current cohort with class_label colors
        df_subset['class_label'] = df['class_label'].to_list()
        df_subset['disease_cohort'] = df['disease_cohort'].to_list()
        for current_cohort in cohorts:
            plt.figure(figsize=(10, 10))
        
            # Filter data for the current cohort
            cohort_data = df_subset[df_subset['disease_cohort'] == current_cohort]
        
            # Map class_label to colors for plotting
            colors = cohort_data['class_label'].map(colorDict)
        
            # Plot each class label in its designated color
            plt.scatter(
                x='UMAP1', 
                y='UMAP2',
                color=colors,
                alpha=1, s=2.5,  # Set appropriate alpha and size
                data=cohort_data
            )
        
            # Setting up the plot with title and labels
            plt.title(f'UMAP - {current_cohort}', fontweight="bold", fontsize=18)
            plt.xlabel("UMAP1", fontweight="bold", fontsize=18)
            plt.ylabel("UMAP2", fontweight="bold", fontsize=18)
            plt.axis('equal')
            plt.xticks(fontweight="bold", fontsize=12)
            plt.yticks(fontweight="bold", fontsize=12)
            plt.tick_params(axis='both', which='both', length=0)
        
            # Save the plot to the specified directory
            plt.savefig(f"{args.write_dir}UMAP_{current_cohort}_class_labels.png", dpi=300)
            plt.show()
            plt.close()

        
    
    #%%
    MFI_cols = ['DAPI_Int-mean', 'CD86_Int-mean', 'Claudin1_Int-mean', 'RORgt_Int-mean',
           'HLAII_Int-mean', 'COLIII_Int-mean', 'CD14_Int-mean', 'Foxp3_Int-mean',
           'CD56_Int-mean', 'GZMB_Int-mean', 'IL10_Int-mean', 'CD69_Int-mean',
           'CD163_Int-mean', 'CD21_Int-mean', 'MERTK_Int-mean', 'CD11c_Int-mean',
           'SLAMF7_Int-mean', 'CD27_Int-mean', 'CD10_Int-mean', 'IFNG_Int-mean',
           'CD43_Int-mean', 'CD31_Int-mean', 'TCRD_Int-mean', 'ICOS_Int-mean',
           'iNOS_Int-mean', 'CD68_Int-mean', 'BDCA2_Int-mean', 'GZMK_Int-mean',
           'Ki67_Int-mean', 'BDCA1_Int-mean', 'CD3_Int-mean', 'GZMA_Int-mean',
           'MUC1_Int-mean', 'CD16_Int-mean', 'MXA_Int-mean', 'PD1_Int-mean',
           'CD20_Int-mean', 'CD8_Int-mean', 'Tbet_Int-mean', 'CD103_Int-mean',
           'CD4_Int-mean', 'CD45_Int-mean', 'CD138_Int-mean', 'mTOC_Int-mean']
    for col in MFI_cols:
        print('col',col)
        
        plt.figure(figsize=(10,10))
        plt.scatter(
            x="UMAP1", y="UMAP2",
            c=df[col],  # Use the values in df[col] for color intensity
            cmap='Blues',  # Choose a colormap that fits the data and visualization needs
            data=df_subset,
            alpha=a,s=s,vmax=100)
        # plt.legend(handles=legends,loc='upper left', bbox_to_anchor=(1,1), ncol=1)
        plt.title('UMAP no PCA n'+str(args.n_neighbors)+'_d'+str(args.min_dist)+'_l'+str(args.dens_lambda)+f' {col}',fontweight="bold", fontsize=labelS)
        plt.xlabel("UMAP1",fontweight="bold", fontsize=labelS)
        plt.ylabel("UMAP2",fontweight="bold", fontsize=labelS)
        plt.axis('equal')
        
        # Create a colorbar with a label
        cbar = plt.colorbar()
        cbar.set_label(col, rotation=270, labelpad=20, fontweight="bold", fontsize=labelS)
    
        plt.xticks(fontweight="bold", fontsize=12)
        plt.yticks(fontweight="bold", fontsize=12)
        plt.tick_params(axis='both', which='both', length=0)
        # plt.tight_layout()
        plt.savefig(args.write_dir+'UMAP_no_PCA_n'+str(args.n_neighbors)+'_d'+str(args.min_dist)+'_l'+str(args.dens_lambda)+f'_{col}.tif',dpi=300)
        plt.show()
        plt.close()
        
    
   
   
    #%%
    
    
    embedding = reducer.fit_transform(X)
    df_subset["UMAP1"] = embedding[:,0]
    df_subset["UMAP2"] = embedding[:,1]
    plt.figure(figsize=(10,10))
    plt.scatter(
        x="UMAP1", y="UMAP2",
        color=color_list,
        data=df_subset,
        alpha=a,s=s)
    plt.legend(handles=legends)
    plt.title("UMAP with PCA initialization",fontweight="bold", fontsize=labelS)
    plt.xlabel("UMAP1",fontweight="bold", fontsize=labelS)
    plt.ylabel("UMAP2",fontweight="bold", fontsize=labelS)
    plt.axis('equal')
    plt.xticks(fontweight="bold", fontsize=12)
    plt.yticks(fontweight="bold", fontsize=12)
    plt.savefig(args.write_dir+'UMAP_with_PCA_n'+str(args.n_neighbors)+'_d'+str(args.min_dist)+'_l'+str(args.dens_lambda)+'.tif',dpi=300)
    plt.show()
    plt.close()
   
    with open(args.write_dir+"UMAP_embeddings_with_PCA.pkl","wb") as f:
        pickle.dump(embedding,f)
    with open(args.write_dir+"filtered_expression_dataframe.pkl","wb") as f:
        pickle.dump(df,f)
    #%%
if __name__=='__main__':
    main()

end_time = time.perf_counter()
print(f"Time script took to run: {end_time-start_time} seconds ({(end_time-start_time)/60} minutes) .")










