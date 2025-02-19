###the packages we use
import time
start_time = time.perf_counter()
import os
import argparse
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.stats import  ttest_ind
from scipy.stats import mannwhitneyu
#%%
import matplotlib.font_manager as font_manager

# List all available fonts
# Check if Arial is available
fonts = [f.name for f in font_manager.fontManager.ttflist]
if "Arial" in fonts:
    plt.rcParams['font.family'] = 'Arial'
else:
    print("Arial font is not available. Using default font.")

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
#%%


def MFI_heatmap_plots_LOO_TTest(df,writeDir,filename,heatmapName):
    print(df.columns) 

    mean_columns = [col for col in df.columns if '_Int-mean' in col]
    
    # Create an empty dataframe to store Z-scores and p-values
    results_list = []
    

    for i, label in enumerate(sorted(df['class_label'].unique())):
        print(i, label)

        # Filter and group the dataframe by 'AccessionNumber' for the current and other class labels
        current_grouped = df[df['class_label'] == label][['AccessionNumber']+ mean_columns ].groupby('AccessionNumber').mean()
        other_grouped = df[df['class_label'] != label][['AccessionNumber']+ mean_columns ].groupby('AccessionNumber').mean()
        

        overall_mean = current_grouped.mean()
        overall_std = current_grouped.std()

        # Dictionary to store test results for this label
        test_results = {'class_label': label}

        for col in mean_columns:
            # Check if the column exists in both grouped DataFrames
            if col in current_grouped.columns and col in other_grouped.columns:
                print(i, label,col)
                
                    # Perform t-test for each column
                statistic, p_value = ttest_ind(current_grouped[col], other_grouped[col], equal_var=False)  # Welch's t-test
    
                # Store the results
                test_results[f'T_{col}'] = statistic
                test_results[f'p_{col}'] = p_value
    
                # Calculate T-scores for current grouped data
                current_t_scores = (current_grouped[col] - overall_mean[col]) / overall_std[col]
                test_results[f'Tscores_{col}'] = current_t_scores.tolist()

        # Add the results for this label to the results DataFrame
        results_list.append(test_results)

    return results_list


def MFI_heatmap_plots_LOO_MWU(df,writeDir,filename,heatmapName):

    mean_columns = [col for col in df.columns if '_Int-mean' in col]
    
    # Create an empty dataframe to store Z-scores and p-values
    results_list = []
    
    for i, label in enumerate(sorted(df['class_label'].unique())):
        print(i, label)

        # Filter and group the dataframe by 'AccessionNumber' for the current and other class labels
        current_grouped = df[df['class_label'] == label]#.groupby('AccessionNumber').mean()
        other_grouped = df[df['class_label'] != label]#.groupby('AccessionNumber').mean()

        # Dictionary to store test results for this label
        test_results = {'class_label': label}

        for col in mean_columns:
            # Check if the column exists in both grouped DataFrames
            if col in current_grouped.columns and col in other_grouped.columns:
                u_statistic, p_value = mannwhitneyu(current_grouped[[col,'AccessionNumber']].dropna().groupby('AccessionNumber').mean(), other_grouped[[col,'AccessionNumber']].dropna().groupby('AccessionNumber').mean(), alternative='two-sided')

                # Store the results
                test_results[f'U_{col}'] = u_statistic
                test_results[f'p_{col}'] = p_value

        # Add the results for this label to the results DataFrame

        results_list.append(test_results)


    return results_list

  
def MFI_heatmap_plots_zscore(df,writeDir,filename,heatmapName,plotCohorts,row_order):
    
    
    vmax, vmin = 6, -6
    figsize = (15, 15)
    ticksize =10
    linewidths = 2
    dpi=600
    

    mean_columns = [col for col in df.columns if '_Int-mean' in col]
    
    # Create an empty dataframe to store Z-scores and p-values
    results_df = pd.DataFrame()
    # Create a mapping of AccessionNumber to disease_cohort
    cohort_mapping = df[['AccessionNumber', 'disease_cohort']].drop_duplicates()
    cohorts_hold_list = []
    # Create an empty dataframe to store Z-scores
    ACCZscoreDf = pd.DataFrame()
    
    
    
    for i, label in enumerate(sorted(df['class_label'].unique())):
        print(i, label)
        
        
        # Calculate means for the current class label
        current_means = df[df['class_label'] == label].groupby('AccessionNumber')[mean_columns].mean()
        
        # Calculate means for all other class labels
        other_means = df[df['class_label'] != label].groupby('AccessionNumber')[mean_columns].mean()
    
        # Compute combined mean and standard deviation for all other class labels
        other_combined_mean = other_means.mean()
        other_combined_std = other_means.std(ddof=1)
    
        # Compute Z-scores for each MFI in the current class compared to other classes for each AccessionNumber
        z_scores = (current_means - other_combined_mean) / other_combined_std
    
        # Include disease_cohort and class_label in the z_scores dataframe
        z_scores = z_scores.merge(cohort_mapping, on='AccessionNumber', how='left')
        z_scores['class_label'] = label  # Add a column for class_label
    
        # Append the results to the results_df
        ACCZscoreDf = pd.concat([ACCZscoreDf, z_scores])
    

    # Reset index of the results dataframe
    ACCZscoreDf.reset_index(drop=True, inplace=True)
    print(ACCZscoreDf)

    if plotCohorts:
        
        for cohort in sorted(ACCZscoreDf['disease_cohort'].unique()):
           
            
            # Filter and average Z-scores for each cohort
            cohort_df = ACCZscoreDf[ACCZscoreDf['disease_cohort'] == cohort]
            mean_df = cohort_df.drop(['AccessionNumber','disease_cohort'],axis=1).groupby('class_label').mean()
            # mean_df_dropped = mean_df.drop(columns=colsToDrop)
            mean_df_dropped = mean_df.loc[row_order]
            mean_df_dropped = mean_df_dropped[sorted(mean_df_dropped.columns)]
        
            # Plotting
            num_rows, num_cols = mean_df_dropped.shape
        
            plt.figure(figsize=figsize)
            ax = sns.heatmap(mean_df_dropped, cmap="PRGn", center=0, annot=False, cbar=True,
                             vmax=vmax, vmin=vmin, linewidths=linewidths, linecolor='black',xticklabels=True, yticklabels=True)
            ax.set_aspect('equal')
            
        
            for label in ax.get_xticklabels():
                label.set_weight('bold')
                label.set_size(ticksize)
        
            for label in ax.get_yticklabels():
                label.set_weight('bold')
                label.set_size(ticksize)
        
            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(linewidths)
            
            plt.title(f"Z-scores Heatmap for {cohort}")
            plt.savefig(f"{writeDir}zscore_{cohort}_heatmap"+filename, dpi=dpi)
            plt.tight_layout()
            plt.show()
            plt.close()
    else:

        
        # Filter and average Z-scores for everything togerther
        cohort_df = ACCZscoreDf
        mean_df = cohort_df.drop(['AccessionNumber','disease_cohort'],axis=1).groupby('class_label').mean()
        mean_df_dropped = mean_df.loc[row_order]
        
        mean_df_dropped = mean_df_dropped[sorted(mean_df_dropped.columns)]
    
        # Plotting
        num_rows, num_cols = mean_df_dropped.shape
        
       
    
        plt.figure(figsize=figsize)
        ax = sns.heatmap(mean_df_dropped, cmap="PRGn", center=0, annot=False, cbar=True,
                         vmax=vmax, vmin=vmin, linewidths=linewidths, linecolor='black',xticklabels=True, yticklabels=True)
        ax.set_aspect('equal')

    
        for label in ax.get_xticklabels():
            label.set_weight('bold')
            label.set_size(ticksize)
    
        for label in ax.get_yticklabels():
            label.set_weight('bold')
            label.set_size(ticksize)
    
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(linewidths)
        
        plt.title(f"Z-scores Heatmap for all")
        plt.savefig(f"{writeDir}zscore_all_heatmap"+filename, dpi=dpi)
        plt.tight_layout()
        plt.show()
        plt.close()
        
  
        results_df= ACCZscoreDf.drop(['AccessionNumber','disease_cohort'],axis=1)
        # results_df.reset_index(drop=True, inplace=True)

        # Filter out Z-scores from results_df
        z_scores_df = results_df[[col for col in results_df.columns if not col.startswith('p_')]].set_index('class_label')
        
        colsToDrop =['DAPI_Int-mean','IL10_Int-mean','CD86_Int-mean','RORgt_Int-mean',
                    "COLIII_Int-mean", "GZMB_Int-mean", "CD69_Int-mean", "SLAMF7_Int-mean", 
        "CD27_Int-mean", "IFNG_Int-mean", "CD43_Int-mean", "iNOS_Int-mean", 
        "GZMK_Int-mean", "Ki67_Int-mean", "GZMA_Int-mean", 
        "MXA_Int-mean", "Tbet_Int-mean", "mTOC_Int-mean","CD21_Int-mean"]
        z_scores_df_dropped = z_scores_df.drop(columns=colsToDrop)
        z_scores_df_dropped = z_scores_df_dropped[sorted(z_scores_df_dropped.columns)]
        # Plot the heatmap
        # Determine square size based on data shape
        num_rows, num_cols = z_scores_df_dropped.shape
        size = max(num_rows, num_cols)  # Taking the larger value in case they differ
        figsize = (10,10)
        linewidths=3
        # Plot the heatmap with square size
        plt.figure(figsize=figsize)
        ax = sns.heatmap(z_scores_df_dropped,  cmap="PRGn", center=0, annot=False, cbar=True, 
                         vmax=vmax, vmin=vmin, linewidths=linewidths,linecolor='black')
        ax.set_aspect('equal')  # Set the aspect of the axis to be "equal"
        # Make x and y tick labels bold and larger
        for label in ax.get_xticklabels():
            label.set_weight('bold')
            label.set_size(14)  # You can adjust this value as per your preference
        
        for label in ax.get_yticklabels():
            label.set_weight('bold')
            label.set_size(14)  # You can adjust this value as per your preference
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(linewidths)  # Set the thickness of the border
    
        plt.title("Z-scores Heatmap by class_label"+heatmapName)
        plt.savefig(writeDir+'zscore_cell_class_MFI_main'+filename,dpi=300)
        plt.tight_layout()
        plt.show()
        plt.close()

        # Filter out Z-scores from results_df
        z_scores_df = results_df[[col for col in results_df.columns if not col.startswith('p_')]].set_index('class_label')
        colsToDrop =['DAPI_Int-mean','CD86_Int-mean','RORgt_Int-mean',
                    "COLIII_Int-mean", "GZMB_Int-mean", "CD69_Int-mean", "SLAMF7_Int-mean", 
        "CD27_Int-mean", "IFNG_Int-mean", "CD43_Int-mean", "iNOS_Int-mean", 
        "GZMK_Int-mean", "Ki67_Int-mean", "GZMA_Int-mean", 
        "MXA_Int-mean", "Tbet_Int-mean", "mTOC_Int-mean" ,"CD21_Int-mean"]
        z_scores_df_dropped = z_scores_df[colsToDrop]
        z_scores_df_dropped = z_scores_df_dropped[sorted(z_scores_df_dropped.columns)]
        # Plot the heatmap
        # Determine square size based on data shape
        num_rows, num_cols = z_scores_df_dropped.shape
        size = max(num_rows, num_cols)  # Taking the larger value in case they differ
        figsize = (10,10)
        
        # Plot the heatmap with square size
        plt.figure(figsize=figsize)
        ax = sns.heatmap(z_scores_df_dropped, cmap="coolwarm", center=0, annot=False, cbar=True, 
                         vmax=vmax, vmin=vmin, linewidths=linewidths,linecolor='black')
        ax.set_aspect('equal')  # Set the aspect of the axis to be "equal"
        # Make x and y tick labels bold and larger
        
        # Set ticks for all labels
        num_rows, num_cols = z_scores_df_dropped.shape
        ax.set_xticks(range(num_cols))
        ax.set_yticks(range(num_rows))
        
        # Set labels for all ticks
        ax.set_xticklabels(z_scores_df_dropped.columns, rotation=90, ha='right', weight='bold')
        ax.set_yticklabels(z_scores_df_dropped.index, rotation=0, weight='bold')

        
        
        for label in ax.get_xticklabels():
            label.set_weight('bold')
            label.set_size(14)  # You can adjust this value as per your preference
        
        for label in ax.get_yticklabels():
            label.set_weight('bold')
            label.set_size(14)  # You can adjust this value as per your preference
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(linewidths)  # Set the thickness of the border
        
        plt.title("Z-scores Heatmap by class_label"+heatmapName)
        plt.tight_layout()
        plt.savefig(writeDir+'zscore_cell_class_MFI_supplemental'+filename,dpi=300)
        plt.show()
        plt.close()
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



    #%%
    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)

    df = convert_mixed_dtype_to_string(pd.DataFrame(pd.read_csv(args.cell_expression_dir,index_col=False)))
    

    
    #%%
    df = df[df['class_label'] != 'general_unknown']
    infoDf = convert_mixed_dtype_to_string(pd.DataFrame(pd.read_csv(args.cell_info_dir,index_col=False)))
    infoDf = infoDf.loc[df.index] ##so they necessarily match
    
    
    #%%
    
    
    row_order = ['Bcell','Plasmablasts','plasma_cells','CD3+_Tcell','CD4+CD8+_Tcell','CD4+Foxp3+_Tcell',
                 'CD4+ICOS+PD1+_Tcell','CD4+ICOS+_Tcell','CD4+PD1+_Tcell','CD4+_Tcell','CD4+_intraepithelial_Tcell',
                 'CD8+Foxp3+_Tcell','CD8+ICOS+PD1+_Tcell','CD8+ICOS+_Tcell','CD8+PD1+_Tcell','CD8+_Tcell',
                 'CD8+_intraepithelial_Tcell','TCRgd_Tcell','NK_Tcell','NK_cells','CD14+CD163+MERTK+_macrophages',
                 'CD14+CD163+_macrophages','CD14+MERTK+_macrophages','CD14+_macrophages','CD16+_macrophages',
                 'HLAII+_Monocytes','HLAII-_Monocytes','Mo-Macrophage','Neutrophils','BDCA2pCD103p_DCs',
                 'cDC1','cDC2','pDC','Distal_collecting_tubules','Endothelial_cells','Inflamed_tubule',
                 'Proximal_tubules','RBCs','general_unknown']
    
    MFI_heatmap_plots_zscore(df=df,writeDir=args.write_dir,filename='.png',heatmapName=' all cohorts',plotCohorts=False,row_order= row_order)
    #%%
    MFI_heatmap_plots_zscore(df=df,writeDir=args.write_dir,filename='.png',heatmapName=' all cohorts',plotCohorts=True,row_order= row_order)
    
    
    #%%
    dfTemp= df.copy()
    dfTemp['class_label'].replace({'CD4+ICOS+_Tcell':'CD4+_Tcell', 
                                        'CD4+PD1+_Tcell':'CD4+_Tcell',
                                        'CD4+Foxp3+_Tcell':'CD4+_Tcell', 
                                        'CD4+ICOS+PD1+_Tcell':'CD4+_Tcell',
                                        'CD14+CD163+MERTK+_macrophages':'CD14+CD163+_macrophages',
                                        'CD8+Foxp3+_Tcell':'CD8+_Tcell', 
                                        'CD8+ICOS+PD1+_Tcell':'CD8+_Tcell', 
                                        'CD8+ICOS+_Tcell':'CD8+_Tcell',
                                        'CD8+PD1+_Tcell':'CD8+_Tcell'}, inplace=True)
    MFI_heatmap_plots_zscore(df=dfTemp,writeDir=args.write_dir,filename='_simplified.tiff',heatmapName=' all cohorts',plotCohorts=False)
    #%%
    dfTemp= df.copy()
    dfTemp['class_label'].replace({'CD4+ICOS+_Tcell':'CD4+_Tcell', 
                                        'CD4+PD1+_Tcell':'CD4+_Tcell',
                                        'CD4+Foxp3+_Tcell':'CD4+_Tcell', 
                                        'CD4+ICOS+PD1+_Tcell':'CD4+_Tcell',
                                        'CD14+CD163+MERTK+_macrophages':'CD14+CD163+_macrophages',
                                        'CD8+Foxp3+_Tcell':'CD8+_Tcell', 
                                        'CD8+ICOS+PD1+_Tcell':'CD8+_Tcell', 
                                        'CD8+ICOS+_Tcell':'CD8+_Tcell',
                                        'CD8+PD1+_Tcell':'CD8+_Tcell'}, inplace=True)
    MFI_heatmap_plots_zscore(df=dfTemp,writeDir=args.write_dir,filename='_simplified_Ztogether.tiff',heatmapName=' all cohorts',plotCohorts=True)
    
    
    #%%
    
    for cohort in df['disease_cohort'].unique():
        MFI_heatmap_plots_zscore(df=df[df['disease_cohort']==cohort],writeDir=args.write_dir,filename=f'_{cohort}.tiff',heatmapName=f' {cohort}',plotCohorts=False)
        dfTemp= df[df['disease_cohort']==cohort].copy()
        dfTemp['class_label'].replace({'CD4+ICOS+_Tcell':'CD4+_Tcell', 
                                            'CD4+PD1+_Tcell':'CD4+_Tcell',
                                            'CD4+Foxp3+_Tcell':'CD4+_Tcell', 
                                            'CD4+ICOS+PD1+_Tcell':'CD4+_Tcell',
                                            'CD14+CD163+MERTK+_macrophages':'CD14+CD163+_macrophages',
                                            'CD8+Foxp3+_Tcell':'CD8+_Tcell', 
                                            'CD8+ICOS+PD1+_Tcell':'CD8+_Tcell', 
                                            'CD8+ICOS+_Tcell':'CD8+_Tcell',
                                            'CD8+PD1+_Tcell':'CD8+_Tcell'}, inplace=True)
        MFI_heatmap_plots_zscore(df=dfTemp,writeDir=args.write_dir,filename=f'_{cohort}_simplified_Zseperate.tiff',heatmapName=f' {cohort}',plotCohorts=False)
  
    
    
    result = MFI_heatmap_plots_LOO_MWU(df=df,writeDir=args.write_dir,filename='.tiff',heatmapName=' all cohorts')
    result = MFI_heatmap_plots_LOO_TTest(df=df,writeDir=args.write_dir,filename='.tiff',heatmapName=' all cohorts')
    
  
    # Prepare a dictionary to store the reformatted data
    data = {}
    
    # Loop through each item in the original list
    for item in result:
        class_label = item['class_label']
        # Extract only 'U_{}_Int-mean' measurements
        data[class_label] = {k: v for k, v in item.items() if 'T_' in k and '_Int-mean' in k}
    
    # Convert the dictionary into a DataFrame
    result_df = pd.DataFrame.from_dict(data, orient='index')
    
    
    #%%
    
        # Set up the matplotlib figure
    # Set up the matplotlib figure with a larger size
    plt.figure(figsize=(20, 16))  # Adjust the size based on the number of columns and rows
    
    # Draw the heatmap with smaller annotation font size for better fit
    ax = sns.heatmap(result_df, annot=False, cmap='RdBu', fmt='g', annot_kws={"size": 8},vmax=10,vmin=-10, center =0)
    
    # Rotate x and y labels to fit them better
    plt.xticks(rotation=90)  # Rotate x labels for better visibility
    plt.yticks(rotation=0)  # Keep y labels horizontal or rotate as needed
    
    # Ensure all labels are shown
    ax.set_xticklabels(ax.get_xticklabels(), minor=False)
    ax.set_yticklabels(ax.get_yticklabels(), minor=False)
    
    # Adjust the aspect ratio to make more space for each cell
    ax.set_aspect("equal")
    
    # Add titles and labels with adjusted font size if necessary
    plt.title('Heatmap of Measurements', fontsize=16)
    plt.xlabel('Markers', fontsize=14)
    plt.ylabel('Class Labels', fontsize=14)
    
    # Show the plot
    plt.show()

    
    
    
if __name__=='__main__':
    main()

end_time = time.perf_counter()
print(f"Time script took to run: {end_time-start_time} seconds ({(end_time-start_time)/60} minutes) .")











