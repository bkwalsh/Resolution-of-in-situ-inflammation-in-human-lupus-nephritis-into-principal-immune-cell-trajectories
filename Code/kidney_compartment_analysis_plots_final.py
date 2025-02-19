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
from skimage import io
import seaborn as sns
from scipy.stats import mannwhitneyu
from itertools import combinations

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
def stain_feature_extract(filename,structurePath,dfCohort,infoDfCohort,cohort,writeDir,classLabelsList,classIgnore,jobNum,thresholdMask,start_time):

    props_dict = collections.defaultdict(list)
    compName = '_'.join(filename.split('_')[0:2])
    img = io.imread(structurePath+filename)
    img[img<thresholdMask] = 0
    img[img>=thresholdMask] = 1
    
    
    img = np.squeeze(img)#[:20000,:20000]
    
    dfComp = dfCohort[infoDfCohort['CompName']==compName]
    infoDfComp = infoDfCohort[infoDfCohort['CompName']==compName]
    combined_df = pd.concat([dfComp, infoDfComp], axis=1)
   
            
    ind = np.where(img)
    all_pts = [[x,y] for x,y in zip(ind[0],ind[1])]
    print('all_pts',all_pts[:20]) 
       
    
        # Convert all_pts into a DataFrame
    pts_df = pd.DataFrame(all_pts, columns=['GlobalMaskCentroidRow', 'GlobalMaskCentroidCol'])
    
    # Convert the columns to integer type for proper matching
    pts_df['GlobalMaskCentroidRow'] = pts_df['GlobalMaskCentroidRow'].astype(int)
    pts_df['GlobalMaskCentroidCol'] = pts_df['GlobalMaskCentroidCol'].astype(int)
    
    # Merge with the original DataFrame to find matching rows
    matching_rows = combined_df.merge(pts_df, on=['GlobalMaskCentroidRow', 'GlobalMaskCentroidCol'])


    matching_rows.to_csv(writeDir+f'{compName}.csv',index=False)
    

def plot_cell_class_proportions_stacked(df,colorDict,cohort,col,write_dir, save_name=None):
    """
    Plots a stacked bar graph showing the proportion of cell classes based on the 'class_label' column,
    filtered by rows where 'inside_interstitium' is True, and grouped by the 'CompName' column.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the relevant data.
    """
    # Filter rows where 'inside_interstitium' is True
    filtered_df = df[df[col]]
      

    # Group by 'CompName' and 'class_label' and count occurrences
    grouped = filtered_df.groupby(['AccessionNumber', 'class_label']).size().reset_index(name='count')

    # Calculate proportions
    total_counts = grouped.groupby('AccessionNumber')['count'].transform('sum')
    grouped['proportion'] = grouped['count'] / total_counts

    # Pivot table for plotting
    pivot_df = grouped.pivot(index='AccessionNumber', columns='class_label', values='proportion').fillna(0)
    class_labels = pivot_df.columns
    colors = [colorDict[label] for label in class_labels if label in colorDict]


    rotation_angle = 90
    dpi = 300
    # Plot the stacked bar graph
    ax = pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors)

    ax.set_xlabel('AccessionNumber')
    ax.set_ylabel('Proportion')
    ax.set_title(f'{cohort}')
    plt.xticks(rotation=rotation_angle)
    plt.legend(title='Class Label', loc='upper left', frameon=False).remove()  # Remove legend
    if save_name:
        plt.savefig(write_dir+f'cell_label_compName_{save_name}.tif', bbox_inches='tight',dpi=dpi)
    plt.show()
    
    
    # Group by 'disease_cohort' and 'class_label' and calculate total proportions
    total_grouped = filtered_df.groupby(['disease_cohort', 'class_label']).size().reset_index(name='count')
    total_counts_cohort = total_grouped.groupby('disease_cohort')['count'].transform('sum')
    total_grouped['proportion'] = total_grouped['count'] / total_counts_cohort

    # Pivot table for total proportions
    total_pivot_df = total_grouped.pivot(index='disease_cohort', columns='class_label', values='proportion').fillna(0)

    # Plot the stacked bar graph for total proportions across all CompName
    ax_total = total_pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors)
    ax_total.set_xlabel('Disease Cohort')
    ax_total.set_ylabel('Total Proportion')
    ax_total.set_title(f'{cohort} - Total Proportion Across All CompNames')
    plt.xticks(rotation=rotation_angle)
    plt.legend(title='Class Label', loc='upper left', frameon=False).remove()  # Remove legend
    if save_name:
        plt.savefig(write_dir+f'cell_label_cohort_{save_name}.tif', bbox_inches='tight',dpi=dpi)
    
    plt.show()
def plot_cell_class_proportions_stacked_areaNormalized(df, area_df, area_col, colorDict, cohort, col, write_dir, save_name=None):
    """
    Plots a stacked bar graph showing the proportion of cell classes based on the 'class_label' column,
    normalized by the area provided in area_df, filtered by rows where 'inside_interstitium' is True,
    and grouped by the 'CompName' column.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the relevant data.
    area_df (pandas.DataFrame): The DataFrame containing the areas for normalization.
    area_col (str): The column name in area_df that contains the area values.
    """
    # Merge df with area_df on 'CompName' and 'disease_cohort'
    merged_df = df.merge(area_df, on=['CompName', 'disease_cohort'], how='left')

    # Filter rows based on the specified condition (e.g., 'inside_interstitium' is True)
    filtered_df = merged_df[merged_df[col]]

    # Group by 'AccessionNumber', 'class_label', count occurrences, and normalize by area
    grouped = filtered_df.groupby(['AccessionNumber', 'class_label']).apply(
        lambda x: len(x) / x[area_col].iloc[0]
    ).reset_index(name='normalized_count')

    # Calculate proportions
    total_counts = grouped.groupby('AccessionNumber')['normalized_count'].transform('sum')
    grouped['proportion'] = grouped['normalized_count'] / total_counts

    # Pivot table for plotting
    pivot_df = grouped.pivot(index='AccessionNumber', columns='class_label', values='proportion').fillna(0)
    class_labels = pivot_df.columns
    colors = [colorDict[label] for label in class_labels if label in colorDict]

    
    rotation_angle = 90
    dpi = 300
    # Plot the stacked bar graph
    ax = pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors, edgecolor='black', linewidth=0.5,width=0.95)

    ax.set_xlabel('AccessionNumber')
    ax.set_ylabel('Proportion')
    ax.set_title(f'{cohort}')
    ax.set_ylim(0, 1) 
    
    for label in plt.gca().get_xticklabels():
        label.set_weight('bold')
        label.set_size(11)  # Adjust size as needed
    for label in plt.gca().get_yticklabels():
        label.set_weight('bold')
        label.set_size(11)  # Adjust size as needed
    # Make the lines around the plot thicker
    ax = plt.gca()  # Get current axes
    linewidth = 4  # Set the desired linewidth
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)
    
    plt.xticks(rotation=rotation_angle)
    plt.legend(title='Class Label', loc='upper left', frameon=False).remove()  # Remove legend
    if save_name:
        plt.savefig(write_dir+f'cell_label_AccNum_{save_name}.tif', bbox_inches='tight',dpi=dpi)
    plt.show()
    
    if False:
        # Group by 'disease_cohort' and 'class_label' and calculate total proportions
        total_grouped = filtered_df.groupby(['disease_cohort', 'class_label']).size().reset_index(name='count')
        total_counts_cohort = total_grouped.groupby('disease_cohort')['count'].transform('sum')
        total_grouped['proportion'] = total_grouped['count'] / total_counts_cohort
    
        # Pivot table for total proportions
        total_pivot_df = total_grouped.pivot(index='disease_cohort', columns='class_label', values='proportion').fillna(0)
    
        # Plot the stacked bar graph for total proportions across all CompName
        ax_total = total_pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors)
        ax_total.set_xlabel('Disease Cohort')
        ax_total.set_ylabel('Total Proportion')
        ax_total.set_title(f'{cohort} - Total Proportion Across All CompNames')
        plt.xticks(rotation=rotation_angle)
        plt.legend(title='Class Label', loc='upper left', frameon=False).remove()  # Remove legend
        if save_name:
            plt.savefig(write_dir+f'cell_label_cohort_{save_name}.tif', bbox_inches='tight',dpi=dpi)
        
        plt.show()

def update_infoDf(infoDf, structure_dir, col, reference_substring=None, ignore_filenames=[]):
    """
    Updates the infoDf DataFrame based on the presence of specific files in a directory structure.

    Parameters:
    infoDf (pandas.DataFrame): The DataFrame to be updated.
    structure_dir (str): The directory containing subfolders with files.
    col (str): The column in infoDf to be updated.
    reference_substring (str, optional): A substring to check for in filenames. Defaults to None.
    ignore_filenames (list of str, optional): A list of filenames to ignore. Defaults to an empty list.
    """
    infoDf[col] = False
    folders = os.listdir(structure_dir)

    print("folders", folders)
    for folder in folders:
        files = os.listdir(os.path.join(structure_dir, folder))
        files = [x for x in files if 'tif' not in x]

        for filename in files:
            # Skip if filename is in the ignore list
            if filename in ignore_filenames:
                continue

            # Check if filename contains the reference substring (if provided)
            if reference_substring is None or reference_substring in filename:
                csv_path = os.path.join(structure_dir, folder, filename)
                csvHold = pd.read_csv(csv_path, index_col=False)
                

                # Create a temporary DataFrame with only the columns used for matching
                matching_columns = ['CompName', 'disease_cohort', 'TileNum', 'TileCellID']
                temp_df = csvHold[matching_columns].drop_duplicates()

                # Update the specified column in infoDf for matching rows
                condition = (infoDf[matching_columns].isin(temp_df.to_dict('list')).all(axis=1))
                infoDf.loc[condition, col] = True


    return infoDf
def plot_cell_class_proportions_heatmap(df, cohort, col, write_dir, save_name=None):
    """
    Plots a heatmap showing the proportion of cell classes based on the 'class_label' column,
    filtered by rows where 'inside_interstitium' is True, and grouped by the 'disease_cohort' column.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the relevant data.
    """
    # Filter rows where 'inside_interstitium' is True
    filtered_df = df[df[col]]
    

    # Group by 'disease_cohort' and 'class_label' and count occurrences
    grouped = filtered_df.groupby(['disease_cohort', 'class_label']).size().reset_index(name='count')

    # Calculate proportions
    total_counts_cohort = grouped.groupby('disease_cohort')['count'].transform('sum')
    grouped['proportion'] = grouped['count'] / total_counts_cohort

    # Pivot table for heatmap
    heatmap_df = grouped.pivot(index='class_label', columns='disease_cohort', values='proportion').fillna(0)

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(heatmap_df, annot=True, cmap='viridis')
    ax.set_xlabel('Disease Cohort')
    ax.set_ylabel('Class Label')
    ax.set_title(f'{cohort} - Cell Class Proportions Heatmap')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_name:
        plt.savefig(write_dir + f'cell_label_heatmap_{save_name}.tif', bbox_inches='tight', dpi=300)
    
    plt.show()
    
    # Calculate proportions across cohorts
    total_counts_label = grouped.groupby('class_label')['count'].transform('sum')
    grouped['proportion_across'] = grouped['count'] / total_counts_label

    # Pivot table for the second heatmap
    heatmap_df_across = grouped.pivot(index='class_label', columns='disease_cohort', values='proportion_across').fillna(0)

    # Plot the second heatmap
    plt.figure(figsize=(10, 6))
    ax2 = sns.heatmap(heatmap_df_across, annot=True, cmap='viridis')
    ax2.set_xlabel('Disease Cohort')
    ax2.set_ylabel('Class Label')
    ax2.set_title(f'{cohort} - Proportion of Total Class Labels in each Cohort')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    if save_name:
        plt.savefig(write_dir + f'cell_label_proportion_across_cohort_{save_name}.tif', bbox_inches='tight', dpi=300)

    plt.show()
    
def calculate_std_of_density(total_grouped_list, area_names, grouping_column='disease_cohort', cell_classes=None):
    """
    Calculates the standard deviation of cell densities for each area, cell class, and cohort grouping.

    Parameters:
    - total_grouped_list (list of pandas.DataFrame): List of DataFrames with normalized counts for each area.
    - area_names (list of str): Names of the areas corresponding to the DataFrames in total_grouped_list.
    - grouping_column (str): Column name for cohort grouping, typically 'disease_cohort'.
    - cell_classes (list of str): List of cell class labels to analyze.

    Returns:
    - std_df (pandas.DataFrame): DataFrame containing the standard deviations, with columns for area, cell class,
                                 cohort group, and standard deviation.
    """
    std_results = []

    for class_label in cell_classes:
        # Combine data for the current class_label across all areas into a single DataFrame
        combined_df = pd.DataFrame()
        for i, df in enumerate(total_grouped_list):
            filtered_df = df[df['class_label'] == class_label].copy()
            filtered_df['Area'] = area_names[i]  # Add area information
            combined_df = pd.concat([combined_df, filtered_df], ignore_index=True)

        # Group by area and disease cohort to calculate std
        grouped = combined_df.groupby(['Area', grouping_column])
        std = grouped['normalized_count'].std().reset_index(name='Standard Deviation')

        # Add cell class info to each row in the std DataFrame
        std['Cell Class'] = class_label
        std_results.append(std)

    # Concatenate all std results into a single DataFrame
    std_df = pd.concat(std_results, ignore_index=True)
    return std_df
    
def perform_mwu_tests(total_grouped_list, area_names, grouping_column='disease_cohort', cell_classes=None, cohort_order=None):
    """
    Performs Mann-Whitney U tests for all unique pairwise comparisons within each disease cohort,
    for each cell class and area, ensuring cohorts are analyzed in a consistent order.

    Parameters:
    - total_grouped_list (list of pandas.DataFrame): List of DataFrames with normalized counts for each area.
    - area_names (list of str): Names of the areas corresponding to the DataFrames in total_grouped_list.
    - grouping_column (str): Column name for cohort grouping, typically 'disease_cohort'.
    - cell_classes (list of str): List of cell class labels to analyze.
    - cohort_order (list of str): Fixed order of cohorts to be considered for comparisons.

    Returns:
    - results_df (pandas.DataFrame): DataFrame containing the test results, with columns for area, cell class,
                                     comparison groups, and p-values.
    """
    results = []

    for class_label in cell_classes:
        combined_df = pd.DataFrame()
        for i, df in enumerate(total_grouped_list):
            filtered_df = df[df['class_label'] == class_label].copy()
            filtered_df['Area'] = area_names[i]
            combined_df = pd.concat([combined_df, filtered_df], ignore_index=True)

        for area in combined_df['Area'].unique():
            area_df = combined_df[combined_df['Area'] == area]
            
            # Make sure the column is ordered according to the cohort_order
            if cohort_order:
                area_df[grouping_column] = pd.Categorical(area_df[grouping_column], categories=cohort_order, ordered=True)
                area_df.sort_values(by=grouping_column, inplace=True)
            
            cohorts = area_df[grouping_column].unique()  # This now respects the order given by cohort_order

            for group1, group2 in combinations(cohorts, 2):
                data1 = area_df[area_df[grouping_column] == group1]['normalized_count']
                data2 = area_df[area_df[grouping_column] == group2]['normalized_count']
                stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                results.append({
                    'Area': area,
                    'Cell Class': class_label,
                    'Group 1': group1,
                    'Group 2': group2,
                    'P-value': p_value
                })

    results_df = pd.DataFrame(results)
    return results_df    


def plot_cell_class_proportions_heatmap_areaNormalized(df,area_df, area_col, cohort, col, write_dir,grouping_column, save_name=None, vmin_1=None, vmax_1=None, vmin_2=None, vmax_2=None, grouping_order=None,row_order=None):
    """
    Plots a heatmap showing the proportion of cell classes based on the 'class_label' column,
    filtered by rows where 'inside_interstitium' is True, and grouped by the 'disease_cohort' column.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the relevant data.
    """

    filtered_df = df[df[col]]

    # Group by 'disease_cohort', 'CompName', 'class_label' and count occurrences
    grouped = filtered_df.groupby([grouping_column, 'CompName', 'class_label','AccessionNumber']).size().reset_index(name='count')


    total_counts_per_cohort = grouped.groupby(grouping_column)['count'].sum()
    # Merge the grouped DataFrame with area_df on 'CompName' and 'disease_cohort'
    merged_grouped = grouped.merge(area_df, on=['CompName', grouping_column], how='left')

    # Normalize counts by area
    merged_grouped['normalized_count'] = merged_grouped['count'] / merged_grouped[area_col]

    # Calculate average counts per patient
    patient_avg = merged_grouped.groupby(['AccessionNumber', grouping_column, 'class_label'])['normalized_count'].mean().reset_index()

    # Group by 'disease_cohort' and calculate total proportions
    total_grouped = patient_avg.groupby([grouping_column, 'class_label']).sum().reset_index()
    total_grouped = total_grouped.rename(columns={'normalized_count': 'avg_normalized_count'})
       
    
    # Reorder rows by 'class_label' if a specific order is provided
    if row_order:
        if not set(row_order).issubset(set(total_grouped['class_label'])):
            raise ValueError("row_order contains invalid class labels.")
        total_grouped['class_label'] = pd.Categorical(total_grouped['class_label'], categories=row_order, ordered=True)
        total_grouped.sort_values(by='class_label', inplace=True)
    

    # Calculate proportions within each cohort
    total_counts_cohort = total_grouped.groupby(grouping_column)['avg_normalized_count'].transform('sum')
    total_grouped['proportion'] = total_grouped['avg_normalized_count'] / total_counts_cohort

    # Pivot table for heatmap
    heatmap_df = total_grouped.pivot(index='class_label', columns=grouping_column, values='proportion').fillna(0)

    annot=False
    annot_kws={"weight": "extra bold"}
    line_width = 0.5
    dpi=300


     # Determine color scale range if not provided
    if vmin_1 is None:
        # vmin_1 = heatmap_df.min().min()
        
        vmin_1=0
    if vmax_1 is None:
        vmax_1 = heatmap_df.max().max()

    # Reorder columns if a specific order is provided
    if grouping_order:
        heatmap_df = heatmap_df[grouping_order]
        total_counts_per_cohort = total_counts_per_cohort[grouping_order]


    # Calculate the total area for each cohort
    total_area_per_cohort = area_df.groupby(grouping_column)[area_col].sum()

    # Normalize total counts by total area for each cohort
    normalized_counts_per_cohort = total_counts_per_cohort / total_area_per_cohort.reindex(total_counts_per_cohort.index)


    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(heatmap_df, annot=annot, cmap='viridis',square=True,annot_kws=annot_kws,vmin=vmin_1,vmax=vmax_1, linewidths=line_width)
    

    for idx, (total, normalized) in enumerate(zip(total_counts_per_cohort, normalized_counts_per_cohort)):
        ax.text(idx + 0.5, -3, f"{int(total)} ({normalized:.5f})", ha='center', va='center', rotation=45)

    
    ax.set_xlabel('Disease Cohort')
    ax.set_ylabel('Class Label')

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_name:
        plt.savefig(write_dir + f'cell_label_heatmap_{save_name}.tif', bbox_inches='tight', dpi=dpi)
    
    plt.show()
    
    # Calculate proportions across cohorts using patient averages
    total_avg_grouped = patient_avg.groupby(['class_label', grouping_column]).sum().reset_index()
    total_counts_label = total_avg_grouped.groupby('class_label')['normalized_count'].transform('sum')
    total_avg_grouped['proportion_across'] = total_avg_grouped['normalized_count'] / total_counts_label
    
    
    # Apply row order before pivoting for second heatmap
    if row_order:
       if not set(row_order).issubset(set(total_avg_grouped['class_label'])):
           raise ValueError("row_order contains invalid class labels.")
       total_avg_grouped['class_label'] = pd.Categorical(total_avg_grouped['class_label'], categories=row_order, ordered=True)
       total_avg_grouped.sort_values(by='class_label', inplace=True)
    

    # Pivot table for the second heatmap
    heatmap_df_across = total_avg_grouped.pivot_table(index='class_label', columns=grouping_column, values='proportion_across', aggfunc='sum').fillna(0)

    
    # Determine color scale range if not provided
    if vmin_2 is None:
        # vmin_2 = heatmap_df_across.min().min()
        vmin_2=0
    if vmax_2 is None:
        vmax_2 = heatmap_df_across.max().max()


    # Reorder columns if a specific order is provided
    if grouping_order:
        heatmap_df_across = heatmap_df_across[grouping_order]

    # Plot the second heatmap
    plt.figure(figsize=(10, 6))
    ax2 = sns.heatmap(heatmap_df_across, annot=annot, cmap='viridis',square=True,annot_kws=annot_kws,vmin=vmin_2,vmax=vmax_2, linewidths=line_width)
    for idx, (total, normalized) in enumerate(zip(total_counts_per_cohort, normalized_counts_per_cohort)):
        ax2.text(idx + 0.5, -3, f"{int(total)} ({normalized:.5f})", ha='center', va='center', rotation=45)

    
    
    
    ax2.set_xlabel('Disease Cohort')
    ax2.set_ylabel('Class Label')
    # ax2.set_title(f'Cell Class Proportions Heatmap (Across Cohorts)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    if save_name:
        plt.savefig(write_dir + f'cell_label_heatmap_across_{save_name}.tif', bbox_inches='tight', dpi=dpi)
    plt.show()   
    return(total_grouped,patient_avg ) ##for next plot


def read_csvs_replace_column_add_folder_and_append(match_name, directory, new_column_name, scale_factor=44032515):
    all_data = []  # List to store DataFrames

    folders = os.listdir(directory)
    for folder in folders:
        folder_path = os.path.join(directory, folder)
        files = os.listdir(folder_path)
        matching_files = [file for file in files if match_name in file]

        for filename in matching_files:
            csv_path = os.path.join(folder_path, filename)
            df = pd.read_csv(csv_path)
            df['disease_cohort'] = folder 
            df = df.rename(columns={'Value': new_column_name,'Key':'CompName'})  # Replace 'Value' with new_column_name
             # Add column with the folder name
            all_data.append(df)

    # Concatenate all DataFrames into one
    combined_df = pd.concat(all_data, ignore_index=True)
    # Scale the specified column by the given scale factor
    if new_column_name in combined_df.columns:
        combined_df[new_column_name] = combined_df[new_column_name]/scale_factor

    return combined_df


def plot_combined_proportions_heatmap(total_grouped_list, write_dir, grouping_column='disease_cohort', save_name=None, vmin_1=None, vmax_1=None, vmin_2=None, vmax_2=None, grouping_order=None):
    """
    Plots heatmaps showing the combined normalized proportion of cell classes across different disease cohorts.
    This function takes already normalized counts across different spatial regions of interest.

    Parameters:
    total_grouped_list (list of pandas.DataFrame): List of DataFrames containing the normalized counts.
    write_dir (str): Directory to save the output plots.
    """
    # Combine all total_grouped dataframes
    combined_df = pd.concat(total_grouped_list)

    # Calculate overall proportions across all cohorts
    total_counts_cohort = combined_df.groupby(grouping_column)['avg_normalized_count'].transform('sum')
    combined_df['proportion'] = combined_df['avg_normalized_count'] / total_counts_cohort

    # Pivot table for the first heatmap
    heatmap_df = combined_df.pivot_table(index='class_label', columns=grouping_column, values='proportion', aggfunc='sum').fillna(0)

    # Determine color scale range if not provided
    if vmin_1 is None:
        vmin_1 = 0
    if vmax_1 is None:
        vmax_1 = heatmap_df.max().max()

    # Reorder columns if a specific order is provided
    if grouping_order:
        heatmap_df = heatmap_df[grouping_order]

    # Plot the first heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_df, annot=False, cmap='viridis', square=True, vmin=vmin_1, vmax=vmax_1, linewidths=0.5)
    plt.xlabel('Disease Cohort')
    plt.ylabel('Class Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    if save_name:
        plt.savefig(f'{write_dir}combined_proportions_heatmap_{save_name}.tif', bbox_inches='tight', dpi=300)
    plt.show()

    # Additional operations or calculations if needed

    # Return the combined DataFrame for further use or verification
    return combined_df 

def plot_joint_normalized_proportions_heatmaps(total_grouped_list, write_dir, area_names, grouping_column='disease_cohort', save_name=None, vmin=None, vmax=None, grouping_order=None,heatmap_names=None):
    """
    Plots separate heatmaps for each area with proportions normalized across all provided areas.

    Parameters:
    total_grouped_list (list of pandas.DataFrame): List of DataFrames containing the normalized counts for each area.
    write_dir (str): Directory to save the output plots.
    area_names (list of str): Names of the areas corresponding to the DataFrames in total_grouped_list, used in the filenames.
    grouping_column (str): Column name to group by, typically 'disease_cohort'.
    save_name (str): Base name for saved plot files. This will be appended with area names for each plot.
    vmin, vmax (float): Min and max values for the heatmap color scale. Calculated from combined data if not provided.
    grouping_order (list of str): Specific order of cohorts for plotting.
    """
    # Combine all dataframes for joint normalization based on class labels
    combined_df = pd.concat(total_grouped_list)
    
    # Calculate total counts per class label across all areas
    total_counts_label = combined_df.groupby('class_label')['avg_normalized_count'].sum()

    # Use calculated totals to determine proportions for each area
    for i, total_grouped in enumerate(total_grouped_list):
        area_name = area_names[i]
        
        # Normalize counts based on total counts for each class label across all areas
        total_grouped['proportion'] = total_grouped.apply(lambda row: row['avg_normalized_count'] / total_counts_label[row['class_label']], axis=1)

        # Pivot table for heatmap
        heatmap_df = total_grouped.pivot(index='class_label', columns=grouping_column, values='proportion').fillna(0)
        
        if grouping_order:
            heatmap_df = heatmap_df.reindex(columns=grouping_order)
        
        # Determine color scale range if not provided
        if vmin is None:
            vmin = heatmap_df.min().min()
        if vmax is None:
            vmax = heatmap_df.max().max()

        # Plot heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_df, annot=False, cmap='viridis', square=True, vmin=vmin, vmax=vmax, linewidths=0.5)
        plt.xlabel('Disease Cohort')
        plt.ylabel('Class Label')
        plt.title(f'Cell Class Proportions Heatmap for {area_name}')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_name:
            plt.savefig(f'{write_dir}{save_name}_{area_name}.tif', bbox_inches='tight', dpi=300)
            print(f'{write_dir}{save_name}_{area_name}.tif')
        plt.show()
        
        
    # Use calculated totals to determine proportions for each area
    for i, total_grouped in enumerate(total_grouped_list):
        print(i,total_grouped)
    
    # Determine all unique cohorts and class labels
    unique_cohorts = pd.concat(total_grouped_list)['disease_cohort'].unique()
    unique_labels = pd.concat(total_grouped_list)['class_label'].unique()

   # Iterate over each unique cohort
    for cohort in unique_cohorts:
       # Create an empty DataFrame to store proportions for each DataFrame in the list
       cohort_data = pd.DataFrame(index=unique_labels, columns=[f"Iteration {i}" for i in range(len(total_grouped_list))])

       # Populate the DataFrame with proportions
       for i, df in enumerate(total_grouped_list):
           # Filter the DataFrame for the current cohort
           filtered_df = df[df['disease_cohort'] == cohort]

           # Loop over each class label to fill the DataFrame
           for label in unique_labels:
               proportion = filtered_df[filtered_df['class_label'] == label]['proportion'].sum()  # Using sum to aggregate multiple entries if exist
               cohort_data.loc[label, f"Iteration {i}"] = proportion
       print(cohort_data)
       
       cohort_data.columns = heatmap_names
       # Plot the heatmap
       plt.figure(figsize=(12, 8))
       sns.heatmap(cohort_data.astype(float), annot=False, cmap='viridis', square=True, vmin=vmin, vmax=vmax, linewidths=0.5)
       plt.title(f'Heatmap for {cohort}')
       plt.ylabel('Class Label')
       plt.xlabel('Iteration')

       if save_name:
           print(f'{write_dir}{save_name}_{cohort}.png')
           plt.savefig(f'{write_dir}{save_name}_{cohort}.png', bbox_inches='tight', dpi=300)
       plt.show()

    # No need to return anything unless you want to use the dataframes outside this function
def read_csvs_replace_column_add_folder_and_append_combined(match_name, directory, scale_factor=44032515,area_names=None): #1mm^2
    all_data = []  # List to store DataFrames

    folders = os.listdir(directory)
    for folder in folders:
        folder_path = os.path.join(directory, folder)
        files = os.listdir(folder_path)
        matching_files = [file for file in files if match_name in file]

        for filename in matching_files:

            sample_id = filename.replace("_areas.csv","")  # Assuming the sample ID is always before the first underscore

            csv_path = os.path.join(folder_path, filename)
            df = pd.read_csv(csv_path)

            df_wide = df.set_index('Key').T  # Transpose after setting 'Key' as index
            df_wide.columns = df_wide.columns.astype(str)  # Ensure columns are string for renaming

            # Optionally rename area columns
            if area_names:
                # Rename based on provided names, ensuring we have enough names for columns
                rename_dict = {str(i+1): name for i, name in enumerate(area_names)}
                df_wide.rename(columns=rename_dict, inplace=True)

            # Scale all area columns
            for column in df_wide.columns:
                df_wide[column] = df_wide[column].astype(float) / scale_factor  # Ensure scaling as float

            df_wide['CompName'] = sample_id
            df_wide['disease_cohort'] = folder
            all_data.append(df_wide.reset_index(drop=True))
    # Concatenate all DataFrames into one
    combined_df = pd.concat(all_data, ignore_index=True)

    return combined_df 

def plot_cell_class_violinplots_areaNormalized(df,area_df, area_col, cohort, col, write_dir,grouping_column, save_name=None, vmin_1=None, vmax_1=None, vmin_2=None, vmax_2=None, grouping_order=None,cohort_colors=None):
    """
    Plots a heatmap showing the proportion of cell classes based on the 'class_label' column,
    filtered by rows where 'inside_interstitium' is True, and grouped by the 'disease_cohort' column.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the relevant data.
    """

    filtered_df = df[df[col]]

    # Group by 'disease_cohort', 'CompName', 'class_label' and count occurrences
    grouped = filtered_df.groupby([grouping_column, 'CompName', 'class_label','AccessionNumber']).size().reset_index(name='count')


    total_counts_per_cohort = grouped.groupby(grouping_column)['count'].sum()
    # Merge the grouped DataFrame with area_df on 'CompName' and 'disease_cohort'
    merged_grouped = grouped.merge(area_df, on=['CompName', grouping_column], how='left')

    # Normalize counts by area
    merged_grouped['normalized_count'] = merged_grouped['count'] / merged_grouped[area_col]

    # Calculate average counts per patient
    patient_avg = merged_grouped.groupby(['AccessionNumber', grouping_column, 'class_label'])['normalized_count'].mean().reset_index()
    print("patient_avg",patient_avg)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='class_label', y='normalized_count', hue=grouping_column, data=patient_avg, split=False, palette=cohort_colors,hue_order=grouping_order)
    plt.title(f'Patient Averages for Cell Classes by Cohort {area_col}')
    plt.xticks(rotation=90)
    plt.tight_layout()

    if save_name is not None:
        plt.savefig(f"{write_dir}/{save_name}.png")
    plt.show()
    
    
    
def plot_violin_plots_of_density(total_grouped_list, write_dir, area_names, grouping_column='disease_cohort', save_name=None, grouping_order=None, cohort_colors=None,cell_classes=None):
    """
    Plots violin plots for each area with densities of cell classes, retaining cohort grouping.
    
    Parameters:
    - total_grouped_list (list of pandas.DataFrame): List of DataFrames with normalized counts for each area.
    - write_dir (str): Directory to save the output plots.
    - area_names (list of str): Names of the areas corresponding to the DataFrames in total_grouped_list.
    - grouping_column (str): Column name for cohort grouping, typically 'disease_cohort'.
    - save_name (str, optional): Base name for saved plot files, appended with area names.
    - grouping_order (list of str, optional): Specific order of cohorts for plotting.
    - cohort_colors (dict, optional): Dictionary mapping cohorts to colors.
    """
    
    for class_label in cell_classes:
        plt.figure(figsize=(12, 6))
        
        # Create a temporary DataFrame for the current class_label across all areas
        temp_df = pd.DataFrame()
        for i, df in enumerate(total_grouped_list):
            filtered_df = df[df['class_label'] == class_label].copy()
            filtered_df['Area'] = area_names[i]  # Add area information
            temp_df = pd.concat([temp_df, filtered_df])
        
        # Plotting
        sns.violinplot(x='Area', y='normalized_count', hue=grouping_column, data=temp_df,
                       hue_order=grouping_order, palette=cohort_colors, split=False)
        plt.title(f'Violin Plot for {class_label}')
        plt.xticks(rotation=45)
        plt.xlabel('Area')
        plt.ylabel('Density')
        # Adjusting the legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        
        # plt.tight_layout(rect=[0,0,0.85,1])  # Adjust the rect parameter as needed
        
        plt.tight_layout()
        
        # Save the plot
        if save_name:
            full_save_name = f"{write_dir}{save_name}_{class_label}.tiff"
            plt.savefig(full_save_name,dpi=150)
            print(f"Plot saved to {full_save_name}")
        plt.show()


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
    parser.add_argument('-f',
            '--filename',
            type=str,
            # default='cell_patches_1um',
            help=''
            )
    parser.add_argument('-jr',
            '--JOINT_ROOT_DIR',
            type=str,
            # default='cell_patches_1um',
            help=''
            )
    
    args,unparsed = parser.parse_known_args()
    
    
    JOINT_ROOT_DIR = args.JOINT_ROOT_DIR
    
    
    
    #%%
    

#%%
    with open(args.color_dict,'rb') as f:
       colorDict = pickle.load(f)
#%%
    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)

#%%
    df = pd.DataFrame(pd.read_csv(args.cell_expression_dir,index_col=False))
    infoDf = pd.DataFrame(pd.read_csv(args.cell_info_dir,index_col=False))
    
    df = convert_mixed_dtype_to_string(df)
    infoDf =convert_mixed_dtype_to_string(infoDf)
    
    infoDf = infoDf.loc[df.index] ##so they necessarily match



    classLabelsList = np.unique(df['class_label'])
    #%%
    color_mappingTT= {'NK': [0, 1, 0, 1],
 'LuN': [0, 0, 1, 1],
 'MR': [1, 0.8, 0.3, 1],
 'TCMR':[0,1,1,1]}
    
    color_mapping = {'Normal_Kidney':[0,1,0,1],'Lupus_Nephritis':[0,0,1,1], 'Renal_Allograft':[1,0,1,1]}

    #%%
    #1=interstitium
    #2=border of tubules
    #3= border of glomeruli
    #4= tubules
    #5= glomeruli
    infoDfTemp = update_infoDf(infoDf= infoDf, 
                           structure_dir=JOINT_ROOT_DIR+'manual_cell_gating/classified_cell_analysis/combined_compartment_analysis_final/', 
                           col='1_interstitium',
                           ignore_filenames=['inside_mask_areas.csv'],reference_substring='_1.csv')
    
    #%%
    infoDfTemp = update_infoDf(infoDf= infoDfTemp, 
                           structure_dir=JOINT_ROOT_DIR+'manual_cell_gating/classified_cell_analysis/combined_compartment_analysis_final/', 
                           col='2_tubular_border',
                           ignore_filenames=['inside_mask_areas.csv'],reference_substring='_2.csv')
    #%%
    infoDfTemp = update_infoDf(infoDf= infoDfTemp, 
                           structure_dir=JOINT_ROOT_DIR+'manual_cell_gating/classified_cell_analysis/combined_compartment_analysis_final/', 
                           col='3_glomerular_border',
                           ignore_filenames=['inside_mask_areas.csv'],reference_substring='_3.csv')
    
    #%%
    infoDfTemp = update_infoDf(infoDf= infoDfTemp, 
                           structure_dir=JOINT_ROOT_DIR+'manual_cell_gating/classified_cell_analysis/combined_compartment_analysis_final/', 
                           col='4_tubules',
                           ignore_filenames=['inside_mask_areas.csv'],reference_substring='_4.csv')
    #%%
    infoDfTemp = update_infoDf(infoDf= infoDfTemp, 
                           structure_dir=JOINT_ROOT_DIR+'manual_cell_gating/classified_cell_analysis/combined_compartment_analysis_final/', 
                           col='5_glomeruli',
                           ignore_filenames=['inside_mask_areas.csv'],reference_substring='_5.csv')
    
    #%%
    

    #%%
    areas_combined = read_csvs_replace_column_add_folder_and_append_combined(match_name='areas', 
                                                   directory=JOINT_ROOT_DIR+'manual_cell_gating/classified_cell_analysis/combined_compartment_analysis_final/',
                                                   area_names=["1_interstitium_area",'2_tubular_border_area',
                                                               '3_glomerular_border_area','4_tubules_area',
                                                               '5_glomeruli_area'])
    
 
    
    #%%
    infoDfTemp = update_infoDf(infoDf= infoDfTemp, 
                           structure_dir=JOINT_ROOT_DIR+'manual_cell_gating/classified_cell_analysis/MXA_analysis_final/', 
                           col='inside_MXA',
                           reference_substring='_inside')
    #%%
    insideMXAArea = read_csvs_replace_column_add_folder_and_append(match_name='inside_', 
                                                   directory=JOINT_ROOT_DIR+'manual_cell_gating/classified_cell_analysis/MXA_analysis_final/', 
                                                   new_column_name='inside_MXA_area')
  
    #%%
    
    infoDfTemp = update_infoDf(infoDf= infoDfTemp, 
                           structure_dir=JOINT_ROOT_DIR+'manual_cell_gating/classified_cell_analysis/MXA_analysis_final/', 
                           col='border_MXA',
                           reference_substring='_border')
    #%%
    
    
    borderMXAArea = read_csvs_replace_column_add_folder_and_append(match_name='border_', 
                                                   directory=JOINT_ROOT_DIR+'manual_cell_gating/classified_cell_analysis/MXA_analysis_final/', 
                                                   new_column_name='border_MXA_area')
    
    #%%
    infoDfTemp = update_infoDf(infoDf= infoDfTemp, 
                           structure_dir=JOINT_ROOT_DIR+'manual_cell_gating/classified_cell_analysis/MXA_analysis_final/', 
                           col='other_tissue_MXA',
                           reference_substring='_other_tissue')
    
    #%%
    otherMXAArea = read_csvs_replace_column_add_folder_and_append(match_name='other_tissue_', 
                                                   directory=JOINT_ROOT_DIR+'manual_cell_gating/classified_cell_analysis/MXA_analysis_final/', 
                                                   new_column_name='other_tissue_MXA_area')
    #%%
    
    infoDfTemp = update_infoDf(infoDf= infoDfTemp, 
                           structure_dir=JOINT_ROOT_DIR+'manual_cell_gating/classified_cell_analysis/collagen_analysis_final/', 
                           col='inside_collagen',
                           reference_substring='_inside')
    
    #%%
    insideCollagenArea = read_csvs_replace_column_add_folder_and_append(match_name='inside_', 
                                                   directory=JOINT_ROOT_DIR+'manual_cell_gating/classified_cell_analysis/collagen_analysis_final/', 
                                                   new_column_name='inside_collagen_area')
  
    
    #%%
    
    infoDfTemp = update_infoDf(infoDf=  infoDfTemp, 
                           structure_dir=JOINT_ROOT_DIR+'manual_cell_gating/classified_cell_analysis/collagen_analysis_final/', 
                           col='border_collagen',
                           reference_substring='_border')
    
    #%%
    
    borderCollagenArea = read_csvs_replace_column_add_folder_and_append(match_name='border_', 
                                                   directory=JOINT_ROOT_DIR+'manual_cell_gating/classified_cell_analysis/collagen_analysis_final/', 
                                                   new_column_name='border_collagen_area')
  
    
    
    #%%
    infoDfTemp = update_infoDf(infoDf= infoDfTemp, 
                         structure_dir=JOINT_ROOT_DIR+'manual_cell_gating/classified_cell_analysis/collagen_analysis_final/', 
                         col='other_tissue_collagen',
                         reference_substring='_other_tissue')
    
  #%%
    otherCollagenArea = read_csvs_replace_column_add_folder_and_append(match_name='other_tissue_', 
                                                   directory=JOINT_ROOT_DIR+'manual_cell_gating/classified_cell_analysis/collagen_analysis_final/', 
                                                   new_column_name='other_tissue_collagen_area')
  
    
  
    #%%
    infoDfTemp['class_label'] = df['class_label']
    #%%
        # Specify the filename and path where you want to save the CSV
    output_file_path = args.cell_info_dir.replace(".csv",'_compartments.csv')
    
    #%%
    for cohort in np.unique(df['disease_cohort']):
        print('Cohort',cohort)
        # cohort ='Lupus_Nephritis'
        
        
        immuneInfoDf = infoDf[~infoDf['class_label'].isin(['Endothelial_cells','Distal_collecting_tubules',
                                                       'Inflamed_tubule','Proximal_tubules', 'RBCs','other','CD4+CD8+_Tcell'])].copy()
        # immuneInfoDf = infoDf[~infoDf['class_label'].isin(['Endothelial_cells','Distal_collecting_tubules',
        #                                        'Inflamed_tubule','Proximal_tubules', 'RBCs','other','CD4+CD8+_Tcell'])].copy()

        immuneInfoDf['class_label'].replace({'CD4+ICOS+_Tcell':'CD4+_Tcell', 
                                            'CD4+PD1+_Tcell':'CD4+_Tcell',
                                            'CD4+Foxp3+_Tcell':'CD4+_Tcell', 
                                            'CD4+ICOS+PD1+_Tcell':'CD4+_Tcell',
                                            'CD14+CD163+MERTK+_macrophages':'CD14+CD163+_macrophages',
                                            'CD8+Foxp3+_Tcell':'CD8+_Tcell', 
                                            'CD8+ICOS+PD1+_Tcell':'CD8+_Tcell', 
                                            'CD8+ICOS+_Tcell':'CD8+_Tcell',
                                            'CD8+PD1+_Tcell':'CD8+_Tcell'}, inplace=True)
        
        
       
       
        plot_cell_class_proportions_stacked_areaNormalized(df=immuneInfoDf[df['disease_cohort']==cohort],
                                                            area_df=insideMXAArea,
                                                            area_col='inside_MXA_area', 
                                                            colorDict=colorDict, 
                                                            cohort=cohort, 
                                                            col='inside_MXA', 
                                                            write_dir=args.write_dir,
                                                            save_name=f"inside_MXA_immuneOnly_{cohort}_normalized")
       
        plot_cell_class_proportions_stacked_areaNormalized(df=immuneInfoDf[df['disease_cohort']==cohort],
                                                            area_df=borderMXAArea,
                                                            area_col='border_MXA_area', 
                                                            colorDict=colorDict, 
                                                            cohort=cohort, 
                                                            col='border_MXA', 
                                                            write_dir=args.write_dir,
                                                            save_name=f"border_MXA_immuneOnly_{cohort}_normalized")
       
        
        plot_cell_class_proportions_stacked_areaNormalized(df=immuneInfoDf[df['disease_cohort']==cohort],
                                                            area_df=insideCollagenArea,
                                                            area_col='inside_collagen_area', 
                                                            colorDict=colorDict, 
                                                            cohort=cohort, 
                                                            col='inside_collagen', 
                                                            write_dir=args.write_dir,
                                                            save_name=f"inside_collagen_immuneOnly_{cohort}_normalized")
       
    
        
        plot_cell_class_proportions_stacked_areaNormalized(df=immuneInfoDf[df['disease_cohort']==cohort],
                                                            area_df=borderCollagenArea,
                                                            area_col='border_collagen_area', 
                                                            colorDict=colorDict, 
                                                            cohort=cohort, 
                                                            col='border_collagen', 
                                                            write_dir=args.write_dir,
                                                            save_name=f"border_collagen_immuneOnly_{cohort}_normalized")
       
        
    #%%
  
    
    immuneInfoDf = infoDfTemp[~infoDfTemp['class_label'].isin(['Endothelial_cells','Distal_collecting_tubules',
                                                   'Inflamed_tubule','Proximal_tubules', 'RBCs','general_unknown','CD4+CD8+_Tcell'])].copy()
    
    #%%

    
    class_order_plot=['Bcell','Plasmablasts','plasma_cells','CD4+Foxp3+_Tcell',
                      'CD4+ICOS+PD1+_Tcell','CD4+ICOS+_Tcell','CD4+PD1+_Tcell','CD4+_Tcell','CD4+_intraepithelial_Tcell',
                      'CD8+Foxp3+_Tcell','CD8+ICOS+PD1+_Tcell','CD8+ICOS+_Tcell','CD8+PD1+_Tcell',
                      'CD8+_Tcell','CD8+_intraepithelial_Tcell','CD3+_Tcell','CD4+CD8+_Tcell','TCRgd_Tcell','NK_Tcell','NK_cells','CD14+CD163+MERTK+_macrophages',
                     'CD14+CD163+_macrophages','CD14+MERTK+_macrophages','CD14+_macrophages','CD16+_macrophages','HLAII+_Monocytes',
                     'HLAII-_Monocytes','Mo-Macrophage','Neutrophils','BDCA2pCD103p_DCs','cDC1','cDC2','pDC', 'Distal_collecting_tubules',
                     'Endothelial_cells','Inflamed_tubule','Proximal_tubules','RBCs','general_unknown']
    #%%
    
    total_grouped_inside_collagen,averageCounts_inside_collagen = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=insideCollagenArea,
                                                       area_col='inside_collagen_area', 
                                                       cohort=cohort, 
                                                       col='inside_collagen',
                                                       write_dir=args.write_dir,
                                                       save_name=f"inside_collagen_immuneOnlySimplified",
                                                       grouping_column='disease_cohort',
                                                       grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,
                                                       row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    # #%%
    
    #%%
    plot_cell_class_violinplots_areaNormalized(df=immuneInfoDf,
                                                       area_df=insideCollagenArea,
                                                       area_col='inside_collagen_area', 
                                                       cohort=cohort, 
                                                       col='inside_collagen',
                                                       write_dir=args.write_dir,
                                                       save_name=f"inside_collagen_immuneOnlySimplified",
                                                       grouping_column='disease_cohort',
                                                       grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,cohort_colors=color_mapping)
    #,
     #                                                  row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    
    #%%
    
    total_grouped_border_collagen,averageCounts_border_collagen = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=borderCollagenArea,
                                                       area_col='border_collagen_area', 
                                                       cohort=cohort, 
                                                       col='border_collagen',
                                                       write_dir=args.write_dir,
                                                       save_name=f"border_collagen_immuneOnlySimplified",
                                                       grouping_column='disease_cohort',
                                                       grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,
                                                       row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    #%%
    
    total_grouped_other_collagen,averageCounts_other_collagen  = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=otherCollagenArea,
                                                       area_col='other_tissue_collagen_area', 
                                                       cohort=cohort, 
                                                       col='other_tissue_collagen',
                                                       write_dir=args.write_dir,
                                                       save_name=f"other_collagen_immuneOnlySimplified",
                                                       grouping_column='disease_cohort',
                                                       grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    
    #%%
    total_grouped_inside_MXA,averageCounts_inside_MXA = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=insideMXAArea,
                                                       area_col='inside_MXA_area', 
                                                       cohort=cohort, 
                                                       col='inside_MXA',
                                                       write_dir=args.write_dir,
                                                       save_name=f"inside_MXA_immuneOnlySimplified",
                                                       grouping_column='disease_cohort',
                                                       grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    #%%
    total_grouped_border_MXA,averageCounts_border_MXA  = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=borderMXAArea,
                                                       area_col='border_MXA_area', 
                                                       cohort=cohort, 
                                                       col='border_MXA',
                                                       write_dir=args.write_dir,
                                                       save_name=f"border_MXA_immuneOnlySimplified",
                                                       grouping_column='disease_cohort',
                                                       grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    #%%
    total_grouped_other_MXA,averageCounts_other_MXA  = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=otherMXAArea,
                                                       area_col='other_tissue_MXA_area', 
                                                       cohort=cohort, 
                                                       col='other_tissue_MXA',
                                                       write_dir=args.write_dir,
                                                       save_name=f"other_MXA_immuneOnlySimplified",
                                                       grouping_column='disease_cohort',
                                                       grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    
    
    
    
    
    #%%
    
    total_grouped_inside_tubule,averageCounts_inside_tubule = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=areas_combined,
                                                       area_col='4_tubules_area', 
                                                       cohort=cohort, 
                                                       col='4_tubules',
                                                       write_dir=args.write_dir,
                                                       save_name=f"inside_tubule_immuneOnlySimplified",
                                                       grouping_column='disease_cohort',
                                                       grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    #%%
    
    total_grouped_inside_interstitium,averageCounts_inside_interstitium = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=areas_combined,
                                                       area_col='1_interstitium_area', 
                                                       cohort=cohort, 
                                                       col='1_interstitium',
                                                       write_dir=args.write_dir,
                                                       save_name=f"inside_interstitium_immuneOnlySimplified",
                                                       grouping_column='disease_cohort',
                                                       grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    #%%
    total_grouped_inside_glomeruli,averageCounts_inside_glomeruli = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=areas_combined,
                                                       area_col='5_glomeruli_area', 
                                                       cohort=cohort, 
                                                       col='5_glomeruli',
                                                       write_dir=args.write_dir,
                                                       save_name=f"inside_glomeruli_immuneOnlySimplified",
                                                       grouping_column='disease_cohort',
                                                       grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    
    #%%
    total_grouped_border_tubule,averageCounts_border_tubule = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=areas_combined,
                                                       area_col='2_tubular_border_area', 
                                                       cohort=cohort, 
                                                       col='2_tubular_border',
                                                       write_dir=args.write_dir,
                                                       save_name=f"border_tubule_immuneOnlySimplified",
                                                       grouping_column='disease_cohort',
                                                       grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    #%%
    total_grouped_border_glomeruli,averageCounts_border_glomeruli = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=areas_combined,
                                                       area_col='3_glomerular_border_area', 
                                                       cohort=cohort, 
                                                       col='3_glomerular_border',
                                                       write_dir=args.write_dir,
                                                       save_name=f"border_glomeruli_immuneOnlySimplified",
                                                       grouping_column='disease_cohort',
                                                       grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    
    #%%
    plot_violin_plots_of_density(total_grouped_list=[averageCounts_inside_interstitium,averageCounts_inside_tubule,averageCounts_inside_glomeruli,averageCounts_border_tubule,averageCounts_border_glomeruli],
                                 write_dir=args.write_dir+'disease_cohort/', 
                                 area_names=['Interstitium','Tubule','Glomeruli','Border Tubule','Border Glomeruli'], 
                                 grouping_column='disease_cohort', 
                                 save_name='cell_density_violin_compartment',
                                 grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'], 
                                 cohort_colors=color_mapping,cell_classes=['Plasmablasts', 'Mo-Macrophage', 'cDC1', 'CD4+_Tcell', 'NK_cells', 'pDC', 'CD14+CD163+_macrophages', 'plasma_cells', 'HLAII+_Monocytes', 'NK_Tcell', 'TCRgd_Tcell', 'Bcell', 'CD14+_macrophages', 'CD8+_Tcell', 'CD16+_macrophages', 'Neutrophils', 'CD3+_Tcell', 'HLAII-_Monocytes', 'CD14+MERTK+_macrophages', 'cDC2'])
    #%%
    import itertools
    from scipy.stats import mannwhitneyu
   
    #%%
    
    
    #%%
    MWU_results = perform_mwu_tests([averageCounts_inside_interstitium,averageCounts_inside_tubule,averageCounts_inside_glomeruli,averageCounts_border_tubule,averageCounts_border_glomeruli], ['Interstitium','Tubule','Glomeruli','Border Tubule','Border Glomeruli'], 
                                    cell_classes=['Plasmablasts', 'Mo-Macrophage', 'cDC1', 'CD4+_Tcell', 'NK_cells', 'pDC', 'CD14+CD163+_macrophages', 'plasma_cells', 'HLAII+_Monocytes', 'NK_Tcell', 'TCRgd_Tcell', 'Bcell', 'CD14+_macrophages', 'CD8+_Tcell', 'CD16+_macrophages', 'Neutrophils', 'CD3+_Tcell', 'HLAII-_Monocytes', 'CD14+MERTK+_macrophages', 'cDC2'],
                                    cohort_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'])
    
    #%%
    output_file_path = args.write_dir+'disease_cohort/MWU_results_compartment.csv'
    print("output_file_path",output_file_path)
    
    # Save the DataFrame to CSV
    MWU_results.to_csv(output_file_path, index=False)
    
    
    #%%
    
    
    
        
    #%%
    
    STD_results = calculate_std_of_density([averageCounts_inside_interstitium,averageCounts_inside_tubule,averageCounts_inside_glomeruli,averageCounts_border_tubule,averageCounts_border_glomeruli], ['Interstitium','Tubule','Glomeruli','Border Tubule','Border Glomeruli'], 
                                    cell_classes=['Plasmablasts', 'Mo-Macrophage', 'cDC1', 'CD4+_Tcell', 'NK_cells', 'pDC', 'CD14+CD163+_macrophages', 'plasma_cells', 'HLAII+_Monocytes', 'NK_Tcell', 'TCRgd_Tcell', 'Bcell', 'CD14+_macrophages', 'CD8+_Tcell', 'CD16+_macrophages', 'Neutrophils', 'CD3+_Tcell', 'HLAII-_Monocytes', 'CD14+MERTK+_macrophages', 'cDC2'])
    output_file_path = args.write_dir+'disease_cohort/STD_results_compartment.csv'
    
    # Save the DataFrame to CSV
    STD_results.to_csv(output_file_path, index=False)
    
    
    #%%
    
    plot_violin_plots_of_density(total_grouped_list=[averageCounts_inside_MXA,averageCounts_border_MXA,averageCounts_other_MXA,averageCounts_inside_collagen,averageCounts_border_collagen,averageCounts_other_collagen],
                                 write_dir=args.write_dir+'disease_cohort/', 
                                 area_names=['Inside MXA','Border MXA','Other MXA','Inside Collagen','Border Collagen','Other Collagen'], 
                                 grouping_column='disease_cohort', 
                                 save_name='cell_density_violin_compartment_stains',
                                 grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'], 
                                 cohort_colors=color_mapping,cell_classes=['Plasmablasts', 'Mo-Macrophage', 'cDC1', 'CD4+_Tcell', 'NK_cells', 'pDC', 'CD14+CD163+_macrophages', 'plasma_cells', 'HLAII+_Monocytes', 'NK_Tcell', 'TCRgd_Tcell', 'Bcell', 'CD14+_macrophages', 'CD8+_Tcell', 'CD16+_macrophages', 'Neutrophils', 'CD3+_Tcell', 'HLAII-_Monocytes', 'CD14+MERTK+_macrophages', 'cDC2'])
    

    
    #%%
    
    
    total_grouped_border_collagen,averageCounts_border_collagen = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=borderCollagenArea,
                                                       area_col='border_collagen_area', 
                                                       cohort=cohort, 
                                                       col='border_collagen',
                                                       write_dir=args.write_dir,
                                                       save_name=f"border_collagen_immuneOnlySimplified",
                                                       grouping_column='TissueType',
                                                       grouping_order=['NK','LuN','MR','TCMR'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    #%%
    
    total_grouped_other_collagen,averageCounts_other_collagen  = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=otherCollagenArea,
                                                       area_col='other_tissue_collagen_area', 
                                                       cohort=cohort, 
                                                       col='other_tissue_collagen',
                                                       write_dir=args.write_dir,
                                                       save_name=f"other_collagen_immuneOnlySimplified",
                                                       grouping_column='TissueType',
                                                       grouping_order=['NK','LuN','MR','TCMR'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    
    #%%
    total_grouped_inside_MXA,averageCounts_inside_MXA = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=insideMXAArea,
                                                       area_col='inside_MXA_area', 
                                                       cohort=cohort, 
                                                       col='inside_MXA',
                                                       write_dir=args.write_dir,
                                                       save_name=f"inside_MXA_immuneOnlySimplified",
                                                       grouping_column='TissueType',
                                                       grouping_order=['NK','LuN','MR','TCMR'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    #%%
    total_grouped_border_MXA,averageCounts_border_MXA  = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=borderMXAArea,
                                                       area_col='border_MXA_area', 
                                                       cohort=cohort, 
                                                       col='border_MXA',
                                                       write_dir=args.write_dir,
                                                       save_name=f"border_MXA_immuneOnlySimplified",
                                                       grouping_column='TissueType',
                                                       grouping_order=['NK','LuN','MR','TCMR'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70)
    #%%
    total_grouped_other_MXA,averageCounts_other_MXA  = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=otherMXAArea,
                                                       area_col='other_tissue_MXA_area', 
                                                       cohort=cohort, 
                                                       col='other_tissue_MXA',
                                                       write_dir=args.write_dir,
                                                       save_name=f"other_MXA_immuneOnlySimplified",
                                                       grouping_column='TissueType',
                                                       grouping_order=['NK','LuN','MR','TCMR'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    
    
   
    
    
    
    #%%
    
    total_grouped_inside_tubule,averageCounts_inside_tubule = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=areas_combined,
                                                       area_col='4_tubules_area', 
                                                       cohort=cohort, 
                                                       col='4_tubules',
                                                       write_dir=args.write_dir,
                                                       save_name=f"inside_tubule_immuneOnlySimplified",
                                                       grouping_column='TissueType',
                                                       grouping_order=['NK','LuN','MR','TCMR'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    #%%
    
    total_grouped_inside_interstitium,averageCounts_inside_interstitium = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=areas_combined,
                                                       area_col='1_interstitium_area', 
                                                       cohort=cohort, 
                                                       col='1_interstitium',
                                                       write_dir=args.write_dir,
                                                       save_name=f"inside_interstitium_immuneOnlySimplified",
                                                       grouping_column='TissueType',
                                                       grouping_order=['NK','LuN','MR','TCMR'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    #%%
    total_grouped_inside_glomeruli,averageCounts_inside_glomeruli = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=areas_combined,
                                                       area_col='5_glomeruli_area', 
                                                       cohort=cohort, 
                                                       col='5_glomeruli',
                                                       write_dir=args.write_dir,
                                                       save_name=f"inside_glomeruli_immuneOnlySimplified",
                                                       grouping_column='TissueType',
                                                       grouping_order=['NK','LuN','MR','TCMR'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    
    #%%
    total_grouped_border_tubule,averageCounts_border_tubule = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=areas_combined,
                                                       area_col='2_tubular_border_area', 
                                                       cohort=cohort, 
                                                       col='2_tubular_border',
                                                       write_dir=args.write_dir,
                                                       save_name=f"border_tubule_immuneOnlySimplified",
                                                       grouping_column='TissueType',
                                                       grouping_order=['NK','LuN','MR','TCMR'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    #%%
    total_grouped_border_glomeruli,averageCounts_border_glomeruli = plot_cell_class_proportions_heatmap_areaNormalized(df=immuneInfoDf,
                                                       area_df=areas_combined,
                                                       area_col='3_glomerular_border_area', 
                                                       cohort=cohort, 
                                                       col='3_glomerular_border',
                                                       write_dir=args.write_dir,
                                                       save_name=f"border_glomeruli_immuneOnlySimplified",
                                                       grouping_column='TissueType',
                                                       grouping_order=['NK','LuN','MR','TCMR'],
                                                       vmax_1=0.2,
                                                       vmax_2=0.70,row_order=[x for x in class_order_plot if x in immuneInfoDf['class_label'].unique()])
    
    #%%
    plot_violin_plots_of_density(total_grouped_list=[averageCounts_inside_interstitium,averageCounts_inside_tubule,averageCounts_inside_glomeruli,averageCounts_border_tubule,averageCounts_border_glomeruli],
                                 write_dir=args.write_dir+'TissueType/', 
                                 area_names=['Interstitium','Tubule','Glomeruli','Border Tubule','Border Glomeruli'], 
                                 grouping_column='TissueType', 
                                 save_name='cell_density_violin_compartment',
                                 grouping_order=['NK','LuN','MR','TCMR'], 
                                 cohort_colors=color_mappingTT,cell_classes=['Plasmablasts', 'Mo-Macrophage', 'cDC1', 'CD4+_Tcell', 'NK_cells', 'pDC', 'CD14+CD163+_macrophages', 'plasma_cells', 'HLAII+_Monocytes', 'NK_Tcell', 'TCRgd_Tcell', 'Bcell', 'CD14+_macrophages', 'CD8+_Tcell', 'CD16+_macrophages', 'Neutrophils', 'CD3+_Tcell', 'HLAII-_Monocytes', 'CD14+MERTK+_macrophages', 'cDC2'])


    #%%
    plot_violin_plots_of_density(total_grouped_list=[averageCounts_inside_MXA,averageCounts_border_MXA,averageCounts_other_MXA,averageCounts_inside_collagen,averageCounts_border_collagen,averageCounts_other_collagen],
                                 write_dir=args.write_dir+'TissueType/', 
                                 area_names=['Inside MXA','Border MXA','Other MXA','Inside Collagen','Border Collagen','Other Collagen'], 
                                 grouping_column='TissueType', 
                                 save_name='cell_density_violin_compartment',
                                 grouping_order=['NK','LuN','MR','TCMR'], 
                                 cohort_colors=color_mapping,cell_classes=['Plasmablasts', 'Mo-Macrophage', 'cDC1', 'CD4+_Tcell', 'NK_cells', 'pDC', 'CD14+CD163+_macrophages', 'plasma_cells', 'HLAII+_Monocytes', 'NK_Tcell', 'TCRgd_Tcell', 'Bcell', 'CD14+_macrophages', 'CD8+_Tcell', 'CD16+_macrophages', 'Neutrophils', 'CD3+_Tcell', 'HLAII-_Monocytes', 'CD14+MERTK+_macrophages', 'cDC2'])
    
    

    #%%
    plot_joint_normalized_proportions_heatmaps(total_grouped_list=[total_grouped_inside_tubule,total_grouped_inside_interstitium,total_grouped_inside_glomeruli],
                                      write_dir=args.write_dir, 
                                      area_names=["tubule",'interstitium','glomeruli'], 
                                      grouping_column='disease_cohort', 
                                      save_name='immuneOnlySimplified_inside',
                                      vmin=None, 
                                      vmax=0.30, 
                                      grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'],
                                      heatmap_names=["tubule",'interstitium','glomeruli']  )
    #%%
    plot_joint_normalized_proportions_heatmaps(total_grouped_list=[total_grouped_inside_collagen,total_grouped_border_collagen,total_grouped_other_collagen],
                                      write_dir=args.write_dir, 
                                      area_names=["inside collagen",'border collagen','other_collagen'], 
                                      grouping_column='disease_cohort', 
                                      save_name='immuneOnlySimplified_collagen',
                                      vmin=None, 
                                      vmax=0.20, 
                                      grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'],heatmap_names=["inside collagen",'border collagen','other_collagen'])
    
    
    #%%
    plot_joint_normalized_proportions_heatmaps(total_grouped_list=[total_grouped_inside_MXA,total_grouped_border_MXA,total_grouped_other_MXA],
                                      write_dir=args.write_dir, 
                                      area_names=["inside MXA",'border MXA','other_MXA'], 
                                      grouping_column='disease_cohort', 
                                      save_name='immuneOnlySimplified_MXA',
                                      vmin=None, 
                                      vmax=0.20, 
                                      grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'],heatmap_names=["inside MXA",'border MXA','other_MXA'])
    
    #%%
    plot_joint_normalized_proportions_heatmaps(total_grouped_list=[total_grouped_border_tubule,total_grouped_border_glomeruli],
                                      write_dir=args.write_dir, 
                                      area_names=["border tubule",'border glomeruli'], 
                                      grouping_column='disease_cohort', 
                                      save_name='immuneOnlySimplified_inside',
                                      vmin=None, 
                                      vmax=0.30, 
                                      grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'])
    
    #%%
    plot_joint_normalized_proportions_heatmaps(total_grouped_list=[total_grouped_inside_tubule,total_grouped_border_tubule],
                                      write_dir=args.write_dir, 
                                      area_names=["inside tubule",'border tubule'], 
                                      grouping_column='disease_cohort', 
                                      save_name=None,
                                      vmin=None, 
                                      vmax=0.30, 
                                      grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'])
    #%%
    plot_joint_normalized_proportions_heatmaps(total_grouped_list=[total_grouped_inside_glomeruli,total_grouped_border_glomeruli],
                                      write_dir=args.write_dir, 
                                      area_names=["inside glomeruli",'border glomeruli'], 
                                      grouping_column='disease_cohort', 
                                      save_name=None,
                                      vmin=None, 
                                      vmax=0.30, 
                                      grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'])
    #%%
    plot_joint_normalized_proportions_heatmaps(total_grouped_list=[total_grouped_inside_interstitium,total_grouped_inside_tubule,total_grouped_inside_glomeruli,total_grouped_border_tubule,total_grouped_border_glomeruli],
                                      write_dir=args.write_dir, 
                                      area_names=['interstitium',"tubule",'glomeruli',"border tubule",'border glomeruli'], 
                                      grouping_column='disease_cohort', 
                                      save_name='immuneOnlySimplified_all',
                                      vmin=None, 
                                      vmax=0.20, 
                                      grouping_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'],heatmap_names=['interstitium',"tubule",'glomeruli',"border tubule",'border glomeruli'])
    
     #%%
    
    # Add a new column with default value False
    infoDf['inside_interstitium'] = False
    folders = os.listdir(args.structure_dir)
    

    for folder in folders:
        files = os.listdir(args.structure_dir+folder)
        # print(folder,files)
        
        for filename in files:
            csvHold = pd.DataFrame(pd.read_csv(args.structure_dir+folder+'/'+filename,index_col=False))

    
            # Create a temporary DataFrame with only the columns used for matching
            matching_columns = ['CompName', 'disease_cohort', 'TileNum', 'TileCellID']
            temp_df = csvHold[matching_columns].drop_duplicates()
        
            # Update 'inside_interstitium' in infoDf for matching rows
            condition = (infoDf[matching_columns].isin(temp_df.to_dict('list')).all(axis=1))
            infoDf.loc[condition, 'inside_interstitium'] = True

    #%%
    infoDf['class_label'] = df['class_label']
    # filtered_df = infoDf[infoDf['inside_interstitium']]
    #%%
    for cohort in np.unique(df['disease_cohort']):
        print('Cohort',cohort)
        # cohort ='Lupus_Nephritis'
        plot_cell_class_proportions_stacked(infoDf[df['disease_cohort']==cohort],colorDict,cohort)
    
    
    
    
    
if __name__=='__main__':
    main()

end_time = time.perf_counter()
print(f"Time script took to run: {end_time-start_time} seconds ({(end_time-start_time)/60} minutes) .")











