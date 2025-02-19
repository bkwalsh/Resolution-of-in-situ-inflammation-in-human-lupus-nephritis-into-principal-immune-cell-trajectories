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

from dateutil.relativedelta import relativedelta
from statsmodels.sandbox.stats.multicomp import multipletests
import matplotlib.colors as mcolors
from scipy.stats import spearmanr,mannwhitneyu
from scipy import stats
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
        df[col] = df[col].astype(str)

        
    return df
def convert_pvalue_to_asterisks(pvalue):
    print('pvalue',pvalue)
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return " "
def cell_count_area_normalized(grpArea,df,cellClass):
    
    
    exportDict = {}
    for comp in np.unique(df['CompName']):
        dfComp = df[df['CompName']==comp]
        dfComp = dfComp[dfComp['class_label'] == cellClass]
        
        if dfComp.shape[0] == 0:
            exportDict[comp] = 0
        else:
            
            if comp in list(grpArea.keys()): ##some CompName areas do not have all tissue mask

                exportDict[comp] = dfComp.shape[0]/grpArea[comp]

    
    exportDict = pd.DataFrame.from_dict(exportDict,orient='index')
    exportList = []
    
    for acc in np.unique(df['AccessionNumber'].astype(str)):

        comps = np.unique(df[df['AccessionNumber']==acc]['CompName'])
        comps = [x for x in comps if x in list(exportDict.index)]
        

        if len(comps) > 0:
            
            exportList.append(np.mean(exportDict.loc[comps],axis=0)[0]) ###mean for that accession number
    return(exportList)
    

def correlation_heatmap_plot(dfTemp,writeDir,filename,areaDicts):
    count_df = dfTemp.groupby(['AccessionNumber', 'class_label']).size().reset_index(name='Count')
    
    # Pivot the DataFrame so that class_labels become columns and AccessionNumbers become rows
    pivot_df = count_df.pivot(index='AccessionNumber', columns='class_label', values='Count').fillna(0)
    
    Acc_CompName = dfTemp.groupby('AccessionNumber')['CompName'].unique().to_dict()
    # List of unique class labels
    class_labels = pivot_df.columns.tolist()
    
    
    ##lets normalize the counts by area of capture
    
    NKAreaDict,LNAreaDict,ARAreaDict = areaDicts
    for acc in pivot_df.index:
        coh = dfTemp[dfTemp['AccessionNumber']==acc]['disease_cohort'].unique()
        
        tempComps = Acc_CompName[acc] ##all the CompNames associated with this particular patient acc #
        
        if coh == 'Normal_Kidney':
            # print(NKAreaDict)
            # Use list comprehension to get the values
            areas = np.sum([NKAreaDict.get(k) for k in tempComps])
            pivot_df.loc[acc] = pivot_df.loc[acc]/areas

        if coh == 'Lupus_Nephritis':
            areas =np.sum([ LNAreaDict.get(k) for k in tempComps])
            pivot_df.loc[acc] = pivot_df.loc[acc]/areas

        if coh == 'Renal_Allograft':
            areas = np.sum([ARAreaDict.get(k) for k in tempComps])
            pivot_df.loc[acc] = pivot_df.loc[acc]/areas

    # Create an empty DataFrame to store the Spearman correlations
    correlation_matrix = pd.DataFrame(index=class_labels, columns=class_labels)
    heatmapSignificance = pd.DataFrame(index=class_labels,columns=class_labels)

    # Loop through all pairs of class labels to compute Spearman's rank correlation
    for i in range(len(class_labels)):
        for j in range(i, len(class_labels)):
            class1 = class_labels[i]
            class2 = class_labels[j]
            correlation, _ = spearmanr(pivot_df[class1], pivot_df[class2])
            correlation_matrix.loc[class1, class2] = correlation
            correlation_matrix.loc[class2, class1] = correlation
    
    ###this is to do proportion
    correlation_matrix = pd.DataFrame(index=class_labels, columns=class_labels)
    p_value_matrix = pd.DataFrame(index=class_labels, columns=class_labels)

    
    # Loop through all pairs of class labels to compute Spearman's rank correlation
    for i in range(len(class_labels)):
        for j in range(i, len(class_labels)):
            class1 = class_labels[i]
            class2 = class_labels[j]
            correlation, p_value = spearmanr(pivot_df[class1]/pivot_df.sum(axis=1), pivot_df[class2]/pivot_df.sum(axis=1))
            correlation_matrix.loc[class1, class2] = correlation
            correlation_matrix.loc[class2, class1] = correlation
            p_value_matrix.loc[class1, class2] = p_value
            p_value_matrix.loc[class2, class1] = p_value
    
    # Plotting the heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(correlation_matrix.astype(float), annot=True, fmt=".1f", cmap='coolwarm', cbar=True, square=True, annot_kws={"weight": "extra bold"},center=0, vmin=-1, vmax=1)
    plt.title('Spearman Rank Correlation Heatmap between Classes Proportions')
    
    # Make tick labels bold
    plt.xticks(weight='extra bold')
    plt.yticks(weight='extra bold')
    plt.tight_layout()
    plt.savefig(writeDir+filename.replace('.tif','_proportion.tif'),dpi=300)

    # Plotting the p-values heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(p_value_matrix.astype(float), annot=False, fmt=".3g", cmap='coolwarm', cbar=True, square=True, annot_kws={"weight": "extra bold"})
    plt.title('P-values for Spearman Rank Correlation between Classes')
        
    # Make tick labels bold
    plt.xticks(weight='extra bold')
    plt.yticks(weight='extra bold')
    plt.tight_layout()
    plt.savefig(writeDir+filename.replace('.tif','_pvalues.tif'),dpi=300)
    
    # Plotting the heatmap
    plt.figure(figsize=(16, 14))
    # sns.heatmap(correlation_matrix.astype(float), annot=True, fmt=".1f", cmap='coolwarm', cbar=True, square=True,center=0,vmin=-0.15,vmax=1, annot_kws={"weight": "bold"})
    sns.heatmap(correlation_matrix.astype(float), annot=True, fmt=".1f", cmap='coolwarm', cbar=True, square=True, annot_kws={"weight": "extra bold"},center=0, vmin=-1, vmax=1)
    plt.title('Spearman Rank Correlation Heatmap between Classes')
    
    # Make tick labels bold
    plt.xticks(weight='extra bold')
    plt.yticks(weight='extra bold')
    plt.tight_layout()
    plt.savefig(writeDir+filename,dpi=300)
    
        
        
def T_Humoral_myeloid_plots(dfTemp,writeDir,X,Y,Z,markers,filename):    
    # Count the occurrences of each (Patient_ID, Class_Label) pair
    count_df = dfTemp.groupby(['AccessionNumber', 'class_label']).size().reset_index(name='Count')
    
    # Calculate the sum of 'Count' for each 'Patient_ID' to get the total number of occurrences for each patient
    total_count =  dfTemp.groupby('AccessionNumber').size().reset_index(name='Total_Count')
    
    # Merge to attach the total_count to each original entry
    count_df = pd.merge(count_df, total_count, on='AccessionNumber')
    
    # Calculate the proportion
    count_df['Proportion'] = count_df['Count'] / count_df['Total_Count']
    
   # Pivot table to get class_labels as columns
    pivot_df = count_df.pivot_table(index='AccessionNumber', columns='class_label', values='Proportion', aggfunc='first').reset_index()
    
    # Merge with the original dfTemp to get the disease_cohort for each AccessionNumber
    merged_df = pd.merge(pivot_df, dfTemp[['AccessionNumber', 'disease_cohort']].drop_duplicates(), on='AccessionNumber')

    # If they are different, replace them accordingly
    color_intensity = pivot_df[Z]
    
    # Normalize the color_intensity values for better coloring
    norm = mcolors.Normalize(vmin=color_intensity.min(), vmax=color_intensity.max())

     # Plot
    plt.figure(figsize=(10, 6))
    for cohort, marker in markers.items():
        subset = merged_df[merged_df['disease_cohort'] == cohort]
        plt.scatter(subset[X], subset[Y], c=subset[Z], cmap='Blues', edgecolors='k', 
                    marker=marker, label=cohort, norm=norm,s=50)


    
    plt.xlabel(f'Proportion of {X} cells', weight='extra bold', fontsize=12)
    plt.ylabel(f'Proportion of {Y} cells', weight='extra bold', fontsize=12)
    plt.title(f'Scatter plot with color representing proportion of {Z} cells', weight='extra bold', fontsize=12)
    plt.grid(True)
    plt.colorbar(label=f'Proportion of {Z} cells')
    plt.style.use('default')
    plt.legend(title='Disease Cohort')
    plt.tight_layout()
    
    # Save the plot if you need to
    plt.savefig(writeDir + filename,dpi=300)
    plt.show()
    plt.close()
    
    
def correlation_clinical_heatmap_plot(dfTemp,writeDir,areaDicts,dfClinical,cohort,factorCols,clinColDrop,proportions,saveFolder):
    

    if not os.path.exists(writeDir+saveFolder):
        os.mkdir(writeDir+saveFolder)
    
    
    dfTemp = dfTemp[dfTemp['disease_cohort']==cohort]
    
    # Find elements in df1 not in df2
    diff_elements = set(dfTemp['AccessionNumber']) - set(dfClinical.index)
    
    # Convert set to list
    diff_elements_list = list(diff_elements)
    
    # Find shared elements between dfTemp['AccessionNumber'] and dfClinical.index
    shared_elements = set(dfTemp['AccessionNumber']) & set(dfClinical.index)
    
    # Convert set to list
    shared_elements_list = list(shared_elements)

    count_df = dfTemp.groupby(['AccessionNumber', 'class_label']).size().reset_index(name='Count')
    
    # Pivot the DataFrame so that class_labels become columns and AccessionNumbers become rows
    pivot_df = count_df.pivot(index='AccessionNumber', columns='class_label', values='Count').fillna(0)
    
    Acc_CompName = dfTemp.groupby('AccessionNumber')['CompName'].unique().to_dict()
    # List of unique class labels
    class_labels = pivot_df.columns.tolist()
    pivot_df =  pivot_df.loc[ shared_elements_list] #only the shared samples
   

    ##lets normalize the counts by area of capture
    
    NKAreaDict,LNAreaDict,ARAreaDict = areaDicts
    for acc in pivot_df.index:
        coh = dfTemp[dfTemp['AccessionNumber']==acc]['disease_cohort'].unique()

        tempComps = Acc_CompName[acc] ##all the CompNames associated with this particular patient acc #

        
        if coh == 'Normal_Kidney':

            areas = np.sum([NKAreaDict.get(k) for k in tempComps])
            pivot_df.loc[acc] = pivot_df.loc[acc]/areas
            
        if coh == 'Lupus_Nephritis':
            
            areas =np.sum([ LNAreaDict.get(k) for k in tempComps])
            pivot_df.loc[acc] = pivot_df.loc[acc]/areas

            
        if coh == 'Renal_Allograft':
            areas = np.sum([ARAreaDict.get(k) for k in tempComps])
            pivot_df.loc[acc] = pivot_df.loc[acc]/areas

    if len(factorCols)>0:
    
                # Create the dummy variables
        dummies = pd.get_dummies(dfClinical[factorCols], prefix=factorCols)
        
        # Mark dummy columns as "Unknown" for rows with original value as "Unknown"
        for col in factorCols:
            mask = dfClinical[col] == 'Unknown'
            for dummy_col in dummies.columns:
                if col in dummy_col:  # Check if dummy column is related to the original column
                    dummies.loc[mask, dummy_col] = 'Unknown'
        
        # Drop the dummy column for "Unknown"
        dummies = dummies.drop(columns=[col + '_Unknown' for col in factorCols if col + '_Unknown' in dummies.columns])
        
        # Concatenate and drop original columns
        dfClinical = pd.concat([dfClinical, dummies], axis=1)
        dfClinical.drop(factorCols, axis=1, inplace=True)
                
        
    if len(clinColDrop)>0:

        dfClinical.drop(clinColDrop, axis=1, inplace=True)

    dfClinical = dfClinical.loc[pivot_df.index] ##so we only use data we need, also to align
    correlation_matrix = pd.DataFrame(index=class_labels, columns=dfClinical.columns)
    # heatmapSignificance= pd.DataFrame(index=class_labels, columns=dfClinical.columns)
    
    p_value_matrix = pd.DataFrame(index=class_labels, columns=dfClinical.columns)

    
    # Loop through all pairs of class labels to compute Spearman's rank correlation
    for i in range(len(pivot_df.columns)):
        for j in range(len(dfClinical.columns)):
            class1 = sorted(pivot_df.columns)[i]
            class2 = sorted(dfClinical.columns)[j]
            
            dfClinicalTemp = dfClinical.copy()
            dfClinicalTemp = dfClinicalTemp[~dfClinicalTemp[class2].isin(['NoComment'])]##removes it
            dfClinicalTemp = dfClinicalTemp[~dfClinicalTemp[class2].isin(['Unknown'])]##removes it
            
            dfClinicalTemp[class2] = pd.to_numeric(dfClinicalTemp[class2], errors='coerce')

            pivot_df_temp = pivot_df.copy()
            pivot_df_temp = pivot_df_temp.loc[dfClinicalTemp.index]
            
            correlation, p_value = spearmanr(pivot_df_temp[class1]/pivot_df_temp.sum(axis=1), dfClinicalTemp[class2])
            
            if proportions:
                x =  pivot_df_temp[class1]/pivot_df_temp.sum(axis=1)
            else:
                x = pivot_df_temp[class1]
            y = dfClinicalTemp[class2]
            

            # Fit the line and get statistics
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Create a line using the slope and intercept
            line = slope * np.array(x) + intercept

            
            # Create scatter plot
            plt.figure(figsize=(8, 8))
            plt.scatter(x, y)
            plt.plot(x, line, color='red', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')

            # Adding title, labels, and statistics
            plt.title(f'R^2 = {r_value**2:.2f}, p = {p_value:.4f}')

            plt.legend()

            # Adding title and labels
            plt.xlabel(class1)
            plt.ylabel(class2)
            
            # Show plot
            plt.savefig(writeDir+saveFolder+f'correlation_{class1}_{class2}.tif',dpi=200)
            plt.close()
            correlation_matrix.loc[class1, class2] = correlation
            # correlation_matrix.loc[class2, class1] = correlation
            p_value_matrix.loc[class1, class2] = p_value
            
    
    # Plotting the heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(correlation_matrix.astype(float), annot=True, fmt=".1f", cmap='coolwarm', cbar=True, square=True, annot_kws={"weight": "extra bold"},center=0, vmin=-1, vmax=1)
    plt.title('Spearman Rank Correlation Heatmap between Classes Proportions')
    
    # Make tick labels bold
    plt.xticks(weight='extra bold',fontsize=12)
    plt.yticks(weight='extra bold',fontsize=12)
    plt.tight_layout()
    plt.savefig(writeDir+saveFolder+f'heatmap_of_correlation_coefficients.tif',dpi=300)
    plt.show()
    
    
    # Create a binary colormap (only blue and white)
    cmap = mcolors.ListedColormap(['blue', 'white'])
    
    # Set the boundary for the colors. We want blue for p-values <= 0.05 and white for p-values > 0.05
    bounds = [0, 0.05, 1]
    
    # Create the norm to color the heatmap
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    
    # Plotting the p-values heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(p_value_matrix.astype(float), annot=False, fmt=".3g", cmap=cmap,norm=norm, cbar=True, square=True, annot_kws={"weight": "extra bold"})
    plt.title('P-values for Spearman Rank Correlation between Classes uncorrected')
        
    # Make tick labels bold
    plt.xticks(weight='extra bold',fontsize=12)
    plt.yticks(weight='extra bold',fontsize=12)
    plt.tight_layout()
    plt.savefig(writeDir+saveFolder+f'heatmap_of_correlation_pvals_uncorrected.tif',dpi=300)
    plt.show()

    
    # Create a new DataFrame to store the corrected p-values
    p_value_matrix_corrected = pd.DataFrame(index=p_value_matrix.index, columns=p_value_matrix.columns)
    
    # Iterate over each column and apply FDR correction
    for column in p_value_matrix.columns:
        p_values = p_value_matrix[column].dropna().values  # Get non-NaN p-values from the column
        
        # Apply FDR correction
        _, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        
        # Store corrected p-values back in the corresponding positions of the new DataFrame
        p_value_matrix_corrected.loc[p_value_matrix[column].notna(), column] = pvals_corrected


    linewidths = 3
    plt.figure(figsize=(16, 14))
    ax = sns.heatmap(p_value_matrix_corrected.astype(float), annot=False, fmt=".3g", cmap=cmap,norm=norm, cbar=True, square=True, annot_kws={"weight": "extra bold"}, linewidths=linewidths,linecolor='black')
    plt.title('P-values for Spearman Rank Correlation between Classes corrected')
        
    # Make tick labels bold
    plt.xticks(weight='extra bold',fontsize=12)
    plt.yticks(weight='extra bold',fontsize=12)
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
    plt.tight_layout()
    # plt.savefig(writeDir+filename.replace('.tif','_pvalues.tif'),dpi=300)
    plt.savefig(writeDir+saveFolder+f'heatmap_of_correlation_pvals_corrected.tif',dpi=300)
    plt.show()
    
    # Plotting the heatmap
    plt.figure(figsize=(16, 14))
    ax = sns.heatmap(correlation_matrix.astype(float), annot=p_value_matrix_corrected.astype(float).applymap(convert_pvalue_to_asterisks), 
                fmt="", cmap='coolwarm', cbar=True, square=True, annot_kws={"weight": "extra bold",'size':22},center=0, vmin=-1, vmax=1, linewidths=linewidths,linecolor='black')
    plt.title('Spearman Rank Correlation Heatmap between Classes')
    
    # Make tick labels bold
    plt.xticks(weight='extra bold',fontsize=12)
    plt.yticks(weight='extra bold',fontsize=12)
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
    plt.tight_layout()
    plt.savefig(writeDir+saveFolder+f'heatmap_of_correlation_significance.tif',dpi=300)
    plt.show()
    
    
    
def calculate_age(birth, biopsy_date):
    if birth == 'Unknown' or biopsy_date == 'Unknown':
        return 'Unknown'
    
    delta = relativedelta(pd.to_datetime(biopsy_date), pd.to_datetime(birth))
    return delta.years
   
def clinical_feature_summary(df,features):
    
    summaryDict = {}
    for feat in features:

        tempDf = df[~df[feat].isin(['Unknown'])]
        
        summaryDict[f'{feat}_mean'] = tempDf[feat].astype(float).mean()
        summaryDict[f'{feat}_std'] = tempDf[feat].astype(float).std()
        summaryDict[f'{feat}_N'] = len(tempDf[feat].values)
        
        # Check if 'Unknown' exists in the value counts, count it if it does
        if 'Unknown' in df[feat].value_counts():
            summaryDict[f'{feat}_UnknownN'] = df[feat].value_counts()['Unknown']
        else:
            summaryDict[f'{feat}_UnknownN'] = 0


    for key, value in summaryDict.items():
        print(f"Key: {key}, Value: {value}")
        
        
def age_summarize(dfClinical,col,saveName):       
    
    
    ageDict = {}

    # Filter the DataFrame to remove 'Unknown' values in the specified column
    non_unknown_df = dfClinical[dfClinical[col] != 'Unknown']

    # Add the number of non-'Unknown' entries as a feature
    ageDict['non_unknown_count'] = len(non_unknown_df)

    # Compute the statistics only on non-'Unknown' entries
    ageDict['mean_age'] = non_unknown_df[col].mean()
    ageDict['std_age'] = non_unknown_df[col].std()
    ageDict['min_age'] = non_unknown_df[col].min()
    ageDict['max_age'] = non_unknown_df[col].max()

    # Convert the ageDict to a DataFrame and save to a CSV file
    pd.DataFrame([ageDict]).to_csv(saveName, index=False)


def clinical_feature_comparison(df1, df2, features):
    summaryDict = {}
    for feat in features:

        # Filter out 'Unknown' values from both dataframes
        tempDf1 = df1[~df1[feat].isin(['Unknown'])]
        tempDf2 = df2[~df2[feat].isin(['Unknown'])]

        # Calculate statistics for each dataframe
        summaryDict[f'{feat}_mean_df1'] = tempDf1[feat].astype(float).mean()
        summaryDict[f'{feat}_std_df1'] = tempDf1[feat].astype(float).std()
        summaryDict[f'{feat}_N_df1'] = len(tempDf1[feat].values)

        summaryDict[f'{feat}_mean_df2'] = tempDf2[feat].astype(float).mean()
        summaryDict[f'{feat}_std_df2'] = tempDf2[feat].astype(float).std()
        summaryDict[f'{feat}_N_df2'] = len(tempDf2[feat].values)

        # Mann-Whitney U test for the two groups
        if len(tempDf1[feat]) > 0 and len(tempDf2[feat]) > 0:
            u_stat, p_val = mannwhitneyu(tempDf1[feat].astype(float), tempDf2[feat].astype(float))
            summaryDict[f'{feat}_MWU_stat'] = u_stat
            summaryDict[f'{feat}_MWU_p_value'] = p_val
        else:
            summaryDict[f'{feat}_MWU_stat'] = 'Not enough data'
            summaryDict[f'{feat}_MWU_p_value'] = 'Not enough data'


    # Print summary dictionary
    for key, value in summaryDict.items():
        if 'p_value' in key:
            print(f"Key: {key}, Value: {value}")
            
    print('##' * 50)
    for key, value in summaryDict.items():
        print(f"Key: {key}, Value: {value}")

    
    
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
    parser.add_argument('-d',
            '--cell_info',
            type=str,
            default='',
            help=''
            )
    parser.add_argument('-i',
            '--area_dict',
            type=str,
            default='',
            help=''
            )
  

    args,unparsed = parser.parse_known_args()

    
    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)
    
    
    df = convert_mixed_dtype_to_string(pd.DataFrame(pd.read_csv(args.cell_expression_dir,index_col=False)))
    dfInfo = convert_mixed_dtype_to_string(pd.DataFrame(pd.read_csv(args.cell_info,index_col=False)))
    dfInfo =dfInfo.loc[df.index]
    df['CompName'] = dfInfo['CompName']

    df = df[~df['class_label'].isin(['other'])]##removes it
    
 
    #%%
    dfClinicalLN = pd.DataFrame(pd.read_csv(args.clinical_dir+'LN_clinical_data.csv',index_col=False)).rename(columns={'Accession': 'AccessionNumber'}).set_index('AccessionNumber')
    dfClinicalRA = pd.DataFrame(pd.read_csv(args.clinical_dir+'RA_clinical_data.csv',index_col=False)).rename(columns={'Accession': 'AccessionNumber'}).set_index('AccessionNumber')
    dfClinicalNK = pd.DataFrame(pd.read_csv(args.clinical_dir+'NK_clinical_data.csv',index_col=False)).rename(columns={'Accession': 'AccessionNumber'}).set_index('AccessionNumber')
    
    
    accToUse =  df['AccessionNumber'].unique()
    dfClinicalLN = dfClinicalLN[dfClinicalLN.index.isin(accToUse)]
    dfClinicalRA = dfClinicalRA[dfClinicalRA.index.isin(accToUse)]
    dfClinicalNK = dfClinicalNK[dfClinicalNK.index.isin(accToUse)]
    
    #%%
    #how many accession numbers are there per cohort?

    for cohort in df['disease_cohort'].unique():

        acc = df[df['disease_cohort']==cohort]['AccessionNumber'].unique()

 
    
    clinical_feature_summary(dfClinicalLN,features=['Serum_Cr_bx','Serum_Cr_most_recent','C3','C4','CI_GTI', 'CI_G', 'CI_TI', 'TF', 'TA', 'TI'])
    
    clinical_feature_summary(dfClinicalRA,features=['PostTx_baseline_Cr', 'Bx_Cr','Tacrolimus_levels','CI_GTI', 'CI_G', 'CI_TI', 'TF', 'TA', 'TI'])
    
    
    #%%
    clinical_feature_comparison(dfClinicalLN, dfClinicalRA, features=['CI_GTI', 'CI_G', 'CI_TI', 'TF', 'TA', 'TI'])
    

    
    dfClinicalRA['age_at_biopsy'] = dfClinicalRA.apply(lambda row: calculate_age(row['DOB'], row['Bx_date']), axis=1)
    age_summarize(dfClinicalRA,'age_at_biopsy',args.write_dir+'RA_age_at_biopsy_stats.csv')

    
    #%%
    dfClinicalRA['age_at_transplant'] = dfClinicalRA.apply(lambda row: calculate_age(row['DOB'], row['Date_Tx_clean']), axis=1)
    age_summarize(dfClinicalRA,'age_at_transplant',args.write_dir+'RA_age_at_transplant_stats.csv')

    
    dfClinicalRA['bx_time_since_transplantation'] = dfClinicalRA.apply(lambda row: calculate_age(row['Date_Tx_clean'], row['Bx_date']), axis=1)
    age_summarize(dfClinicalRA,'bx_time_since_transplantation',args.write_dir+'RA_bx_time_since_transplantation_stats.csv')

    

    
    dfClinicalLN['age_at_biopsy'] = dfClinicalLN.apply(lambda row: calculate_age(row['DOB'], row['Bx_date']), axis=1)
    age_summarize(dfClinicalLN,'age_at_biopsy',args.write_dir+'LN_age_at_biopsy_stats.csv')
    
    #%%
    
    dfClinicalLN['age_at_diagnosis'] = dfClinicalLN.apply(lambda row: calculate_age(row['DOB'], row['date_sle_dx_clean']), axis=1)
    age_summarize(dfClinicalLN,'age_at_diagnosis',args.write_dir+'LN_age_at_diagnosis_stats.csv')
    
    
    
    #%%
    dfClinicalLN['time_since_diagnosis'] = dfClinicalLN.apply(lambda row: calculate_age(row['date_sle_dx_clean'], row['Bx_date']), axis=1)
    age_summarize(dfClinicalLN,'time_since_diagnosis',args.write_dir+'LN_time_since_diagnosi_stats.csv')
    
    
    dfClinicalNK['age_at_biopsy'] = dfClinicalNK.apply(lambda row: calculate_age(row['DOB'], row['Bx_date']), axis=1)
    age_summarize(dfClinicalNK,'age_at_biopsy',args.write_dir+'NK_age_at_biopsy_stats.csv')
    
    
    #%%
    colsToUse = [ 'Race', 'Ethnicity', 'Sex', 'DOB',
           'Biopsy indication', 'Bx_date', 'Hx_CKD', 'Cr_bx', 'Notes',
           'age_at_biopsy']
    for feat in colsToUse:
        print('Feature',feat)
        print(dfClinicalNK[feat].value_counts())
    
    
    #%%
    colsToUse = ['Race','Sex','Hypertension_bx','ESRD_status','dsDNA','preBxPred>20',
    'preBxPulseSteroids', 'preBxMMF', 'preBxRituximab', 'LN_class']
    for feat in colsToUse:
        print('Feature',feat)
        print(dfClinicalLN[feat].value_counts())
    #%%
    print(dfClinicalRA.columns)
    colsToUse = ['Race','Sex','HTN','TissueType','Tx_type','Acute_rejection_type','DM','Medication_compliant']
    for feat in colsToUse:
        print('Feature',feat)
        print(dfClinicalRA[feat].value_counts())
    
    #%%
    

    
    
    #%%
    

if __name__=='__main__':
    main()

end_time = time.perf_counter()
print(f"Time script took to run: {end_time-start_time} seconds ({(end_time-start_time)/60} minutes) .")











