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
import matplotlib.colors as mcolors
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from math import pi

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
    
def cell_count_area_normalized(grpArea,df,cellClass):
    
    ##lets make a dictionary with the area by composite
   
    exportDict = {}
    for comp in np.unique(df['CompName']):

        dfComp = df[df['CompName']==comp]

        dfComp = dfComp[dfComp['class_label'] == cellClass]
        
        if dfComp.shape[0] == 0:

            exportDict[comp] = 0
        else:
            
            if comp in list(grpArea.keys()): ##some CompName areas do not have all tissue mask
            
    
                exportDict[comp] = dfComp.shape[0]/grpArea[comp]
        print("_"*20)
    
    exportDict = pd.DataFrame.from_dict(exportDict,orient='index')
   
    exportList = []
    

    for acc in np.unique(df['AccessionNumber'].astype(str)):
        # print(exportDict)
        comps = np.unique(df[df['AccessionNumber']==acc]['CompName'])
        comps = [x for x in comps if x in list(exportDict.index)]

        
        if len(comps) > 0:

            exportList.append(np.mean(exportDict.loc[comps],axis=0)[0]) ###mean for that accession number
            print("*"*20)

    return(exportList)
    

def correlation_heatmap_plot(dfTemp,writeDir,filename,class_labels,proportion_df,count_df):
    
    # class_labels = dfTemp['class_label'].value_counts().index
    correlation_matrix = pd.DataFrame(index=class_labels, columns=class_labels)
    p_value_matrix = pd.DataFrame(index=class_labels, columns=class_labels)

    # Loop through all pairs of class labels to compute Spearman's rank correlation
    for i in range(len(class_labels)):
        for j in range(i, len(class_labels)):
            class1 = class_labels[i]
            class2 = class_labels[j]
           
            correlation, p_value = spearmanr(count_df[class1], count_df[class2])
            
            correlation_matrix.loc[class1, class2] = correlation
            correlation_matrix.loc[class2, class1] = correlation
            p_value_matrix.loc[class1, class2] = p_value
            p_value_matrix.loc[class2, class1] = p_value
    
    
    densityCor= correlation_matrix
    # Plotting the heatmap
    plt.figure(figsize=(16, 14))
    # sns.heatmap(correlation_matrix.astype(float), annot=True, fmt=".1f", cmap='coolwarm', cbar=True, square=True,center=0,vmin=-0.15,vmax=1, annot_kws={"weight": "bold"})
    sns.heatmap(correlation_matrix.astype(float), annot=False, fmt=".1f", cmap='coolwarm', cbar=True, square=True, annot_kws={"weight": "extra bold"},center=0, vmin=-1, vmax=1)
    plt.title(filename.replace('.png','_density.png'))
    
    # Make tick labels bold
    fontsize=14
    plt.xticks(weight='extra bold',fontsize=fontsize)
    plt.yticks(weight='extra bold',fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(writeDir+filename.replace('.png','_density.png'),dpi=300)
    
    
    # Create a binary colormap (only blue and white)
    cmap = mcolors.ListedColormap(['blue', 'white'])
    
    # Set the boundary for the colors. We want blue for p-values <= 0.05 and white for p-values > 0.05
    bounds = [0, 0.05, 1]
    
    # Create the norm to color the heatmap
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    
    # Create a new DataFrame to store the corrected p-values
    p_value_matrix_corrected = pd.DataFrame(index=p_value_matrix.index, columns=p_value_matrix.columns)
    
    # Iterate over each column and apply FDR correction
    for column in sorted(p_value_matrix.columns):
        
        
        p_values = p_value_matrix[column].dropna().values  # Get non-NaN p-values from the column
        # Apply FDR correction
        _, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        
        # Store corrected p-values back in the corresponding positions of the new DataFrame
        p_value_matrix_corrected.loc[p_value_matrix[column].notna(), column] = pvals_corrected


    densityPcorrected=p_value_matrix_corrected
    p_value_matrix_density=p_value_matrix
     # Plotting the p-values heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(p_value_matrix.astype(float), annot=False, fmt=".3g", cmap=cmap,norm=norm, cbar=True, square=True, annot_kws={"weight": "extra bold"})
    plt.title(filename.replace('.png','_density_pvalues_uncorrected.png'))
         
     # Make tick labels bold
    plt.xticks(weight='extra bold',fontsize=fontsize)
    plt.yticks(weight='extra bold',fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(writeDir+filename.replace('.png','_density_pvalues_uncorrected.png'),dpi=300)
     
     
     # Plotting the p-values heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(p_value_matrix_corrected.astype(float), annot=False, fmt=".3g", cmap=cmap,norm=norm, cbar=True, square=True, annot_kws={"weight": "extra bold"})
    plt.title(filename.replace('.png','_density_pvalues_corrected.png'))
         
     # Make tick labels bold
    plt.xticks(weight='extra bold')
    plt.yticks(weight='extra bold')
    plt.tight_layout()
    plt.savefig(writeDir+filename.replace('.png','_density_pvalues_corrected.png'),dpi=300)

    
    correlation_matrix = pd.DataFrame(index=class_labels, columns=class_labels)
    p_value_matrix = pd.DataFrame(index=class_labels, columns=class_labels)

    
    # Loop through all pairs of class labels to compute Spearman's rank correlation
    for i in range(len(class_labels)):
        for j in range(i, len(class_labels)):
            class1 = class_labels[i]
            class2 = class_labels[j]
            # print(pivot_df[class1])
            # print(werwerwe)
            correlation, p_value = spearmanr(proportion_df[class1], proportion_df[class2])
            correlation_matrix.loc[class1, class2] = correlation
            correlation_matrix.loc[class2, class1] = correlation
            p_value_matrix.loc[class1, class2] = p_value
            p_value_matrix.loc[class2, class1] = p_value
    
    
    
    # Create a new DataFrame to store the corrected p-values
    p_value_matrix_corrected = pd.DataFrame(index=p_value_matrix.index, columns=p_value_matrix.columns)
    
    # Iterate over each column and apply FDR correction
    for column in p_value_matrix.columns:
        p_values = p_value_matrix[column].dropna().values  # Get non-NaN p-values from the column
        
        # Apply FDR correction
        _, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        
        # Store corrected p-values back in the corresponding positions of the new DataFrame
        p_value_matrix_corrected.loc[p_value_matrix[column].notna(), column] = pvals_corrected


    proportionCor= correlation_matrix
    proportionPcorrected=p_value_matrix_corrected
    p_value_matrix_proportion = p_value_matrix
    # Plotting the heatmap
    plt.figure(figsize=(16, 14))
    # sns.heatmap(correlation_matrix.astype(float), annot=True, fmt=".1f", cmap='coolwarm', cbar=True, square=True,center=0,vmin=-0.15,vmax=1, annot_kws={"weight": "bold"})
    sns.heatmap(correlation_matrix.astype(float), annot=False, fmt=".1f", cmap='coolwarm', cbar=True, square=True, annot_kws={"weight": "extra bold"},center=0, vmin=-1, vmax=1)
    plt.title(filename.replace('.png','_proportion.png'))
    
    # Make tick labels bold
    plt.xticks(weight='extra bold',fontsize=fontsize)
    plt.yticks(weight='extra bold',fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(writeDir+filename.replace('.png','_proportion.png'),dpi=300)
    
    plt.show()

    # Plotting the p-values heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(p_value_matrix.astype(float), annot=False, fmt=".3g", cmap=cmap,norm=norm, cbar=True, square=True, annot_kws={"weight": "extra bold"})
    plt.title(filename.replace('.png','_proportion_pvalues_uncorrected.png'))
        
    # Make tick labels bold
    plt.xticks(weight='extra bold',fontsize=fontsize)
    plt.yticks(weight='extra bold',fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(writeDir+filename.replace('.png','_proportion_pvalues_uncorrected.png'),dpi=300)
    
    
    # Plotting the p-values heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(p_value_matrix_corrected.astype(float), annot=False, fmt=".3g", cmap=cmap,norm=norm, cbar=True, square=True, annot_kws={"weight": "extra bold"})
    plt.title(filename.replace('.png','_proportion_pvalues_corrected.png'))
        
    # Make tick labels bold
    plt.xticks(weight='extra bold',fontsize=fontsize)
    plt.yticks(weight='extra bold',fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(writeDir+filename.replace('.png','_proportion_pvalues_corrected.png'),dpi=300)
    plt.show()
 
    
 
    
    mask = p_value_matrix_density > 0.05
    plt.figure(figsize=(16, 14))
    # sns.heatmap(correlation_matrix.astype(float), annot=True, fmt=".1f", cmap='coolwarm', cbar=True, square=True,center=0,vmin=-0.15,vmax=1, annot_kws={"weight": "bold"})
    sns.heatmap(densityCor.astype(float), annot=False, fmt=".1f", cmap='coolwarm', cbar=True, square=True, annot_kws={"weight": "extra bold"},center=0, vmin=-1, vmax=1, mask=mask)
    plt.title(filename.replace('.png','_density_masked_uncorrected.png'))
    
    # Make tick labels bold
    fontsize=14
    plt.xticks(weight='extra bold',fontsize=fontsize)
    plt.yticks(weight='extra bold',fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(writeDir+filename.replace('.png','_density_masked_uncorrected.png'),dpi=300)
    plt.show()
    
    
    mask = densityPcorrected > 0.05
    plt.figure(figsize=(16, 14))
    sns.heatmap(densityCor.astype(float), annot=False, fmt=".1f", cmap='coolwarm', cbar=True, square=True, annot_kws={"weight": "extra bold"},center=0, vmin=-1, vmax=1, mask=mask)
    plt.title(filename.replace('.png','_density_masked_corrected.png'))
    
    # Make tick labels bold
    fontsize=14
    plt.xticks(weight='extra bold',fontsize=fontsize)
    plt.yticks(weight='extra bold',fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(writeDir+filename.replace('.png','_density_masked_corrected.png'),dpi=300)
    plt.show()
    
    
    mask = p_value_matrix_proportion > 0.05
    plt.figure(figsize=(16, 14))
    sns.heatmap(proportionCor.astype(float), annot=False, fmt=".1f", cmap='coolwarm', cbar=True, square=True, annot_kws={"weight": "extra bold"},center=0, vmin=-1, vmax=1, mask=mask)
    plt.title(filename.replace('.png','_proportion_masked_uncorrected.tif'))
    
    # Make tick labels bold
    fontsize=14
    plt.xticks(weight='extra bold',fontsize=fontsize)
    plt.yticks(weight='extra bold',fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(writeDir+filename.replace('.png','_proportion_masked_uncorrected.png'),dpi=300)
    plt.show()

    mask = proportionPcorrected > 0.05
    plt.figure(figsize=(16, 14))
    sns.heatmap(proportionCor.astype(float), annot=False, fmt=".1f", cmap='coolwarm', cbar=True, square=True, annot_kws={"weight": "extra bold"},center=0, vmin=-1, vmax=1, mask=mask)
    plt.title(filename.replace('.png','_proportion_masked_corrected.png'))
    
    # Make tick labels bold
    fontsize=14
    plt.xticks(weight='extra bold',fontsize=fontsize)
    plt.yticks(weight='extra bold',fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(writeDir+filename.replace('.png','_proportion_masked_corrected.png'),dpi=300)
    plt.show()


    
def cell_class_vs_total_df(writeDir,cell_class,markers,colors,df,total_count,filename): 
    x =  total_count['Total_Count']

    if len(cell_class)>1:
        y =  df[cell_class].sum(axis=1)
    else:
        y =  df[cell_class]
    
    # fig, ax = plt.subplots(1)
    plt.figure(figsize=(10, 6))


    # Create a DataFrame to store results
    df_results = pd.DataFrame(columns=["Feature", "Coefficient", "P-value", "Category"])

    
    for category, color in colors.items():

        mask =df['disease_cohort'] == category
       
        
        if not x[mask].empty:
        
            X,Y =x[mask].to_frame(),y[mask]

            X_const = sm.add_constant(X)  # Adds a constant term to the predictor variables
            model = sm.OLS(Y, X_const).fit()
            
            # Extract coefficients and p-values
            coefficients = model.params
            pvalues = model.pvalues
            
            # Append results to the DataFrame
            for feature, coeff, pval in zip(X_const.columns, coefficients, pvalues):
                df_results = df_results.append({
                    "Feature": feature,
                    "Coefficient": coeff,
                    "P-value": pval,
                    "Category": category
                }, ignore_index=True)
            
            y_pred = model.predict(X_const)
            r_sq = model.rsquared

            plt.scatter(x[mask], y[mask], c=color, s=35, label=category)
            plt.plot(X,y_pred,c=color)
        

    plt.title(filename, weight='extra bold', fontsize=12)
    plt.grid(True)
    plt.ylabel(f'Density of {cell_class} cells', weight='extra bold', fontsize=12)
    plt.xlabel('Total Density of Cells', weight='extra bold', fontsize=12)
    plt.style.use('default')
    plt.legend(title='Disease Cohort', loc='upper left', bbox_to_anchor=(1.05, 1))
    for label in plt.gca().get_xticklabels():
        label.set_weight('bold')
        label.set_size(11)  # Adjust size as needed
    for label in plt.gca().get_yticklabels():
        label.set_weight('bold')
        label.set_size(11)  # Adjust size as needed
    for spine in plt.gca().spines.values():
        spine.set_linewidth(8)  # Set the thickness here
        
    plt.tight_layout()
    
    # Save the plot if you need to
    plt.savefig(writeDir+filename,dpi=300)
    plt.show()
    plt.close()
    
    
    df_results.to_csv(writeDir+filename.replace('tif','csv'), index=False)


def cell_class_ratio_plot_2D(dfTemp,writeDir,X,Y,colors,xlabel,ylabel,title,filename,proportion_df,count_df):
    
   
    results_list = []
    cohort_counter = 0
    # Plot of proportion og T cell, Myeloid Cells
    plt.figure(figsize=(10, 8))
    for cohort, color in colors.items():
        subset = count_df[count_df['disease_cohort'] == cohort]
        

        if not subset.empty:
            plt.scatter(subset[X].sum(axis=1), subset[Y].sum(axis=1), c=color, 
                         label=cohort,s=50)
            
            
            
            X_,Y_ =subset[X].sum(axis=1),subset[Y].sum(axis=1)
            
         
            X_const = sm.add_constant(X_)  # Adds a constant term to the predictor variables
            model = sm.OLS(Y_, X_const).fit()
            
            # Extract coefficients and p-values
            coefficients = model.params
            pvalues = model.pvalues
            
            # Append results to the DataFrame
            counter =0
            
            for feature, coeff, pval in zip(X_const.columns, coefficients, pvalues):
                print(feature,coeff,pval,cohort)
                results= {
                    "Feature": feature,
                    "Coefficient": coeff,
                    "P-value": pval,
                    "Category": cohort
                }
                results_list.append(results)
                counter +=1
            
            y_pred = model.predict(X_const)
            r_sq = model.rsquared
            print(f"coefficient of determination: {r_sq}")
    
            plt.plot(X_,y_pred,c=color)
            
            if counter == 2:
                specific_result = results  # The specific result you mentioned
                plt.figtext(0.85, 0.01+(cohort_counter*0.05), 
                            f"Feature: {specific_result['Feature']}, Coeff: {specific_result['Coefficient']:.2f}, P-value: {specific_result['P-value']:.2e}, Category: {specific_result['Category']}", 
                            ha="center", fontsize=7, bbox={"facecolor":"orange", "alpha":0.5, "pad":3})
                
                cohort_counter +=1
 

    plt.xlabel(xlabel, weight='extra bold', fontsize=12)
    plt.ylabel(ylabel, weight='extra bold', fontsize=12)
    plt.title(title+ 'Cell Density', weight='extra bold', fontsize=12)
    plt.grid(True)
    plt.style.use('default')
    plt.legend(title='Disease Cohort', loc='upper left', bbox_to_anchor=(1.2, 1))
    
    

    
    
    
    for label in plt.gca().get_xticklabels():
        label.set_weight('bold')
        label.set_size(18)  # Adjust size as needed
    for label in plt.gca().get_yticklabels():
        label.set_weight('bold')
        label.set_size(18)  # Adjust size as needed
        # Looping over spines to set their linewidth
    for spine in plt.gca().spines.values():
        spine.set_linewidth(4)  # Set the thickness here
    plt.tight_layout()
    
    # Save the plot if you need to
    plt.savefig(writeDir + filename+'_density.tiff',dpi=300)
    plt.show()
    plt.close()
    
    df_results = pd.DataFrame(results_list)
    
    
    df_results.to_csv(writeDir+filename+'_density.csv', index=False)
    df_results = pd.DataFrame(columns=["Feature", "Coefficient", "P-value", "Category"])

    
    results_list = []
    cohort_counter = 0

    # Plot of proportion og T cell, Myeloid Cells
    plt.figure(figsize=(10, 8))
    for cohort, color in colors.items():
        subset = proportion_df[proportion_df['disease_cohort'] == cohort]
        

        if not subset.empty:
        
            plt.scatter(subset[X].sum(axis=1), subset[Y].sum(axis=1), c=color
                     , label=cohort,s=50)

            
            X_,Y_ =subset[X].sum(axis=1),subset[Y].sum(axis=1)
      
            X_const = sm.add_constant(X_)  # Adds a constant term to the predictor variables
            model = sm.OLS(Y_, X_const).fit()
            
            # Extract coefficients and p-values
            coefficients = model.params
            pvalues = model.pvalues
            counter =0
            # Append results to the DataFrame
            for feature, coeff, pval in zip(X_const.columns, coefficients, pvalues):
                results = {
                    "Feature": feature,
                    "Coefficient": coeff,
                    "P-value": pval,
                    "Category": cohort
                }
                counter +=1
            results_list.append(results)
            y_pred = model.predict(X_const)
            r_sq = model.rsquared
            
            plt.plot(X_,y_pred,c=color)
            
            if counter == 2:
                specific_result = results  # The specific result you mentioned
                plt.figtext(0.85, 0.01+(cohort_counter*0.05), 
                            f"Feature: {specific_result['Feature']}, Coeff: {specific_result['Coefficient']:.2f}, P-value: {specific_result['P-value']:.2e}, Category: {specific_result['Category']}", 
                            ha="center", fontsize=7, bbox={"facecolor":"orange", "alpha":0.5, "pad":3})
                
                cohort_counter +=1


    plt.xlabel(xlabel, weight='extra bold', fontsize=12)
    plt.ylabel(ylabel, weight='extra bold', fontsize=12)
    plt.title(title+' Proportion', weight='extra bold', fontsize=12)
    plt.grid(True)
    plt.style.use('default')
    plt.legend(title='Disease Cohort', loc='upper left', bbox_to_anchor=(1.2, 1))
    for label in plt.gca().get_xticklabels():
        label.set_weight('bold')
        label.set_size(18)  # Adjust size as needed
    for label in plt.gca().get_yticklabels():
        label.set_weight('bold')
        label.set_size(18)  # Adjust size as needed
        # Looping over spines to set their linewidth
    for spine in plt.gca().spines.values():
        spine.set_linewidth(4)  # Set the thickness here
    plt.tight_layout()
    
    # Save the plot if you need to
    plt.savefig(writeDir + filename+'_proportion.tiff',dpi=300)
    plt.show()
    plt.close()
    df_results = pd.DataFrame(results_list)
    df_results.to_csv(writeDir+filename+'_proportion.csv', index=False)
    

def T_Humoral_myeloid_plots(merged_df,total_count,writeDir,X,Y,Z,markers,colors,color_mapping,column,filename): 
   
   
  
    # Normalize the Z values for alpha scaling (transparency)
    z_values = merged_df[Z].sum(axis=1)
    norm = plt.Normalize(z_values.min(), z_values.max())

    plt.figure(figsize=(10, 10))
    for cohort in merged_df[column].unique():
        subset = merged_df[merged_df[column] == cohort]
        
        # Base RGB color from the color_mapping dictionary
        rgb_color = color_mapping.get(cohort, [0, 0, 0])[:3]  # Default to black if not found

        # Iterate over each point in the subset to set individual alpha and edge colors
        for i, z_val in enumerate(subset[Z].sum(axis=1)):
            alpha = norm(z_val)
            rgba_color = rgb_color + [alpha]  # Construct RGBA color
            plt.scatter(subset[X].iloc[i], subset[Y].iloc[i], color=[rgba_color], edgecolor=rgb_color, s=80)


    plt.grid(True)

    # Style adjustments
    for label in plt.gca().get_xticklabels():
        label.set_weight('bold')
        label.set_size(12)
    for label in plt.gca().get_yticklabels():
        label.set_weight('bold')
        label.set_size(12)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(8)
    plt.tight_layout()

    plt.savefig(writeDir + 'T_Humoral_myeloid_density_all_cohorts'+filename,dpi=300)
    plt.show()
    plt.close()
    
  
    x = total_count['Total_Count']
    y =  merged_df[X]/merged_df[Y]
   
    plt.figure(figsize=(10, 6))

    # Create a scatter plot with color coding based on 'color' column
    for category, color in colors.items():
        mask = merged_df[column] == category
        x = total_count['Total_Count']
        # y =  merged_df[X]/merged_df[Y]
        plt.scatter(x[mask], y[mask], c=color, s=35, label=category)


    plt.title(f'Scatter plot Ratio of {X}/{Y} cells v. total # of Cells', weight='extra bold', fontsize=12)
    plt.grid(True)
    plt.ylabel(f'Ratio of {X}/{Y} cells', weight='extra bold', fontsize=12)
    plt.xlabel('Total Density of Cells', weight='extra bold', fontsize=12)
    plt.style.use('default')
    plt.legend(title='Disease Cohort', loc='upper left', bbox_to_anchor=(1.05, 1))
    for label in plt.gca().get_xticklabels():
        label.set_weight('bold')
        label.set_size(11)  # Adjust size as needed
    for label in plt.gca().get_yticklabels():
        label.set_weight('bold')
        label.set_size(10)  # Adjust size as needed
    for spine in plt.gca().spines.values():
        spine.set_linewidth(8)  # Set the thickness here
    plt.tight_layout()
    
    # Save the plot if you need to
    plt.savefig(writeDir + 'T_myeloid_ratio_v_density'+filename,dpi=300)
    plt.show()
    plt.close()
    


def cell_class_ratio_plot_3D(dfTemp,writeDir,X,Y,Z,markers,xlabel,ylabel,zlabel,title,filename,merged_df,color_mapping,column,sort_column):
    
    
    merged_df['Sum_X'] = merged_df[X].sum(axis=1)
    merged_df['Sum_Y'] = merged_df[Y].sum(axis=1)
    merged_df['Sum_Z'] = merged_df[Z].sum(axis=1)  # Assuming Z refers to multiple columns for summing

 
    summary_df = merged_df[['AccessionNumber', 'disease_cohort', 'TissueType', 'Sum_X', 'Sum_Y', 'Sum_Z']].copy()
    
    # Proceed with your plotting as before
    plt.figure(figsize=(10, 10))
    for cohort in merged_df[column].unique():
        subset = merged_df[merged_df[column] == cohort]
        rgb_color = color_mapping.get(cohort, [0, 0, 0])[:3]  # Default to black if not found

        for i, row in subset.iterrows():
            
            size = row['Sum_Z'] * 200  # Adjust the scaling factor as needed
            rgba_color = rgb_color + [1]  # Construct RGBA color
            plt.scatter(row['Sum_X'], row['Sum_Y'], color=[rgba_color], edgecolor=rgb_color, s=size)

    plt.xlabel(xlabel, weight='extra bold', fontsize=12)
    plt.ylabel(ylabel, weight='extra bold', fontsize=12)
    plt.title(title + ' Cell Density', weight='extra bold', fontsize=12)
    plt.grid(True)
    
    # Style adjustments
    for label in plt.gca().get_xticklabels():
        label.set_weight('bold')
        label.set_size(12)
    for label in plt.gca().get_yticklabels():
        label.set_weight('bold')
        label.set_size(12)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()

    plt.savefig(writeDir + filename + '_density.png', dpi=300)
    plt.show()
    plt.close()
    
    summary_df = summary_df.sort_values(by=sort_column, ascending=False)
    summary_df.to_csv(writeDir + filename + '_summary.csv', index=False)
    
    return(summary_df)


def stacked_bar_plot_cell_class(dfTemp,writeDir,filename,colorDict,unique_labels):
    # Count the occurrences of each (Patient_ID, Class_Label) pair
    count_df = dfTemp.groupby(['AccessionNumber', 'class_label']).size().reset_index(name='Count')
    
    # Calculate the sum of 'Count' for each 'Patient_ID' to get the total number of occurrences for each patient
    total_count =  dfTemp.groupby('AccessionNumber').size().reset_index(name='Total_Count')
    
    # Merge to attach the total_count to each original entry
    count_df = pd.merge(count_df, total_count, on='AccessionNumber')
    
    # Calculate the proportion
    count_df['Proportion'] = count_df['Count'] / count_df['Total_Count']
    
    # Create the stacked barplot
    
    plt.figure(figsize=(14, 8))
     
    unique_patients = count_df['AccessionNumber'].unique()
    

    bottoms = np.zeros(len(unique_patients))
    
    for i, label in enumerate(unique_labels):
        print(i,label)
        heights = []
        for index, patient in enumerate(unique_patients):
            subset = count_df[(count_df['AccessionNumber'] == patient) & (count_df['class_label'] == label)]
            height = subset['Proportion'].values[0] if not subset.empty else 0
            heights.append(height)
        plt.bar(range(len(unique_patients)), heights, bottom=bottoms, label=label,color=colorDict[label])
         
        bottoms += np.array(heights)
 
    plt.style.use('default')
    plt.title('Proportion Breakdown of Class Labels per Unique Patient ID')
    plt.xlabel('Patient ID')
    plt.ylabel('Proportion')
    # Set the x-axis ticks and labels
    plt.xticks(range(len(unique_patients)), unique_patients, rotation=45, ha='right')  # Rotate for better readability if needed

    for label in plt.gca().get_xticklabels():
        label.set_weight('bold')
        label.set_size(11)  # Adjust size as needed
    for label in plt.gca().get_yticklabels():
        label.set_weight('bold')
        label.set_size(11)  # Adjust size as needed
    # Make the lines around the plot thicker
    ax = plt.gca()  # Get current axes
    linewidth = 2  # Set the desired linewidth
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)
    plt.legend(title='Class Label', bbox_to_anchor=(1.05, 1), loc='upper left',fontsize='small')
    plt.tight_layout()
    plt.savefig(writeDir+filename,dpi=300)
    plt.show()
    
    plt.close()
def stacked_bar_plot_cell_class(dfTemp, writeDir, filename, colorDict, unique_labels, order_by=None):
    # Count the occurrences of each (Patient_ID, Class_Label) pair
    count_df = dfTemp.groupby(['AccessionNumber', 'class_label']).size().reset_index(name='Count')

    # Calculate the sum of 'Count' for each 'Patient_ID' to get the total number of occurrences for each patient
    total_count = dfTemp.groupby('AccessionNumber').size().reset_index(name='Total_Count')

    # Merge to attach the total_count to each original entry
    count_df = pd.merge(count_df, total_count, on='AccessionNumber')
    
    # Calculate the proportion
    count_df['Proportion'] = count_df['Count'] / count_df['Total_Count']

    # Optional ordering by another column
    if order_by:
        # Merge with order_by column
        order_df = dfTemp[['AccessionNumber', order_by]].drop_duplicates()
        count_df = pd.merge(count_df, order_df, on='AccessionNumber')
       
        # Order based on order_by column and AccessionNumber
        count_df.sort_values(by=[order_by, 'AccessionNumber'], inplace=True)
        
    # Create the stacked barplot
    plt.figure(figsize=(14, 8))

    unique_patients = count_df['AccessionNumber'].unique()

    bottoms = np.zeros(len(unique_patients))

    for i, label in enumerate(unique_labels):
        heights = []
        for patient in unique_patients:
            subset = count_df[(count_df['AccessionNumber'] == patient) & (count_df['class_label'] == label)]
            height = subset['Proportion'].values[0] if not subset.empty else 0
            heights.append(height)

        plt.bar(range(len(unique_patients)), heights, bottom=bottoms, label=label, color=colorDict[label])
        bottoms += np.array(heights)

    plt.style.use('default')
    plt.title('Proportion Breakdown of Class Labels per Unique Patient ID')
    plt.xlabel('Patient ID')
    plt.ylabel('Proportion')

    # Set the x-axis ticks and labels
    plt.xticks(range(len(unique_patients)), unique_patients, rotation=45, ha='right')
    for label in plt.gca().get_xticklabels():
        label.set_weight('bold')
        label.set_size(11)  # Adjust size as needed
    for label in plt.gca().get_yticklabels():
        label.set_weight('bold')
        label.set_size(11)  # Adjust size as needed

    ax = plt.gca()  # Get current axes
    linewidth = 2  # Set the desired linewidth
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)
    plt.legend(title='Class Label', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(writeDir + filename, dpi=300)
    plt.show()

    plt.close()   
def plot_spider_charts_grid(selected_columns, count_df, color_mapping, save_name,use_x_label,use_sample_name):
    data = count_df[selected_columns]
    
    # Categories for the radar chart, excluding 'disease_cohort' and 'AccessionNumber'
    categories = selected_columns[:-2]
    data['std_dev'] = data[categories].std(axis=1)
    
    
    # Sort data by standard deviation in descending order
    data = data.sort_values(by='std_dev', ascending=False)

    # Remove the std_dev column for plotting
    data.drop(columns=['std_dev'], inplace=True)
        
    N = len(categories)
    
    # Prepare the angles for the radar chart
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # complete the loop

    # Number of plots, configure the grid size
    num_samples = data.shape[0]
    num_cols = 5  # Number of columns in the grid
    num_rows = (num_samples + num_cols - 1) // num_cols  # Calculate rows needed
    
    # Create figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*6, num_rows*6), subplot_kw=dict(polar=True))
    axes = np.array(axes).flatten()  # Flatten in case of a single row or column

    # Plot each entry
    for idx, (ax, (index, row)) in enumerate(zip(axes, data.iterrows())):
        values = row[categories].tolist()
        values += values[:1]  # complete the loop
        cohort = row['disease_cohort']
        accession_number = row['AccessionNumber']
    
        # Plot data on individual subplot
        ax.plot(angles, values, '-', linewidth=1, label=f'{cohort} ({accession_number})', color=color_mapping[cohort])
        ax.fill(angles, values, alpha=0.05, color=color_mapping[cohort])
        
        if use_x_label:
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, color='black', size=10)
        else:
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([])
            
        if use_sample_name:
            # Set title to include accession number, customize as needed
            ax.set_title(f'{accession_number}', fontsize=34, fontweight='extra bold')
            
            
        # ax.set_ylim(0)  # Optionally adjust dynamically based on data
        # print(f"angles: {angles}")
        # print(f"values: {values}")
        # print(f"angles[:-1]: {angles[:-1]}")
        # print(f"categories: {categories}")
        # print("list: ",list(map(str, range(0, 101, 20))))
        
    
        # # Draw ylabels on the left-most charts
        # if idx % num_cols == 0:
        #     ax.set_rlabel_position(0)
        #     ax.set_yticklabels(map(str, range(0, 101, 20)), color="black", size=8)

    # Hide unused subplots
    for ax in axes[num_samples:]:
        ax.set_visible(False)
    
    # Save and show the plot
    plt.tight_layout()
    print(save_name.replace('.tiff', '_biopsies_grid.tiff'))
    plt.savefig(save_name.replace('.tiff', '_biopsies_grid.tiff'), dpi=300)
    plt.show()    
    
def density_normalize(dfTemp,areaDicts):

    Acc_CompName = dfTemp.groupby('AccessionNumber')['CompName'].unique().to_dict()
     
    # Count the occurrences of each (Patient_ID, Class_Label) pair
    count_df = dfTemp.groupby(['AccessionNumber', 'class_label']).size().reset_index(name='Count')
    
    ##lets normalize the counts by area of capture
     
    NKAreaDict,LNAreaDict,ARAreaDict = areaDicts
    for acc in count_df['AccessionNumber'].unique():
        coh = dfTemp[dfTemp['AccessionNumber']==acc]['disease_cohort'].unique()
        tempComps = Acc_CompName[acc] ##all the CompNames associated with this particular patient acc #
        
        if coh == 'Normal_Kidney':
            areas = np.sum([NKAreaDict.get(k) for k in tempComps])
            dat = count_df[count_df['AccessionNumber']==acc]['Count']
            count_df['Count'][count_df['AccessionNumber']==acc] = dat/areas
        if coh == 'Lupus_Nephritis':
            
            areas = np.sum([LNAreaDict.get(k) for k in tempComps])
            
            dat = count_df[count_df['AccessionNumber']==acc]['Count']
           
            count_df['Count'][count_df['AccessionNumber']==acc] = dat/areas
            
        if coh == 'Renal_Allograft':
            areas = np.sum([ARAreaDict.get(k) for k in tempComps])
            dat = count_df[count_df['AccessionNumber']==acc]['Count']
            count_df['Count'][count_df['AccessionNumber']==acc] = dat/areas
            
        print('#$%#$%#$%$3')
     
     # Calculate the total count for each 'AccessionNumber'
    total_count = count_df.groupby('AccessionNumber')['Count'].sum().reset_index(name='Total_Count')
    
     # Merge to attach the total_count to each original entry
    count_df = pd.merge(count_df, total_count, on='AccessionNumber')
    
     # Calculate the proportion
    count_df['Proportion'] = count_df['Count'] / count_df['Total_Count']
    # Pivot table to get class_labels as columns
    proportion_df = count_df.pivot_table(index='AccessionNumber', columns='class_label', values='Proportion', aggfunc='first').reset_index()
    count_df = count_df.pivot_table(index='AccessionNumber', columns='class_label', values='Count', aggfunc='first').reset_index()
    
    count_df.fillna(0, inplace=True)
    
    proportion_df.fillna(0, inplace=True)
     # Merge with the original dfTemp to get the disease_cohort for each AccessionNumber
    proportion_df = pd.merge( proportion_df, dfTemp[['AccessionNumber', 'disease_cohort','TissueType']].drop_duplicates(), on='AccessionNumber')
    count_df = pd.merge( count_df, dfTemp[['AccessionNumber', 'disease_cohort','TissueType']].drop_duplicates(), on='AccessionNumber')

    
    return(proportion_df,count_df,total_count)

def stain_score_df_generate(df,stain_dir,stain):
    

    dfs = []  # Initialize an empty list to store the dataframes

    for cohort in df['disease_cohort'].unique():
        print(cohort)
        scoreDf = pd.DataFrame(pd.read_csv(stain_dir + f'{cohort}/mask_score_tmp/{cohort}_{stain}_mask_score.csv', index_col=False))
        
        scoreDf['disease_cohort'] = cohort
        dfs.append(scoreDf)  # Append the dataframe to the list
    
    # Concatenate all dataframes in the list
    scoreDf = pd.concat(dfs, ignore_index=True)
    return(scoreDf)


def add_optimal_threshold_column(score_df,thresh):
    # Function to determine the optimal threshold
    def determine_optimal_threshold(row):
        return row['th3_score'] if row['th1_score'] > thresh else row['th2_score']
    
    
    # Apply the function across the rows
    score_df['th_optimal'] = score_df.apply(determine_optimal_threshold, axis=1)
    return score_df    



#%%


def spider_plot(selected_columns,count_df,color_mapping,saveName,use_x_label):
    data = count_df[selected_columns]
    
    # Categories for the radar chart
    categories = selected_columns[:-2]  # Exclude 'disease_cohort' and 'AccessionNumber'
    N = len(categories)
    
    # Prepare the angles for the radar chart
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # complete the loop
    
    # Initialize the spider plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot each entry
    for index, row in data.iterrows():
        values = row[categories].tolist()
        values += values[:1]  # complete the loop
        cohort = row['disease_cohort']
        accession_number = row['AccessionNumber']
    
        # Plot data
        ax.plot(angles, values, '-', linewidth=1, label=f'{cohort} ({accession_number})', color=color_mapping[cohort])
        ax.fill(angles, values, alpha=0.05, color=color_mapping[cohort])
    
    # Add labels for each axis
    if use_x_label:
        plt.xticks(angles[:-1], categories, color='black', size=10)
    else:
        plt.xticks(angles[:-1], [], color='black', size=10)
        
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks(color="black", size=8)
    plt.ylim(0)  # Set dynamically based on data, if required
    
    # Add legend
    plt.savefig(saveName.replace('.tiff','_biopsies.tiff'), dpi=300)
    plt.show()

    
    data = count_df[categories+['disease_cohort']]
    
    # Group data by 'disease_cohort' and calculate means
    grouped_data = data.groupby('disease_cohort').mean()
    
    # Categories for the radar chart
    categories = grouped_data.columns.tolist()
    N = len(categories)
    
    # Prepare the angles for the radar chart
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # complete the loop
    
    # Initialize the spider plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot each disease cohort
    for cohort, values in grouped_data.iterrows():
        values = values.tolist()
        values += values[:1]  # complete the loop
    
        ax.plot(angles, values, '-', linewidth=2, label=cohort, color=color_mapping.get(cohort, 'gray'))
        ax.fill(angles, values, alpha=0.1, color=color_mapping.get(cohort, 'gray'))
    
    # Add labels for each axis
    if use_x_label:
        plt.xticks(angles[:-1], categories, color='black', size=10)
    else:
        plt.xticks(angles[:-1], [], color='black', size=10)
    
    # Draw ylabels
    ax.set_rlabel_position(1)
    plt.yticks(color="black", size=8)
    plt.ylim(0)  # Set dynamically based on data
    

    plt.savefig(saveName.replace('.tiff','_cohort.tiff'), dpi=300)
    plt.show()

def plot_grouped_dot_plot(df, groups,saveName,colToUse,title,use_labels):
    # Prepare the plot
    fig, ax = plt.subplots()

    # Iterate over each group and plot the data
    for i, group in enumerate(groups):
        # Filter the dataframe to include only the rows with AccessionNumbers in the current group
        group_df = df[df['AccessionNumber'].isin(group)]
        
        # Calculate mean and standard error
        mean = group_df[colToUse].mean()
        error = group_df[colToUse].sem() 
        ax.errorbar(i, mean, yerr=error, fmt='o', color='black',capsize=5)
        
        
        # Plotting the 'th_optimal' values for the current group
        ax.scatter([i] * len(group_df), group_df[colToUse], label='Group {}'.format(i+1))
        
        # Add a text label to the x-axis at the position of each group
        if use_labels:
            ax.text(i, -0.05, '\n'.join(group), ha='center', va='top', fontsize=8, rotation=45)

    # Set plot title and labels
    if use_labels:
        ax.set_title(title)
        ax.set_xlabel('Groups of AccessionNumbers')
        ax.set_ylabel(colToUse)
    ax.legend(title="Groups", loc='upper left', bbox_to_anchor=(1, 1))
    
    # Display the plot
    plt.xticks(range(len(groups)))  # Set x-ticks to be the groups
   
    plt.savefig(saveName, dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_groups_with_ols_stats(df, x_col, y_col, groups,y_name,saveName,use_labels):
    # Setting up the color cycle for distinct group colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))
    
    # Create a plot
    fig, ax = plt.subplots(figsize=(8, 8))
    stats_text = []
    
    # Loop over each group and their corresponding color
    for group, color in zip(groups, colors):
        # Filter the dataframe for the current group
        group_df = df[df['AccessionNumber'].isin(group)]
        
        # Scatter plot for the group
        ax.scatter(group_df[x_col], group_df[y_col], label=f'Group {groups.index(group)+1}')
        
        # Fit OLS regression
        if not group_df.empty:
            X = sm.add_constant(group_df[x_col])  # adding a constant for the intercept
            model = sm.OLS(group_df[y_col], X).fit()
            line = model.params['const'] + model.params[x_col] * np.linspace(group_df[x_col].min(), group_df[x_col].max(), 100)
            ax.plot(np.linspace(group_df[x_col].min(), group_df[x_col].max(), 100), line)
            x_range = np.linspace(group_df[x_col].min(), group_df[x_col].max(), 100)

            # Print out the R-squared and p-value for each group's regression
            print(f"Group {groups.index(group)+1} (Color index {groups.index(group)}):")
            print(f" R-squared: {model.rsquared:.4f}")
            print(f" p-value for {x_col}: {model.pvalues[x_col]:.4g}")  # p-value for the slope
            
            # ax.text(x_range.mean()+0.1, line.mean(), 
            #     f'$R^2 = {model.rsquared:.3f}, p = {model.pvalues[x_col]:.3g}$', 
            #     color=color, fontsize=10)
            # Store R-squared and p-value for annotation
            stats_text.append(f"Group {groups.index(group)+1}: $R^2 = {model.rsquared:.3f}, p = {model.pvalues[x_col]:.3g}$")
        
            
    # Set labels and title
    if use_labels:
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_name)
        ax.set_title(f'Scatter plot of {x_col} vs {y_name} with OLS regression lines')
    
    # Show legend
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Show legend to the right of the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Annotate R-squared and p-value beneath the legend
    for i, text in enumerate(stats_text):
        plt.gcf().text(0.95, 0.75 - i * 0.05, text, fontsize=10, verticalalignment='top', horizontalalignment='left')


    # Show plot
    plt.savefig(saveName,dpi=300, bbox_inches='tight')
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
    parser.add_argument('-sm',
            '--stain_mask_score_dir',
            type=str,
            default='',
            help=''
            )
    parser.add_argument('-jr',
            '--JOINT_ROOT_DIR',
            type=str,
            default='',
            help=''
            )
    parser.add_argument('-cd',
            '--color_dict',
            type=str,
            default='',
            help=''
            )


    args,unparsed = parser.parse_known_args()

    
    
    #%%
    with open(args.color_dict,'rb') as f:
       colorDict = pickle.load(f)
    
    
    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)

    
    df = pd.DataFrame(pd.read_csv(args.cell_expression_dir,index_col=False))
    df =convert_mixed_dtype_to_string(df)
    dfInfo = pd.DataFrame(pd.read_csv(args.cell_info,index_col=False))
    dfInfo =convert_mixed_dtype_to_string(dfInfo)

    
    df['class_label'].unique()
    
    
    #%%
     
    MXAscoreDf = stain_score_df_generate(df,stain_dir=args.stain_mask_score_dir,stain='MXA')
    COLIIIscoreDf = stain_score_df_generate(df,stain_dir=args.stain_mask_score_dir,stain='COLIII')
   
    MXAscoreDf = add_optimal_threshold_column(score_df=MXAscoreDf,thresh =0.70)
    COLIIIscoreDf = add_optimal_threshold_column(score_df=COLIIIscoreDf,thresh =0.70)
    
    MXAscoreDf = MXAscoreDf.merge(dfInfo[['disease_cohort', 'CompName','AccessionNumber']].drop_duplicates(), on=['disease_cohort', 'CompName'], how='inner')
    COLIIIscoreDf = COLIIIscoreDf.merge(dfInfo[['disease_cohort', 'CompName','AccessionNumber']].drop_duplicates(), on=['disease_cohort', 'CompName'], how='inner')
    
#%%
    
    with open(args.area_dict+'NKAreaDict.pkl','rb') as f:
        NKAreaDict = pickle.load(f)
    with open(args.area_dict+'LNAreaDict.pkl','rb') as f:
        LNAreaDict = pickle.load(f)
    with open(args.area_dict+'ARAreaDict.pkl','rb') as f:
        ARAreaDict = pickle.load(f)

    #%%
    dfInfo =dfInfo.loc[df.index]
    df['CompName'] = dfInfo['CompName']
    
    df = df[~df['class_label'].isin(['general_unknown'])]##removes it
    dfTemp= df.copy()
    dfTempSimplified = df.copy()
    
    
    #%%
    immuneCells= ['Bcell', 'CD14+CD163+MERTK+_macrophages',
           'CD14+CD163+_macrophages', 'CD14+MERTK+_macrophages',
           'CD14+_macrophages', 'CD16+_macrophages', 'CD3+_Tcell',
           'CD4+CD8+_Tcell', 'CD4+Foxp3+_Tcell', 'CD4+ICOS+PD1+_Tcell',
           'CD4+ICOS+_Tcell', 'CD4+PD1+_Tcell', 'CD4+_Tcell', 'CD8+Foxp3+_Tcell',
           'CD8+ICOS+PD1+_Tcell', 'CD8+ICOS+_Tcell', 'CD8+PD1+_Tcell',
           'CD8+_Tcell','CD4+_intraepithelial_Tcell','CD8+_intraepithelial_Tcell' ,'BDCA2pCD103p_DCs',
           'HLAII+_Monocytes', 'HLAII-_Monocytes', 
           'Mo-Macrophage', 'NK_Tcell', 'NK_cells', 'Neutrophils', 'Plasmablasts',
            'TCRgd_Tcell', 'cDC1', 'cDC2', 'pDC',
           'plasma_cells']
    
    thresholds = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]

    for thresh in thresholds:
        print(thresh)
        
            
        MXAscoreDf = stain_score_df_generate(df,stain_dir=args.stain_mask_score_dir,stain='MXA')
        COLIIIscoreDf = stain_score_df_generate(df,stain_dir=args.stain_mask_score_dir,stain='COLIII')
       
        MXAscoreDf = add_optimal_threshold_column(score_df=MXAscoreDf,thresh =thresh)
        COLIIIscoreDf = add_optimal_threshold_column(score_df=COLIIIscoreDf,thresh =thresh)
        
        MXAscoreDf = MXAscoreDf.merge(dfInfo[['disease_cohort', 'CompName','AccessionNumber']].drop_duplicates(), on=['disease_cohort', 'CompName'], how='inner')
        COLIIIscoreDf = COLIIIscoreDf.merge(dfInfo[['disease_cohort', 'CompName','AccessionNumber']].drop_duplicates(), on=['disease_cohort', 'CompName'], how='inner')
        

        dfInfo =dfInfo.loc[df.index]
        df['CompName'] = dfInfo['CompName']
        
        df = df[~df['class_label'].isin(['general_unknown'])]##removes it
        dfTemp= df.copy()
        dfTempSimplified = df.copy()
    

        color_mapping = {'Normal_Kidney':[0,1,0,1],'Lupus_Nephritis':[0,0,1,1], 'Renal_Allograft':[1,0,1,1]}
        proportion_df,count_df,total_count = density_normalize(dfTemp,areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
        
        mean_df = COLIIIscoreDf.groupby('AccessionNumber')['th_optimal'].mean().reset_index()
        
        proportion_df_colIII = proportion_df.merge(mean_df, on=['AccessionNumber'], how='inner')
        count_df_colIII = count_df.merge(mean_df, on=['AccessionNumber'], how='inner')
   
    
        cell_class_ratio_plot_2D(count_df_colIII,args.write_dir,Y=['CD14+MERTK+_macrophages'],X=['th_optimal'],
                              ylabel='CD14+MerTk+ Macrophages',
                              colors=color_mapping,
                              xlabel='COLIII score',
                              title='CD14+MerTk+ Ma v COLIII (Patient level)',
                              filename=f'{thresh}_CD14MerTkMacrophage_collagen_score', proportion_df= proportion_df_colIII ,count_df=count_df_colIII)
        
    
   
    #%%
    color_mapping = {
        'Normal_Kidney': 'blue',  # category A will be blue
        'Lupus_Nephritis': 'red',   # category B will be red
        'Renal_Allograft': 'orange' # category C will be orange
        # Add more categories and colors if needed
    }
    
    # Assuming dfTemp is your DataFrame
    dfTempSimplified['class_label'] = dfTempSimplified['class_label'].replace('CD4+Foxp3+_Tcell', 'CD4+_Tcell')
    dfTempSimplified['class_label'] = dfTempSimplified['class_label'].replace('CD4+ICOS+PD1+_Tcell', 'CD4+_Tcell')
    dfTempSimplified['class_label'] = dfTempSimplified['class_label'].replace('CD4+PD1+_Tcell', 'CD4+_Tcell')
    dfTempSimplified['class_label'] = dfTempSimplified['class_label'].replace('CD4+ICOS+_Tcell', 'CD4+_Tcell')
    dfTempSimplified['class_label'] = dfTempSimplified['class_label'].replace('CD4+_intraepithelial_Tcell', 'CD4+_Tcell')
    dfTempSimplified['class_label'] = dfTempSimplified['class_label'].replace('CD8+Foxp3+_Tcell', 'CD8+_Tcell')
    dfTempSimplified['class_label'] = dfTempSimplified['class_label'].replace('CD8+ICOS+PD1+_Tcell', 'CD8+_Tcell')
    dfTempSimplified['class_label'] = dfTempSimplified['class_label'].replace('CD8+PD1+_Tcell', 'CD8+_Tcell')
    dfTempSimplified['class_label'] = dfTempSimplified['class_label'].replace('CD8+ICOS+_Tcell', 'CD8+_Tcell')
    dfTempSimplified['class_label'] = dfTempSimplified['class_label'].replace('CD8+_intraepithelial_Tcell', 'CD8+_Tcell')
    dfTempSimplified['class_label'] = dfTempSimplified['class_label'].replace('CD14+CD163+MERTK+_macrophages', 'CD14+CD163+_macrophages')
    dfTempSimplified['class_label'] = dfTempSimplified['class_label'].replace('CD14+MERTK+_macrophages', 'CD14+CD163-_macrophages')
    dfTempSimplified['class_label'] = dfTempSimplified['class_label'].replace('CD14+_macrophages', 'CD14+CD163-_macrophages')
    
    #%%
    color_mapping = {'Normal_Kidney':[0,1,0,1],'Lupus_Nephritis':[0,0,1,1], 'Renal_Allograft':[1,0,1,1]}
    proportion_df,count_df,total_count = density_normalize(dfTempSimplified,areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
    
    
    #%%
    # Step 2: Filter and focus on the relevant data
    selected_columns = [
         'CD4+_Tcell',
                           'CD8+_Tcell',
                          'CD14+CD163+_macrophages','CD14+CD163-_macrophages','HLAII+_Monocytes',
                          'HLAII-_Monocytes',
        'AccessionNumber','disease_cohort'
    ]
    
    
    spider_plot(selected_columns,count_df,color_mapping,args.write_dir+'simplified_spider_plot.tiff',True)
    spider_plot(selected_columns,count_df,color_mapping,args.write_dir+'simplified_spider_plot_noLabels.tiff',False)
    #%%

    for cohort in count_df['disease_cohort'].unique():
        spider_plot(selected_columns,count_df[count_df['disease_cohort']==cohort],color_mapping,args.write_dir+f'{cohort}_simplified_spider_plot.tiff',True)
        spider_plot(selected_columns,count_df[count_df['disease_cohort']==cohort],color_mapping,args.write_dir+f'{cohort}_simplified_spider_plot.tiff_noLabels.tiff',False)
    #%%
    
    
#%%

    for cohort in count_df['disease_cohort'].unique():

        plot_spider_charts_grid(selected_columns, count_df[count_df['disease_cohort']==cohort], color_mapping, f'{args.write_dir}{cohort}_filename.tiff',True,False)
    
        
        for col in selected_columns:
            print(count_df[count_df['disease_cohort']==cohort][col])
#%%
#
    for cohort in count_df['disease_cohort'].unique():
        print(cohort)
        
        plot_spider_charts_grid(selected_columns, count_df[count_df['disease_cohort']==cohort], color_mapping, f'{args.write_dir}{cohort}_filename_Accession.tiff',True,True)
    
        
        for col in selected_columns:
            print(count_df[count_df['disease_cohort']==cohort][col])
   #%%

    for cohort in count_df['disease_cohort'].unique():
        print(cohort)

        plot_spider_charts_grid(selected_columns, count_df[count_df['disease_cohort']==cohort], color_mapping, f'{args.write_dir}{cohort}_filename_noLabels.tiff',False,False)
    
        
        for col in selected_columns:
            print(count_df[count_df['disease_cohort']==cohort][col])
 #%%

    for cohort in count_df['disease_cohort'].unique():
        print(cohort)
        plot_spider_charts_grid(selected_columns, count_df[count_df['disease_cohort']==cohort], color_mapping, f'{args.write_dir}{cohort}_filename_noLabels_Accession.tiff',False,True)
    
        
        for col in selected_columns:
            print(count_df[count_df['disease_cohort']==cohort][col])

    #%%
    selected_columns = [
         'Bcell','Plasmablasts','plasma_cells','CD4+Foxp3+_Tcell',
                           'CD4+ICOS+PD1+_Tcell','CD4+ICOS+_Tcell','CD4+PD1+_Tcell','CD4+_Tcell','CD4+_intraepithelial_Tcell',
                           'CD8+Foxp3+_Tcell','CD8+ICOS+PD1+_Tcell','CD8+ICOS+_Tcell','CD8+PD1+_Tcell',
                           'CD8+_Tcell','CD8+_intraepithelial_Tcell','CD3+_Tcell','CD4+CD8+_Tcell','TCRgd_Tcell','NK_Tcell','NK_cells','CD14+CD163+MERTK+_macrophages',
                          'CD14+CD163+_macrophages','CD14+MERTK+_macrophages','CD14+_macrophages','CD16+_macrophages','HLAII+_Monocytes',
                          'HLAII-_Monocytes','Mo-Macrophage','Neutrophils','BDCA2pCD103p_DCs','cDC1','cDC2','pDC',
        'AccessionNumber','disease_cohort'
    ]
    spider_plot(selected_columns,count_df,color_mapping)
    
    
  
    
    #%%
   
    color_mapping = {'Normal_Kidney':[0,1,0,1],'Lupus_Nephritis':[0,0,1,1], 'Renal_Allograft':[1,0,1,1]}
    proportion_df,count_df,total_count = density_normalize(dfTemp,areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
    
    #%%
    # Step 2: Filter and focus on the relevant data
    selected_columns = [
         'Bcell','Plasmablasts','plasma_cells','CD4+Foxp3+_Tcell',
                           'CD4+ICOS+PD1+_Tcell','CD4+ICOS+_Tcell','CD4+PD1+_Tcell','CD4+_Tcell','CD4+_intraepithelial_Tcell',
                           'CD8+Foxp3+_Tcell','CD8+ICOS+PD1+_Tcell','CD8+ICOS+_Tcell','CD8+PD1+_Tcell',
                           'CD8+_Tcell','CD8+_intraepithelial_Tcell','CD3+_Tcell','CD4+CD8+_Tcell','TCRgd_Tcell','NK_Tcell','NK_cells','CD14+CD163+MERTK+_macrophages',
                          'CD14+CD163+_macrophages','CD14+MERTK+_macrophages','CD14+_macrophages','CD16+_macrophages','HLAII+_Monocytes',
                          'HLAII-_Monocytes','Mo-Macrophage','Neutrophils','BDCA2pCD103p_DCs','cDC1','cDC2','pDC'
                          ,'Distal_collecting_tubules','Endothelial_cells','Inflamed_tubule','Proximal_tubules','RBCs',
        'AccessionNumber','disease_cohort'
    ]
    
    
    spider_plot(selected_columns,count_df,color_mapping)
    #%%
    selected_columns = [
         'Bcell','Plasmablasts','plasma_cells','CD4+Foxp3+_Tcell',
                           'CD4+ICOS+PD1+_Tcell','CD4+ICOS+_Tcell','CD4+PD1+_Tcell','CD4+_Tcell','CD4+_intraepithelial_Tcell',
                           'CD8+Foxp3+_Tcell','CD8+ICOS+PD1+_Tcell','CD8+ICOS+_Tcell','CD8+PD1+_Tcell',
                           'CD8+_Tcell','CD8+_intraepithelial_Tcell','CD3+_Tcell','CD4+CD8+_Tcell','TCRgd_Tcell','NK_Tcell','NK_cells','CD14+CD163+MERTK+_macrophages',
                          'CD14+CD163+_macrophages','CD14+MERTK+_macrophages','CD14+_macrophages','CD16+_macrophages','HLAII+_Monocytes',
                          'HLAII-_Monocytes','Mo-Macrophage','Neutrophils','BDCA2pCD103p_DCs','cDC1','cDC2','pDC',
        'AccessionNumber','disease_cohort'
    ]
    spider_plot(selected_columns,count_df,color_mapping)
    #%%
    immuneCells= ['Bcell', 'CD14+CD163+MERTK+_macrophages',
           'CD14+CD163+_macrophages', 'CD14+MERTK+_macrophages',
           'CD14+_macrophages', 'CD16+_macrophages', 'CD3+_Tcell',
           'CD4+CD8+_Tcell', 'CD4+Foxp3+_Tcell', 'CD4+ICOS+PD1+_Tcell',
           'CD4+ICOS+_Tcell', 'CD4+PD1+_Tcell', 'CD4+_Tcell', 'CD8+Foxp3+_Tcell',
           'CD8+ICOS+PD1+_Tcell', 'CD8+ICOS+_Tcell', 'CD8+PD1+_Tcell',
           'CD8+_Tcell','CD4+_intraepithelial_Tcell','CD8+_intraepithelial_Tcell' ,'BDCA2pCD103p_DCs',
           'HLAII+_Monocytes', 'HLAII-_Monocytes', 
           'Mo-Macrophage', 'NK_Tcell', 'NK_cells', 'Neutrophils', 'Plasmablasts',
            'TCRgd_Tcell', 'cDC1', 'cDC2', 'pDC',
           'plasma_cells']
    cell_class_ratio_plot_2D(count_df,args.write_dir,Y=['CD14+CD163+_macrophages'],X=immuneCells,
                          ylabel='CD14+CD163+ Macrophages',
                          colors=color_mapping,
                          xlabel='Immune infiltrate',
                          title='CD14+CD163+ v. immune infiltrate (Patient level)',
                          filename='CD14CD163Macrophage_immuneInfiltrate',proportion_df= proportion_df, count_df=count_df)
    
    #%%
    
    cell_class_ratio_plot_2D(count_df,args.write_dir,Y=['CD14+MERTK+_macrophages'],X=immuneCells,
                          ylabel='CD14+MERTK+ Macrophages',
                          colors=color_mapping,
                          xlabel='Immune infiltrate',
                          title='CD14+MERTK+ v. immune infiltrate (Patient level)',
                          filename='CD14MERTK+Macrophage_immuneInfiltrate',proportion_df= proportion_df, count_df=count_df)
    #%%
    
    cell_class_ratio_plot_2D(count_df,args.write_dir,Y=['CD4+_Tcell',
     'CD4+Foxp3+_Tcell',
     'CD4+PD1+_Tcell',
     'CD4+CD8+_Tcell',
     'CD4+ICOS+PD1+_Tcell','CD4+_intraepithelial_Tcell',
     'CD4+ICOS+_Tcell'],X=immuneCells,
                          ylabel='CD4+ T-cells',
                          colors=color_mapping,
                          xlabel='Immune infiltrate',
                          title='CD4+ T cell v. immune infiltrate (Patient level)',
                          filename='CD4T_immuneInfiltrate',proportion_df= proportion_df, count_df=count_df)
    
    #%%
    
   
    cell_class_ratio_plot_2D(count_df,args.write_dir,Y=['CD8+_Tcell',
     'CD8+Foxp3+_Tcell',
     'CD8+PD1+_Tcell',
     'CD4+CD8+_Tcell',
     'CD8+ICOS+PD1+_Tcell','CD8+_intraepithelial_Tcell',
     'CD8+ICOS+_Tcell'],X=immuneCells,
                          ylabel='CD8+ T-cells',
                          colors=color_mapping,
                          xlabel='Immune infiltrate',
                          title='CD8+ T cell v. immune infiltrate (Patient level)',
                          filename='CD8T_immuneInfiltrate',proportion_df= proportion_df, count_df=count_df)
    
    #%%
  
    cell_class_ratio_plot_2D(count_df,args.write_dir,Y=['HLAII+_Monocytes'],X=immuneCells,
                          ylabel='HLAII+ Monocytes',
                          colors=color_mapping,
                          xlabel='Immune infiltrate',
                          title='HLAII+ monocytes v. immune infiltrate (Patient level)',
                          filename='HLAIIpMonocytes_immuneInfiltrate',proportion_df= proportion_df, count_df=count_df)
    
    
    #%%
   
    cell_class_ratio_plot_2D(count_df,args.write_dir,Y=['HLAII-_Monocytes'],X=immuneCells,
                          ylabel='HLAII- Monocytes',
                          colors=color_mapping,
                          xlabel='Immune infiltrate',
                          title='HLAII- monocytes v. immune infiltrate (Patient level)',
                          filename='HLAIInMonocytes_immuneInfiltrate',proportion_df= proportion_df, count_df=count_df)
    
    
    
    #%%lets look at collagen
    mean_df = COLIIIscoreDf.groupby('AccessionNumber')['th_optimal'].mean().reset_index()
    
    proportion_df_colIII = proportion_df.merge(mean_df, on=['AccessionNumber'], how='inner')
    count_df_colIII = count_df.merge(mean_df, on=['AccessionNumber'], how='inner')
    #%%lets look at MXA and IFN signaling 
    mean_df = MXAscoreDf.groupby('AccessionNumber')['th_optimal'].mean().reset_index()
    
    proportion_df_MXA = proportion_df.merge(mean_df, on=['AccessionNumber'], how='inner')
    count_df_MXA = count_df.merge(mean_df, on=['AccessionNumber'], how='inner')
    
    
    #%%
    


    # Define groups of AccessionNumbers
    groups = [
        ["S18-11155",'S22-29131','S18-23952','S21-6213','S22-20909','S22-5488'],
        ['S21-11744','S21-29302','S21-16420','S21-20619','S21-18286','S20-30434'],
        ['S21-4590','S22-17533','S22-23708','S21-21884','S19-8368','S22-16103','S19-20137'],
        
        ['S22-14977','S21-9639','S21-38060','S21-32478','S21-30119','S22-1624']
    ]
    #%%
    # Call the function
    plot_grouped_dot_plot(count_df_colIII, groups,args.write_dir+'LuN_colIII_score_dotplot.png','th_optimal','COLIII',True)
    plot_grouped_dot_plot(count_df_MXA, groups,args.write_dir+'LuN_MXA_score_dotplot.png','th_optimal','MXA',True)
    plot_grouped_dot_plot(count_df, groups,args.write_dir+'LuN_Inflamed_tubules_dotplot.png','Inflamed_tubule','Inflamed_tubule',True)
 
    plot_grouped_dot_plot(count_df_colIII, groups,args.write_dir+'LuN_colIII_score_dotplot_noLabels.png','th_optimal','COLIII',False)
    plot_grouped_dot_plot(count_df_MXA, groups,args.write_dir+'LuN_MXA_score_dotplot_noLabels.png','th_optimal','MXA',False)
    plot_grouped_dot_plot(count_df, groups,args.write_dir+'LuN_Inflamed_tubules_dotplot_noLabels.png','Inflamed_tubule','Inflamed_tubule',False)
  
    
    holdDf = count_df.copy()
    
    immuneCells=['Bcell',
     'CD14+CD163+_macrophages',
     'CD14+MERTK+_macrophages',
     'CD14+_macrophages',
     'CD16+_macrophages',
     'CD3+_Tcell',
     'CD4+CD8+_Tcell',
     'CD4+Foxp3+_Tcell',
     'CD4+ICOS+PD1+_Tcell',
     'CD4+ICOS+_Tcell',
     'CD4+PD1+_Tcell',
     'CD4+_Tcell',
     'CD8+Foxp3+_Tcell',
     'CD8+ICOS+PD1+_Tcell',
     'CD8+ICOS+_Tcell',
     'CD8+PD1+_Tcell',
     'CD8+_Tcell',
     'CD4+_intraepithelial_Tcell',
     'CD8+_intraepithelial_Tcell',
     'BDCA2pCD103p_DCs',
     'HLAII+_Monocytes',
     'HLAII-_Monocytes',
     'Mo-Macrophage',
     'NK_Tcell',
     'NK_cells',
     'Neutrophils',
     'Plasmablasts',
     'TCRgd_Tcell',
     'cDC1',
     'cDC2',
     'pDC',
     'plasma_cells']
    
    holdDf['Total_ImmuneCells'] = holdDf[immuneCells].sum(axis=1) 
    
    plot_grouped_dot_plot(holdDf, groups,args.write_dir+'LuN_ImmuneCells_dotplot.png','Total_ImmuneCells','Immune Cells',True)
    plot_grouped_dot_plot(holdDf, groups,args.write_dir+'LuN_ImmuneCells_dotplot_noLabels.png','Total_ImmuneCells','Immune Cells',False)
    
 
        
    #%%
    
    result_df = pd.merge(holdDf, count_df_colIII[['AccessionNumber', 'th_optimal']], on='AccessionNumber', how='left')
    plot_groups_with_ols_stats(result_df, 'Total_ImmuneCells', 'th_optimal', groups,'COLIII',args.write_dir+'COLIII_totalImmune_stratified.png',True)
    plot_groups_with_ols_stats(result_df, 'Total_ImmuneCells', 'th_optimal', groups,'COLIII',args.write_dir+'COLIII_totalImmune_stratified_noLabels.png',False)
   
    #%%
    plot_groups_with_ols_stats(result_df, 'Total_ImmuneCells', 'Inflamed_tubule', groups,'Inflamed tubules',args.write_dir+'InflamedTubules_totalImmune_stratified.png',True)
    plot_groups_with_ols_stats(result_df, 'Total_ImmuneCells', 'Inflamed_tubule', groups,'Inflamed tubules',args.write_dir+'InflamedTubules_totalImmune_stratified_noLabels.png',False)
    #%%
    result_df = pd.merge(holdDf, count_df_MXA[['AccessionNumber', 'th_optimal']], on='AccessionNumber', how='left')
    plot_groups_with_ols_stats(result_df, 'Total_ImmuneCells', 'th_optimal', groups,'MXA',args.write_dir+'MXA_totalImmune_stratified.png',True)
    plot_groups_with_ols_stats(result_df, 'Total_ImmuneCells', 'th_optimal', groups,'MXA',args.write_dir+'MXA_totalImmune_stratified_noLabels.png',False)   
    #%%
    cell_class_ratio_plot_2D(count_df_MXA,args.write_dir,Y=['CD14+CD163+_macrophages'],X=['th_optimal'],
                          ylabel='CD14+CD163+ Macrophages',
                          colors=color_mapping,
                          xlabel='MXA score',
                          title='CD14+CD163+ v. MXA (Patient level)',
                          filename='CD14CD163Macrophage_MXA_score',proportion_df= proportion_df_MXA, count_df=count_df_MXA)
    
    #%%
    
    cell_class_ratio_plot_2D(count_df_MXA,args.write_dir,Y=['CD14+MERTK+_macrophages'],X=['th_optimal'],
                          ylabel='CD14+MerTk+ Macrophages',
                          colors=color_mapping,
                          xlabel='MXA score',
                          title='CD14+MerTk+ v. MXA  (Patient level)',
                          filename='CD14MerTkMacrophage_MXA_score', proportion_df= proportion_df_MXA ,count_df=count_df_MXA)
    
    #%%
    cell_class_ratio_plot_2D(count_df_MXA,args.write_dir,Y=['CD8+_Tcell',
     'CD8+Foxp3+_Tcell',
     'CD8+PD1+_Tcell',
     'CD4+CD8+_Tcell',
     'CD8+ICOS+PD1+_Tcell','CD8+_intraepithelial_Tcell',
     'CD8+ICOS+_Tcell'],X=['th_optimal'],
                          ylabel='CD8+ T cell',
                          colors=color_mapping,
                          xlabel='MXA score',
                          title='CD8+ T-cell v MXA (Patient level)',
                          filename='CD8Tcell_MXA_score', proportion_df= proportion_df_MXA ,count_df=count_df_MXA)
    
    #%%
    
    
    cell_class_ratio_plot_2D(count_df_MXA,args.write_dir,Y=['CD4+_Tcell',
     'CD4+Foxp3+_Tcell',
     'CD4+PD1+_Tcell',
     'CD4+CD8+_Tcell',
     'CD4+ICOS+PD1+_Tcell','CD4+_intraepithelial_Tcell',
     'CD4+ICOS+_Tcell'],X=['th_optimal'],
                          ylabel='CD4+ T cell',
                          colors=color_mapping,
                          xlabel='MXA score',
                          title='CD4+ T-cell v MXA (Patient level)',
                          filename='CD4Tcell_MXA_score', proportion_df= proportion_df_MXA ,count_df=count_df_MXA)
    
    #%%
    
    cell_class_ratio_plot_2D(count_df_MXA,args.write_dir,Y=['HLAII-_Monocytes'],X=['th_optimal'],
                          ylabel='HLAII- Monocytes cell',
                          colors=color_mapping,
                          xlabel='MXA score',
                          title='HLAII- Monocytes v MXA (Patient level)',
                          filename='HLAIInMonocytes_MXA_score', proportion_df= proportion_df_MXA ,count_df=count_df_MXA)
    
    #%%
    cell_class_ratio_plot_2D(count_df_MXA,args.write_dir,Y=['HLAII+_Monocytes'],X=['th_optimal'],
                          ylabel='HLAII+ Monocytes cell',
                          colors=color_mapping,
                          xlabel='MXA score',
                          title='HLAII+ Monocytes v MXA (Patient level)',
                          filename='HLAIIpMonocytes_MXA_score', proportion_df= proportion_df_MXA ,count_df=count_df_MXA)
    
    
    
    
    
    
    #%%%
    
    cell_class_ratio_plot_2D(count_df_colIII,args.write_dir,Y=['CD14+CD163+_macrophages'],X=['th_optimal'],
                          ylabel='CD14+CD163+ Macrophages',
                          colors=color_mapping,
                          xlabel='COLIII score',
                          title='CD14+CD163+ Ma v COLIII (Patient level)',
                          filename='CD14CD163Macrophage_collagen_score', proportion_df= proportion_df_colIII ,count_df=count_df_colIII)
    
    #%%
    
    cell_class_ratio_plot_2D(count_df_colIII,args.write_dir,Y=['CD14+MERTK+_macrophages'],X=['th_optimal'],
                          ylabel='CD14+MerTk+ Macrophages',
                          colors=color_mapping,
                          xlabel='COLIII score',
                          title='CD14+MerTk+ Ma v COLIII (Patient level)',
                          filename='CD14MerTkMacrophage_collagen_score', proportion_df= proportion_df_colIII ,count_df=count_df_colIII)
    
    #%%
    cell_class_ratio_plot_2D(count_df_colIII,args.write_dir,Y=['CD8+_Tcell',
     'CD8+Foxp3+_Tcell',
     'CD8+PD1+_Tcell',
     'CD4+CD8+_Tcell',
     'CD8+ICOS+PD1+_Tcell','CD8+_intraepithelial_Tcell',
     'CD8+ICOS+_Tcell'],X=['th_optimal'],
                          ylabel='CD8+ T-cells',
                          colors=color_mapping,
                          xlabel='COLIII score',
                          title='CD8+ T-cell v COLIII (Patient level)',
                          filename='CD8Tcell_collagen_score', proportion_df= proportion_df_colIII ,count_df=count_df_colIII)
    
    
    #%%
    
    
    cell_class_ratio_plot_2D(count_df_colIII,args.write_dir,Y=['CD4+_Tcell',
     'CD4+Foxp3+_Tcell',
     'CD4+PD1+_Tcell',
     'CD4+CD8+_Tcell',
     'CD4+ICOS+PD1+_Tcell','CD4+_intraepithelial_Tcell',
     'CD4+ICOS+_Tcell'],X=['th_optimal'],
                          ylabel='CD4+ T-cells',
                          colors=color_mapping,
                          xlabel='COLIII score',
                          title='CD4+ T-cell v COLIII (Patient level)',
                          filename='CD4Tcell_collagen_score', proportion_df= proportion_df_colIII ,count_df=count_df_colIII)
    
    #%%
    
    cell_class_ratio_plot_2D(count_df_colIII,args.write_dir,Y=['HLAII+_Monocytes'],X=['th_optimal'],
                          ylabel='HLAII+ Monocytes T-cells',
                          colors=color_mapping,
                          xlabel='COLIII score',
                          title='HLAII+ Monocytes v COLIII (Patient level)',
                          filename='HLAIIpMonocytes_collagen_score', proportion_df= proportion_df_colIII ,count_df=count_df_colIII)
    
    #%%
    cell_class_ratio_plot_2D(count_df_colIII,args.write_dir,Y=['HLAII-_Monocytes'],X=['th_optimal'],
                          ylabel='HLAII- Monocytes T-cells',
                          colors=color_mapping,
                          xlabel='COLIII score',
                          title='HLAII- Monocytes v COLIII (Patient level)',
                          filename='HLAIInMonocytes_collagen_score', proportion_df= proportion_df_colIII ,count_df=count_df_colIII)
    
    
    
    #%%%
    
    cell_class_ratio_plot_2D(dfTemp,args.write_dir,Y=['CD14+CD163+_macrophages'],X=['Inflamed_tubule'],
                          ylabel='CD14+CD163+ Macrophages',
                          colors=color_mapping,
                          xlabel='Inflamed tubules',
                          title='CD14+CD163+ v inflamed tubule (Patient level)',
                          filename='CD14CD163Macrophage_InflamedTubules', proportion_df= proportion_df,count_df=count_df)
    
    #%%
    
    cell_class_ratio_plot_2D(dfTemp,args.write_dir,Y=['CD14+MERTK+_macrophages'],X=['Inflamed_tubule'],
                          ylabel='CD14+MerTk+ Macrophages',
                          colors=color_mapping,
                          xlabel='Inflamed tubules',
                          title='CD14+MerTk+ v inflamed tubule (Patient level)',
                          filename='CD14MerTkMacrophage_InflamedTubules', proportion_df= proportion_df,count_df=count_df)
    
    #%%
    cell_class_ratio_plot_2D(dfTemp,args.write_dir,Y=['CD8+_Tcell',
     'CD8+Foxp3+_Tcell',
     'CD8+PD1+_Tcell',
     'CD4+CD8+_Tcell','CD8+_intraepithelial_Tcell',
     'CD8+ICOS+PD1+_Tcell',
     'CD8+ICOS+_Tcell'],X=['Inflamed_tubule'],
                          ylabel='CD8+ T-cells',
                          colors=color_mapping,
                          xlabel='Inflamed tubules',
                          title='CD8+ T v. inflamed (Patient level)',
                          filename='CD8_InflamedTubules', proportion_df= proportion_df,count_df=count_df)
   #%%
   
    cell_class_ratio_plot_2D(dfTemp,args.write_dir,Y=['CD4+_Tcell',
    'CD4+Foxp3+_Tcell',
    'CD4+PD1+_Tcell',
    'CD4+CD8+_Tcell',
    'CD4+ICOS+PD1+_Tcell','CD4+_intraepithelial_Tcell',
    'CD4+ICOS+_Tcell'],X=['Inflamed_tubule'],
                         ylabel='CD4+ T-cells',
                         colors=color_mapping,
                         xlabel='Inflamed tubules',
                         title='CD4+ T v. inflamed (Patient level)',
                         filename='CD4_InflamedTubules', proportion_df= proportion_df,count_df=count_df)
  
    
    #%%
    cell_class_ratio_plot_2D(dfTemp,args.write_dir,Y=['HLAII-_Monocytes'],X=['Inflamed_tubule'],
                         ylabel='HLAII- Monocytes',
                         colors=color_mapping,
                         xlabel='Inflamed tubules',
                         title='HLAII- Monocytes v. inflamed (Patient level)',
                         filename='HLAIInMonocytes_InflamedTubules', proportion_df= proportion_df,count_df=count_df)
  #%%
    cell_class_ratio_plot_2D(dfTemp,args.write_dir,Y=['HLAII+_Monocytes'],X=['Inflamed_tubule'],
                         ylabel='HLAII+ Monocytes',
                         colors=color_mapping,
                         xlabel='Inflamed tubules',
                         title='HLAII+ Monocytes v. inflamed (Patient level)',
                         filename='HLAIIpMonocytes_InflamedTubules', proportion_df= proportion_df,count_df=count_df)
  
     
    #%%

    # # Define a dictionary to map disease cohorts to markers
    color_mapping = {'Normal_Kidney':[0,1,0,1],'Lupus_Nephritis':[0,0,1,1], 'Renal_Allograft':[1,0,1,1]}
    markers = {'Lupus_Nephritis': 's', 'Normal_Kidney': 'o', 'Renal_Allograft': '^'}  # Adjust as per your unique values in disease_cohort
    returned = cell_class_ratio_plot_3D(dfTemp,args.write_dir,Y=['CD14+CD163+_macrophages'],X=['CD14+MERTK+_macrophages'],
                          Z=['CD8+Foxp3+_Tcell','CD8+ICOS+PD1+_Tcell','CD8+ICOS+_Tcell','CD8+_Tcell','CD8+PD1+_Tcell','CD8+_intraepithelial_Tcell'],
                          ylabel='CD14+CD163+ Macrophages',
                          markers=markers,
                          column='disease_cohort',
                          xlabel='C CD14+MerTK Macrophages',
                          zlabel='Proportion of CD8 T cells',
                          title='Ratio of 3 cell classes (Patient level)',
                          filename='CD14CD163Macrophage_CD14MerTk_CD8Tcells',merged_df=count_df,color_mapping=color_mapping,sort_column='Sum_Z')
    
    color_mappingTT= {'NK': [0, 1, 0, 1],
 'LuN': [0, 0, 1, 1],
 'MR': [1, 0.8, 0.3, 1],
 'TCMR':[0,1,1,1]}
    returned = cell_class_ratio_plot_3D(dfTemp,args.write_dir,Y=['CD14+CD163+_macrophages'],X=['CD14+MERTK+_macrophages'],
                          Z=['CD8+Foxp3+_Tcell','CD8+ICOS+PD1+_Tcell','CD8+ICOS+_Tcell','CD8+_Tcell','CD8+PD1+_Tcell','CD8+_intraepithelial_Tcell'],
                          ylabel='CD14+CD163+ Macrophages',
                          markers=markers,
                          column='TissueType',
                          xlabel='C CD14+MerTK Macrophages',
                          zlabel='Proportion of CD8 T cells',
                          title='Ratio of 3 cell classes (Patient level)',
                          filename='CD14CD163Macrophage_CD14MerTk_CD8Tcells_TissueType',merged_df=count_df,color_mapping=color_mappingTT,sort_column='Sum_Z')
    #%%
    
    
    T_cells = ['CD4+ICOS+_Tcell','CD4+PD1+_Tcell','CD4+Foxp3+_Tcell',
               'CD4+ICOS+PD1+_Tcell','CD8+Foxp3+_Tcell','CD8+ICOS+PD1+_Tcell','CD8+ICOS+_Tcell',
               'CD8+PD1+_Tcell','CD3+_Tcell','CD4+_Tcell','NK_Tcell','CD8+_Tcell','TCRgd_Tcell',
               'CD4+CD8+_Tcell','CD8+_intraepithelial_Tcell','CD4+_intraepithelial_Tcell','NK_cells']
    Myeloid_cells =['CD14+CD163+_macrophages','CD14+MERTK+_macrophages','CD14+CD163+MERTK+_macrophages',
                   'HLAII+_Monocytes','HLAII-_Monocytes','Mo-Macrophage','CD16+_macrophages',
                   'pDC','cDC2','cDC1','CD14+_macrophages','Neutrophils']
    Humoral_cells =['Bcell','plasma_cells','Plasmablasts']
 
    
    returned = cell_class_ratio_plot_3D(dfTemp,args.write_dir,X=T_cells,Y=Myeloid_cells,
                          Z=Humoral_cells,
                          xlabel='T cells',
                          markers=markers,
                          column='disease_cohort',
                          ylabel='Myeloid cells',
                          zlabel='Humoral cells',
                          title='Ratio of 3 cell classes (Patient level)',
                          filename='Tcells_myeloid_humoral',merged_df=count_df,color_mapping=color_mapping,sort_column='Sum_Z')
  
    
    returned = cell_class_ratio_plot_3D(dfTemp,args.write_dir,X=T_cells,Y=Myeloid_cells,
                          Z=Humoral_cells,
                          xlabel='T cells',
                          markers=markers,
                          column='TissueType',
                          ylabel='Myeloid cells',
                          zlabel='Humoral cells',
                          title='Ratio of 3 cell classes (Patient level)',
                          filename='Tcells_myeloid_humoral_TissueType',merged_df=count_df,color_mapping=color_mappingTT,sort_column='Sum_Z')
  
    
   
    #%%
 
    
    
    class_labels =['Bcell','Plasmablasts','plasma_cells','CD4+Foxp3+_Tcell',
                      'CD4+ICOS+PD1+_Tcell','CD4+ICOS+_Tcell','CD4+PD1+_Tcell','CD4+_Tcell','CD4+_intraepithelial_Tcell',
                      'CD8+Foxp3+_Tcell','CD8+ICOS+PD1+_Tcell','CD8+ICOS+_Tcell','CD8+PD1+_Tcell',
                      'CD8+_Tcell','CD8+_intraepithelial_Tcell','CD3+_Tcell','CD4+CD8+_Tcell','TCRgd_Tcell','NK_Tcell','NK_cells','CD14+CD163+MERTK+_macrophages',
                     'CD14+CD163+_macrophages','CD14+MERTK+_macrophages','CD14+_macrophages','CD16+_macrophages','HLAII+_Monocytes',
                     'HLAII-_Monocytes','Mo-Macrophage','Neutrophils','BDCA2pCD103p_DCs','cDC1','cDC2','pDC', 'Distal_collecting_tubules',
                     'Endothelial_cells','Inflamed_tubule','Proximal_tubules','RBCs']
    
    
    for cohort in np.unique(dfTemp['disease_cohort']):
        print('Cohort',cohort)
        proportion_df_cohort,count_df_cohort,total_count_cohort = density_normalize(dfTemp[dfTemp['disease_cohort']==cohort],areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
        correlation_heatmap_plot(dfTemp,args.write_dir,f'class_frequency_correlation_all_expanded_{cohort}.png',class_labels=class_labels,proportion_df=proportion_df_cohort,count_df=count_df_cohort)

    #%%
    
    for cohort in np.unique(dfTemp['TissueType']):
        print('Cohort',cohort)
        proportion_df_cohort,count_df_cohort,total_count_cohort = density_normalize(dfTemp[dfTemp['TissueType']==cohort],areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
        correlation_heatmap_plot(dfTemp,args.write_dir,f'class_frequency_correlation_all_expanded_{cohort}.png',class_labels=class_labels,proportion_df=proportion_df_cohort,count_df=count_df_cohort)
    

    
    #%%Marcus really wants these plots
    immuneDf = dfTemp[~dfTemp['class_label'].isin(['Endothelial_cells','Distal_collecting_tubules',
                                                   'Inflamed_tubule','Proximal_tubules', 'RBCs'])].copy()
    immuneDf['class_label'].replace({'CD4+ICOS+_Tcell':'Tcell', 
                                    'CD4+PD1+_Tcell':'Tcell',
                                    'CD4+Foxp3+_Tcell':'Tcell', 
                                    'CD4+ICOS+PD1+_Tcell':'Tcell',
                                    'CD8+Foxp3+_Tcell':'Tcell', 
                                    'CD8+ICOS+PD1+_Tcell':'Tcell', 
                                    'CD8+ICOS+_Tcell':'Tcell',
                                    'CD8+PD1+_Tcell':'Tcell',
                                    'CD14+CD163+_macrophages':'Myeloid', 
                                    'CD14+MERTK+_macrophages':'Myeloid',
                                    'CD14+CD163+MERTK+_macrophages':'Myeloid',
                                    'HLAII+_Monocytes':'Myeloid', 
                                    'HLAII-_Monocytes':'Myeloid',
                                    'CD3+_Tcell':'Tcell',
                                    'CD4+_Tcell':'Tcell',
                                    'NK_Tcell':'Tcell',
                                    'CD8+_Tcell':'Tcell',
                                    'TCRgd_Tcell':'Tcell',
                                    'Bcell':'Humoral',
                                     'CD4+CD8+_Tcell':'Tcell',
                                     'Mo-Macrophage':'Myeloid',
                                     'plasma_cells':'Humoral', 
                                     'CD16+_macrophages':'Myeloid',
                                     'Plasmablasts':'Humoral',
                                     'cDC1':'Myeloid',
                                     'NK_cells':'Myeloid',
                                     'pDC':'Myeloid',
                                     'cDC2':'Myeloid',
                                     'CD14+_macrophages':'Myeloid',
                                     'Neutrophils':'Myeloid'}, inplace=True)
    print(immuneDf['class_label'].unique())
    # Define a dictionary to map disease cohorts to markers
    markers = {'Lupus_Nephritis': 's', 'Normal_Kidney': 'o', 'Renal_Allograft': '^'}  # Adjust as per your unique values in disease_cohort
    
    proportion_df,count_df,total_count = density_normalize(immuneDf,areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
#%%
    T_Humoral_myeloid_plots(merged_df=count_df,total_count=total_count,writeDir=args.write_dir,X='Tcell',Y='Myeloid',Z=['Humoral'],markers=markers,colors=color_mapping,color_mapping=color_mapping,column='disease_cohort',filename='.tiff')
    
#%%
#['LuN', 'MR', 'TCMR', 'NK']
    color_mappingTT= {'NK': [0, 1, 0, 1],
 'LuN': [0, 0, 1, 1],
 'MR': [1, 0.8, 0.3, 1],
 'TCMR':[0,1,1,1]}
    T_Humoral_myeloid_plots(merged_df=count_df,total_count=total_count,writeDir=args.write_dir,X='Tcell',Y='Myeloid',Z=['Humoral'],markers=markers,
                            colors=color_mappingTT,color_mapping=color_mappingTT,
                            column='TissueType',filename='_TissueType.tiff')    
    
     #%%
    proportion_df,count_df,total_count = density_normalize(dfTemp,areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
    cell_class_vs_total_df(args.write_dir,cell_class=['CD14+CD163+_macrophages'],markers=markers,
                                colors=color_mapping,df=count_df,total_count=total_count,filename='CD14+CD163+_macrophage_vs_total_density_all_cohorts.tif')
  #%%
    cell_class_vs_total_df(args.write_dir,cell_class=['CD14+CD163+_macrophages'],markers=markers,
                                colors=color_mapping,df=proportion_df,total_count=total_count,filename='CD14+CD163+_macrophage_vs_total_proportion_all_cohorts.tif')
  
    
    
    
     #%%
    for cohort in np.unique(dfTemp['disease_cohort']):
        print('Cohort',cohort)
        proportion_df,count_df,total_count = density_normalize(dfTemp[dfTemp['disease_cohort']==cohort],areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
        cell_class_vs_total_df(args.write_dir,cell_class=['CD14+CD163+_macrophages'],markers=markers,colors=color_mapping,
                                    total_count=total_count,df=count_df,
                                    filename=f'vs_total_density_{cohort}.tif')
        
        #%%
    
    
    immuneDf = dfTemp[~dfTemp['class_label'].isin(['Endothelial_cells','Distal_collecting_tubules',
                                                   'Inflamed_tubule','Proximal_tubules', 'RBCs'])].copy()

    immuneDf['class_label'].replace({
                                    'CD8+Foxp3+_Tcell':'CD8+_Tcell', 
                                    'CD8+ICOS+PD1+_Tcell':'CD8+_Tcell', 
                                    'CD8+ICOS+_Tcell':'CD8+_Tcell',
                                    'CD8+PD1+_Tcell':'CD8+_Tcell',
                                    }, inplace=True)

    proportion_df,count_df,total_count = density_normalize(immuneDf,areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
    
    #%%
    sorted_df = count_df[count_df['disease_cohort']=='Lupus_Nephritis'][['CD14+CD163+_macrophages','AccessionNumber']].sort_values(by='CD14+CD163+_macrophages')
    sorted_df = pd.merge(sorted_df, total_count, on='AccessionNumber',how='inner').to_csv(args.write_dir+"CD14+CD163+densities_LN.csv", index=False)
    
    
    #%%
    cell_class_vs_total_df(args.write_dir,cell_class=['CD14+CD163+_macrophages'],markers=markers,
                                colors=color_mapping,df=count_df,total_count=total_count,filename='CD14+CD163+_macrophages_vs_total_density_all_cohorts_immune_only.tif')
  #%%
    cell_class_vs_total_df(args.write_dir,cell_class=['CD14+CD163+_macrophages'],markers=markers,
                                colors=color_mapping,df=proportion_df,total_count=total_count,filename='CD14+CD163+_macrophages_vs_total_proportion_all_cohorts_immune_only.tif')
  
    #%%
    cell_class_vs_total_df(args.write_dir,cell_class=['CD14+MERTK+_macrophages'],markers=markers,
                                colors=color_mapping,df=count_df,total_count=total_count,filename='CD14+MERTK+_macrophages_vs_total_density_all_cohorts_immune_only.tif')
    
    #%%
    cell_class_vs_total_df(args.write_dir,cell_class=['CD14+MERTK+_macrophages'],markers=markers,
                                colors=color_mapping,df=proportion_df,total_count=total_count,filename='CD14+MERTK+_macrophages_vs_total_proportion_all_cohorts_immune_only.tif')
    #%%

    cell_class_vs_total_df(args.write_dir,cell_class=['CD8+_Tcell'],markers=markers,
                                colors=color_mapping,df=count_df,total_count=total_count,filename='CD8+_Tcell_vs_total_density_all_cohorts_immune_only.tif')
    
    
    #%%
    cell_class_vs_total_df(args.write_dir,cell_class=['CD8+_Tcell'],markers=markers,
                                colors=color_mapping,df=proportion_df,total_count=total_count,filename='CD8+_Tcell_vs_total_proportion_all_cohorts_immune_only.tif')
    
    
       
        #%%
    for cohort in np.unique(dfTemp['disease_cohort']):
        print('Cohort',cohort)
        proportion_df,count_df,total_count = density_normalize(immuneDf[immuneDf['disease_cohort']==cohort],areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
        cell_class_vs_total_df(args.write_dir,cell_class=['CD14+CD163+_macrophages'],markers=markers,colors=color_mapping,
                                    total_count=total_count,df=count_df,
                                    filename=f'vs_total_density_{cohort}_immune_only.tif')
     
        
        
        #%%
    for cohort in np.unique(dfTemp['disease_cohort']):
        print('Cohort',cohort)
        proportion_df,count_df,total_count = density_normalize(immuneDf[immuneDf['disease_cohort']==cohort],areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
        cell_class_vs_total_df(args.write_dir,cell_class=['CD14+CD163+_macrophages'],markers=markers,colors=color_mapping,
                                    total_count=total_count,df=proportion_df,
                                    filename=f'vs_total_proportion_{cohort}_immune_only.tif')    
       
        
    #%%
    proportion_df,count_df,total_count = density_normalize(dfTemp,areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
    class_labels =sorted(dfTemp['class_label'].value_counts().index)
    correlation_heatmap_plot(dfTemp,args.write_dir,'class_frequency_correlation_all_expanded.tiff',class_labels=class_labels,proportion_df=proportion_df,count_df=count_df)
    
    
    #%%
    for cohort in np.unique(dfTemp['TissueType']):
        print('Cohort',cohort)
        proportion_df,count_df,total_count = density_normalize(dfTemp[dfTemp['TissueType']==cohort],areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
        correlation_heatmap_plot(dfTemp[dfTemp['TissueType']==cohort],args.write_dir,f'class_frequency_correlation_{cohort}_expanded.tiff',class_labels=class_labels,count_df=count_df,proportion_df=proportion_df)

    #%%
    for cohort in np.unique(dfTemp['disease_cohort']):
        print('Cohort',cohort)
        proportion_df,count_df,total_count = density_normalize(dfTemp[dfTemp['disease_cohort']==cohort],areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
        correlation_heatmap_plot(dfTemp[dfTemp['disease_cohort']==cohort],args.write_dir,f'class_frequency_correlation_{cohort}_expanded.tiff',class_labels=class_labels,count_df=count_df,proportion_df=proportion_df)

    #%%
    
    immuneDf = dfTemp[~dfTemp['class_label'].isin(['Endothelial_cells','Distal_collecting_tubules',
                                                   'Inflamed_tubule','Proximal_tubules', 'RBCs'])]
    proportion_df,count_df,total_count = density_normalize(immuneDf,areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
    class_labels =sorted(immuneDf['class_label'].value_counts().index)
    #%%
    correlation_heatmap_plot(immuneDf,args.write_dir,'class_frequency_correlation_immune_only.tiff',class_labels=class_labels,count_df=count_df,proportion_df=proportion_df)
    
    
    #%%
    for cohort in np.unique(dfTemp['TissueType']):
        print('Cohort',cohort)
        proportion_df,count_df,total_count = density_normalize(immuneDf[immuneDf['TissueType']==cohort],areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
        correlation_heatmap_plot(immuneDf[immuneDf['TissueType']==cohort],args.write_dir,f'class_frequency_correlation_{cohort}_immune_only.tiff',class_labels=class_labels,count_df=count_df,proportion_df=proportion_df)

    #%%
    for cohort in np.unique(dfTemp['disease_cohort']):
        print('Cohort',cohort)
        proportion_df,count_df,total_count = density_normalize(immuneDf[immuneDf['disease_cohort']==cohort],areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
        correlation_heatmap_plot(immuneDf[immuneDf['disease_cohort']==cohort],args.write_dir,f'class_frequency_correlation_{cohort}_immune_only.tiff',class_labels=class_labels,count_df=count_df,proportion_df=proportion_df)
#%%


    immuneDf['class_label'].replace({'CD4+ICOS+_Tcell':'CD4+_Tcell', 
                                    'CD4+PD1+_Tcell':'CD4+_Tcell',
                                    'CD4+Foxp3+_Tcell':'CD4+_Tcell', 
                                    'CD4+ICOS+PD1+_Tcell':'CD4+_Tcell',
                                    'CD8+Foxp3+_Tcell':'CD8+_Tcell', 
                                    'CD8+ICOS+PD1+_Tcell':'CD8+_Tcell', 
                                    'CD8+ICOS+_Tcell':'CD8+_Tcell',
                                    'CD8+PD1+_Tcell':'CD8+_Tcell',
                                    'CD14+CD163+_macrophages':'CD14+_macrophages', 
                                    'CD14+MERTK+_macrophages':'CD14+_macrophages',
                                    'HLAII+_Monocytes':'Monocytes', 
                                    'HLAII-_Monocytes':'Monocytes'}, inplace=True)
#%%

    proportion_df,count_df,total_count = density_normalize(immuneDf,areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
    class_labels =sorted(immuneDf['class_label'].value_counts().index)
    correlation_heatmap_plot(immuneDf,args.write_dir,'class_frequency_correlation_immune_only_short.tiff',class_labels=class_labels,count_df=count_df,proportion_df=proportion_df)
#%%
    for cohort in np.unique(dfTemp['TissueType']):
        print('Cohort',cohort)
        proportion_df,count_df,total_count = density_normalize(immuneDf[immuneDf['TissueType']==cohort],areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
        correlation_heatmap_plot(immuneDf[immuneDf['TissueType']==cohort],args.write_dir,f'class_frequency_correlation_{cohort}_immune_only_short.tiff',class_labels=class_labels,count_df=count_df,proportion_df=proportion_df)

#%%

    for cohort in np.unique(dfTemp['disease_cohort']):
        print('Cohort',cohort)
        
        proportion_df,count_df,total_count = density_normalize(immuneDf[immuneDf['disease_cohort']==cohort],areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
        correlation_heatmap_plot(immuneDf[immuneDf['disease_cohort']==cohort],args.write_dir,f'class_frequency_correlation_{cohort}_immune_only_short.tiff',class_labels=class_labels,count_df=count_df,proportion_df=proportion_df)

    #%%
    dfTemp = df.copy()
    
    class_order_plot=['Bcell','Plasmablasts','plasma_cells','CD4+Foxp3+_Tcell',
                      'CD4+ICOS+PD1+_Tcell','CD4+ICOS+_Tcell','CD4+PD1+_Tcell','CD4+_Tcell','CD4+_intraepithelial_Tcell',
                      'CD8+Foxp3+_Tcell','CD8+ICOS+PD1+_Tcell','CD8+ICOS+_Tcell','CD8+PD1+_Tcell',
                      'CD8+_Tcell','CD8+_intraepithelial_Tcell','CD3+_Tcell','CD4+CD8+_Tcell','TCRgd_Tcell','NK_Tcell','NK_cells','CD14+CD163+MERTK+_macrophages',
                     'CD14+CD163+_macrophages','CD14+MERTK+_macrophages','CD14+_macrophages','CD16+_macrophages','HLAII+_Monocytes',
                     'HLAII-_Monocytes','Mo-Macrophage','Neutrophils','BDCA2pCD103p_DCs','cDC1','cDC2','pDC', 'Distal_collecting_tubules',
                     'Endothelial_cells','Inflamed_tubule','Proximal_tubules','RBCs','general_unknown']
    
    
    colors= stacked_bar_plot_cell_class(dfTemp=dfTemp,writeDir=args.write_dir,filename='class_label_barplot_stacked_all.tiff',
                                        colorDict=colorDict,unique_labels=[x for x in class_order_plot if x in dfTemp['class_label'].unique()])
    dfTemp['class_label'].replace({'CD4+ICOS+_Tcell':'CD4+_Tcell', 
                                        'CD4+PD1+_Tcell':'CD4+_Tcell',
                                        'CD4+Foxp3+_Tcell':'CD4+_Tcell', 
                                        'CD4+ICOS+PD1+_Tcell':'CD4+_Tcell',
                                        'CD14+CD163+MERTK+_macrophages':'CD14+CD163+_macrophages',
                                        'CD8+Foxp3+_Tcell':'CD8+_Tcell', 
                                        'CD8+ICOS+PD1+_Tcell':'CD8+_Tcell', 
                                        'CD8+ICOS+_Tcell':'CD8+_Tcell',
                                        'CD8+PD1+_Tcell':'CD8+_Tcell'}, inplace=True)
    colors= stacked_bar_plot_cell_class(dfTemp=dfTemp,writeDir=args.write_dir,filename='class_label_barplot_stacked_simplified.tiff',
                                        colorDict=colorDict,unique_labels=[x for x in class_order_plot if x in dfTemp['class_label'].unique()])
    
    
    #%%
    dfTemp['class_label'].replace({'CD4+ICOS+_Tcell':'CD4+_Tcell', 
                                        'CD4+PD1+_Tcell':'CD4+_Tcell',
                                        'CD4+Foxp3+_Tcell':'CD4+_Tcell', 
                                        'CD4+ICOS+PD1+_Tcell':'CD4+_Tcell',
                                        'CD8+Foxp3+_Tcell':'CD8+_Tcell', 
                                        'CD8+ICOS+PD1+_Tcell':'CD8+_Tcell', 
                                        'CD8+ICOS+_Tcell':'CD8+_Tcell',
                                        'CD8+PD1+_Tcell':'CD8+_Tcell',
                                        'CD14+CD163+_macrophages':'CD14+_macrophages', 
                                        'CD14+MERTK+_macrophages':'CD14+_macrophages',
                                        'HLAII+_Monocytes':'Monocytes', 
                                        'HLAII-_Monocytes':'Monocytes'}, inplace=True)
    
    
    #%%
    
    proportion_df,count_df,total_count = density_normalize(dfTemp,areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
    class_labels =sorted(dfTemp['class_label'].value_counts().index)
    
    correlation_heatmap_plot(dfTemp,args.write_dir,'class_frequency_correlation_all_short.tiff',class_labels=class_labels,count_df=count_df,proportion_df=proportion_df)
    
    #%%
    for cohort in np.unique(dfTemp['TissueType']):
       print('Cohort',cohort)
       proportion_df,count_df,total_count = density_normalize(dfTemp[dfTemp['TissueType']==cohort],areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
       correlation_heatmap_plot(dfTemp[dfTemp['TissueType']==cohort],args.write_dir,f'class_frequency_correlation_{cohort}_short.tiff',class_labels=class_labels,count_df=count_df,proportion_df=proportion_df)

    
    #%%
    df = pd.DataFrame(pd.read_csv(args.cell_expression_dir,index_col=False))
    df =convert_mixed_dtype_to_string(df)
    
    
    for cohort in np.unique(dfTemp['disease_cohort']):
        print('Cohort',cohort)
        
        dfTemp = df.copy()
        dfTemp2 = df.copy()
        dfTemp['class_label'].replace({'BDCA2pCD103p_DCs':'general_uknown'},inplace=True) 

        stacked_bar_plot_cell_class(dfTemp=dfTemp[dfTemp['disease_cohort']==cohort],writeDir=args.write_dir,
                                    filename=f'class_label_barplot_stacked_{cohort}.png',colorDict=colorDict,unique_labels=[x for x in class_order_plot if x in dfTemp['class_label'].unique()],order_by='TissueType')
  
        
        dfTemp2=dfTemp.copy()
        dfTemp2=dfTemp[dfTemp['disease_cohort']==cohort]
        stacked_bar_plot_cell_class(dfTemp=dfTemp2[~dfTemp2['class_label'].isin(['Proximal_tubules','Inflamed_tubule','Distal_collecting_tubules','Endothelial_cells','general_unknown','RBCs'])],writeDir=args.write_dir,
                                    filename=f'class_label_barplot_stacked_{cohort}_immuneOnly.png',colorDict=colorDict,unique_labels=[x for x in class_order_plot if x in dfTemp2['class_label'].unique()],order_by='TissueType')
        dfTemp['class_label'].replace({'CD4+ICOS+_Tcell':'CD4+_Tcell', 
                                       'BDCA2pCD103p_DCs':'general_uknown',
                                            'CD4+PD1+_Tcell':'CD4+_Tcell',
                                            'CD4+Foxp3+_Tcell':'CD4+_Tcell', 
                                            'CD4+ICOS+PD1+_Tcell':'CD4+_Tcell',
                                            'CD14+CD163+MERTK+_macrophages':'CD14+CD163+_macrophages',
                                            'CD8+Foxp3+_Tcell':'CD8+_Tcell', 
                                            'CD8+ICOS+PD1+_Tcell':'CD8+_Tcell', 
                                            'CD8+ICOS+_Tcell':'CD8+_Tcell',
                                            'CD8+PD1+_Tcell':'CD8+_Tcell'}, inplace=True)
        stacked_bar_plot_cell_class(dfTemp=dfTemp[dfTemp['disease_cohort']==cohort],writeDir=args.write_dir,
                                    filename=f'class_label_barplot_stacked_{cohort}_simplified.png',colorDict=colorDict,unique_labels=[x for x in class_order_plot if x in dfTemp['class_label'].unique()],order_by='TissueType')

        dfTemp2=dfTemp.copy()
        dfTemp2=dfTemp[dfTemp['disease_cohort']==cohort]
        stacked_bar_plot_cell_class(dfTemp=dfTemp2[~dfTemp2['class_label'].isin(['Proximal_tubules','Inflamed_tubule','Distal_collecting_tubules','Endothelial_cells','general_unknown','RBCs'])],writeDir=args.write_dir,
                                    filename=f'class_label_barplot_stacked_{cohort}_immuneOnly_simplified.png',colorDict=colorDict,unique_labels=[x for x in class_order_plot if x in dfTemp2['class_label'].unique()],order_by='TissueType')

    #%%
    for cohort in np.unique(dfTemp['disease_cohort']):
        print('Cohort',cohort)
        proportion_df,count_df,total_count = density_normalize(dfTemp[dfTemp['disease_cohort']==cohort],areaDicts=[NKAreaDict,LNAreaDict,ARAreaDict])
        correlation_heatmap_plot(dfTemp[dfTemp['disease_cohort']==cohort],args.write_dir,f'class_frequency_correlation_{cohort}_short.tiff',class_labels=class_labels,count_df=count_df,proportion_df=proportion_df)

    
    
    
    #%%
    

if __name__=='__main__':
    main()

end_time = time.perf_counter()
print(f"Time script took to run: {end_time-start_time} seconds ({(end_time-start_time)/60} minutes) .")











