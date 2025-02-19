###the packages we use
import time
start_time = time.perf_counter()
import os
import argparse
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns
import collections
from scipy import stats
from statsmodels.stats import multitest
import matplotlib.colors as mcolors
#function

#%%

def NonLinCdict(steps, hexcol_array):
    cdict = {'red': (), 'green': (), 'blue': ()}
    for s, hexcol in zip(steps, hexcol_array):
        rgb = matplotlib.colors.hex2color(hexcol)
        cdict['red'] = cdict['red'] + ((s, rgb[0], rgb[0]),)
        cdict['green'] = cdict['green'] + ((s, rgb[1], rgb[1]),)
        cdict['blue'] = cdict['blue'] + ((s, rgb[2], rgb[2]),)
    return(cdict)


def cluster_phenotype(km,DBSCANDf,colorDict,histPlot,writeDir):
    
    
    
    heatmapDfPVals = pd.DataFrame(index = list(DBSCANDf.columns))
    heatmapDfTStat = pd.DataFrame(index = list(DBSCANDf.columns))
    heatmapDf = pd.DataFrame(index = list(DBSCANDf.columns))
    for lab in np.unique(km.labels_):
        print(f'DBSCAN Cluster {lab}')
        name1 = f'Cluster {lab}'
        name2 = "Other aggregates"
        
        clusterA = DBSCANDf[km.labels_ == lab]
        clusterB = DBSCANDf[km.labels_ != lab]
        
        
        comparisonDf = collections.defaultdict(list)
        
        
        for col in clusterA.columns:
            bins = 10

             ##lets plot proportions of CD20+
            if histPlot:
                plt.figure()
                plt.hist([clusterA[col],clusterB[col]], bins =bins, stacked=False,color=[ colorDict[lab], "k"])
                plt.axvline(np.mean(clusterA[col]), 0, 1, color =colorDict[lab])
                plt.axvline(np.mean(clusterB[col]), 0, 1, color ="k")
                plt.xlabel(col)
                plt.savefig(writeDir+f'histogram_{col}.tif',dpi=75)
                plt.close()
                # plt.show() 

            # Perform t-test
            t, p = stats.ttest_ind(clusterA[col],clusterB[col], equal_var=False)
            comparisonDf['t_stat'].append(t)
            comparisonDf['t_pval'].append(p)
            
           # Perform K-S test
            ks_statistic, p_value = stats.ks_2samp(clusterA[col],clusterB[col])
            comparisonDf['ks_stat'].append(ks_statistic)
            comparisonDf['ks_pval'].append(p_value)
            print(f'K-S statistic: {ks_statistic}')
            print(f'p-value: {p_value}')  
            
            print("####################################################################")
        
        
        hc = ['#f5f5ff', '#e2e2f4', '#a7a7d7', '#39399f', '#000080']
        th = [0, 0.01, 0.051, 0.1, 1]

        cdict = NonLinCdict(th, hc)
        cmPval = matplotlib.colors.LinearSegmentedColormap('test', cdict)

        comparisonDf = pd.DataFrame(comparisonDf)
        comparisonDf['t_stat'] = comparisonDf['t_stat'].fillna(0)
        comparisonDf['t_pval'] = comparisonDf['t_pval'].fillna(1)
        
   
        # Apply FDR correction
        rejected, p_values_fdr, _, _ = multitest.multipletests(comparisonDf['t_pval'], method='fdr_bh')
        comparisonDf['t_pval_corrected'] =p_values_fdr
        # print(multitest.multipletests(Pvals, method='fdr_bh'))
        dfPval = pd.DataFrame(p_values_fdr,columns =["P_Values_Corrected"], index = clusterA.columns)
        
        heatmapDfPVals[f'Cluster_{lab}'] = p_values_fdr
        heatmapDfTStat[f'Cluster_{lab}'] = comparisonDf['t_stat'].values

        plt.figure()
        ax1 = plt.subplots(figsize=(10,10))
        sns.set(font_scale=0.9)
        ax1 = sns.heatmap(dfPval,xticklabels=True, yticklabels=True,annot=True,cmap=cmPval)
        plt.title(name1)
        plt.savefig(writeDir+f'heatmap_pval_{col}.tif',dpi=75)
        plt.close()
        
        
        dfTstat = pd.DataFrame(comparisonDf['t_stat'].values,columns =['Cluster {lab}'], index = clusterA.columns)

        hc = ['#ff0000','#4f0202','#000000','#02590c','#00ff1d']
        numbs =dfTstat
        zero = stats.percentileofscore(numbs, 0)
        step = ((100-zero)/2)
        zero_plus_Step= np.percentile(numbs, zero +step)
        step = zero/2
        zero_minus_Step= np.percentile(numbs, zero -step)
        zero_plus_Step = stats.percentileofscore(numbs, zero_plus_Step)
        zero_minus_Step = stats.percentileofscore(numbs, zero_minus_Step)

        if not np.isnan(th).any(): ##otherwise will crash
            
            cdict = NonLinCdict(th, hc)
            cmContrast = matplotlib.colors.LinearSegmentedColormap('test', cdict) 
            heatmapDf = pd.concat([heatmapDf, dfTstat], axis=1)
    
            plt.figure()
            ax1 = plt.subplots(figsize=(10,10))
            sns.set(font_scale=0.9)
            ax1 = sns.heatmap(dfTstat,xticklabels=True, yticklabels=True,annot=True,cmap =cmContrast)
            plt.title(name1)
            plt.savefig(writeDir+f'heatmap_tstat_{col}.tif',dpi=75)
            plt.show(block=False)
            plt.pause(5)
            plt.close()
            plt.close()
            
            
        
    return(heatmapDfPVals,heatmapDfTStat)


def DBSCAN_cluster_by_group_plots(km,DBSCANDf,feat,writeDir,dpi, bar_order=None):
    
    value_counts = DBSCANDf['DBSCAN_cluster_type'].value_counts()
        # Get colormap
    cmap = mcolors.ListedColormap(plt.cm.tab10.colors)
    
    # Get a list of colors from the colormap equal to the number of unique categories
    colors = [cmap(i) for i in range(len(value_counts.index))]

    categories =value_counts.index
    
    # Initialize an empty DataFrame
    all_value_counts = pd.DataFrame(columns=np.unique(list(DBSCANDf[feat])), index=categories)

    
    for fe in np.unique(list(DBSCANDf[feat])):
        print('fe',fe)

        dfFe = DBSCANDf[DBSCANDf[feat]==fe]
       
        value_counts = dfFe['DBSCAN_cluster_type'].value_counts().reindex(categories).fillna(0)
       
        # Create the pie chart
        patches, texts, autotexts = plt.pie(value_counts, autopct='%1.1f%%', colors=colors, startangle=140, textprops={'weight': 'extra bold'} , wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})

        
        # Move out the percentages to sit just outside the slices
        for autotext in autotexts:
            autotext.set_weight('extra bold')
            autotext.set_fontsize(14)  # Adjust as needed
        
        plt.title(fe)
        plt.savefig(writeDir+f'{fe}_pie_chart.tif',dpi=dpi)
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        plt.close()
        
        # Bar graph
        value_counts.plot(kind='bar', color=colors)
        plt.title(f'Bar graph for {fe}')
        plt.savefig(writeDir+f'{fe}_bar_graph.tiff',dpi=dpi)
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        

        # Create the second pie chart without percentage labels
        plt.figure()  # Start a new figure
        patches, texts = plt.pie(value_counts, colors=colors, startangle=140, textprops={'weight': 'extra bold'}, wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
        plt.title(fe)
        plt.savefig(writeDir + f'{fe}_pie_chart_without_labels.tiff', dpi=dpi)
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        # Insert the value counts into the corresponding column in all_value_counts
        all_value_counts[fe] = dfFe['DBSCAN_cluster_type'].value_counts().reindex(categories).fillna(0)

        
    color_map = {
    'Lupus_Nephritis': (0, 0, 1, 1),  # blue
    'Normal_Kidney': (0, 1, 0, 1),  # green
    'Renal_Allograft': (1, 0, 1, 1)  # purple 
    }
    
    if bar_order:
        all_value_counts = all_value_counts[bar_order]
    
    # Make x and y axis labels bold
    ax = plt.gca()
    ax.xaxis.label.set_weight('bold')  # Make x axis label bold
    ax.yaxis.label.set_weight('bold')  # Make y axis label bold
    # Reset to default Matplotlib style to ensure a white background
    plt.style.use('default')
    plt.figure(figsize=(10,10))
    all_value_counts.plot(kind='bar', color=color_map, rot=0)
    plt.title('Bar graph for all features')
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))  # Added line to position the legend
    # Set the background color of the plot to white
    ax.set_facecolor('white')  # Setting the axes background color
    plt.gcf().set_facecolor('white')  # Setting the figure background color

    for spine in plt.gca().spines.values():
        spine.set_linewidth(5)  # Set the thickness here
        spine.set_color('black')  # Set the color to black
    # Set black grid lines
    plt.grid(color='black', linestyle='--', linewidth=1)  # Adjust linewidth as needed


    plt.xticks(fontweight="bold", fontsize=14)
    plt.yticks(fontweight="bold", fontsize=14)
    plt.savefig(writeDir+f'bar_graph_all_features.tiff',dpi=dpi)
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    plt.close()    
   
    
def create_heatmap(heatmap_df, write_dir, filename, title,columns_to_drop=None,include_counts=True, include_proportions=True, show_annotations=True,exclude_classes=None):
    """
    Create a heatmap from the given DataFrame.

    Parameters:
    heatmap_df (pd.DataFrame): DataFrame to be visualized in the heatmap.
    write_dir (str): Directory where the heatmap image will be saved.
    filename (str, optional): Filename for the saved heatmap image. Defaults to 'DBSCAN_cluster_phenotype_heatmap.tif'.
    title (str, optional): Title for the heatmap. Defaults to 'Heatmap of log2FC Cell Count'.
    """
    
    # Drop specified columns if any
    if columns_to_drop is not None:
        heatmap_df = heatmap_df.drop(columns=columns_to_drop, errors='ignore')
        
    # Exclude specified classes if any
    if exclude_classes is not None:
        classes_to_exclude = [cls + suffix for cls in exclude_classes for suffix in ['_proportion', '_count']]
        heatmap_df = heatmap_df.drop(index=classes_to_exclude, errors='ignore')

        
    # Filter rows based on include flags
    if not include_counts:
        heatmap_df = heatmap_df[~heatmap_df.index.str.contains('count')]
    if not include_proportions:
        heatmap_df = heatmap_df[~heatmap_df.index.str.contains('proportion')]
        
    

    linewidths = 1.5
    border_linewidth = 1.5  # Set the desired border linewidth here
    cmap = sns.color_palette("RdBu", as_cmap=True)
    
    fig, ax = plt.subplots(figsize=(10, 10))  # Use 'ax' for the Axes object
    cg = sns.heatmap(heatmap_df, ax=ax, cmap=cmap.reversed(), fmt='.2f', yticklabels=1, annot=show_annotations, 
                     annot_kws={"size": 10, "weight": "extra bold"}, vmax=15, vmin=-15, linewidths=linewidths, linecolor='black')
    plt.title(title, size=14, weight="extra bold")
    plt.xlabel('Biopsy Cohorts Compared', size=14, weight="extra bold")
    plt.ylabel('Cell Types', size=14, weight="extra bold")
    ax.set_yticklabels(cg.get_yticklabels(), fontsize=10, weight="extra bold")
    ax.set_aspect('equal')  # Set the aspect of the axis to be "equal"
    
    # Rotate x-axis labels and set their font properties
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_weight('bold')
        label.set_size(10)
    
    for label in ax.get_yticklabels():
        label.set_weight('bold')
        label.set_size(10)
    
    # Set the border linewidth
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(border_linewidth)  # Set the thickness of the border
    
    plt.tight_layout()
    plt.savefig(write_dir + filename, dpi=300)
    plt.show()
    plt.close()
    
        
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
    parser.add_argument('-md',
            '--modeldir',
            type=str,
            default='channel_expression_csvs',
            help=''
            )
    parser.add_argument('-ci',
            '--cell_info_dir',
            type=str,
            default='channel_expression_csvs',
            help=''
            )
    parser.add_argument('-rmd',
    '--remove_doublets',
    type=str,
    # default='cell_patches_1um',
    help=''
    )
    parser.add_argument('--remove_below',
                        type=int, 
                        default=None,
                        help='Threshold for removing doublets')
    
    args,unparsed = parser.parse_known_args()

   
    
    with open(args.modeldir+"fitted_km.pkl",'rb') as f:
        km = pickle.load(f)
    
    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)
        

    infoDf = pd.DataFrame(pd.read_csv(args.cell_info_dir,index_col=False))
    
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
    plt.close()
    
    
    DBSCANDf = DBSCANDf[DBSCANDf['total_cell_count'] <= 3000]
    
    
    # if ast.literal_eval(args.remove_doublets):
    
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
    plt.close()
    
    
    #%%
    clust0 = DBSCANDf[km.labels_ ==0]
    clust1 = DBSCANDf[km.labels_ ==1]
    
    #%%
    

    # Get the labels
    labels = km.labels_
    
    # Get unique labels
    unique_labels = np.unique(labels)
    
    # Generate a list of random colors, one for each unique label
    colors = np.random.rand(len(unique_labels), 4)  # For RGBA, we need 4 values
    
    # Make sure the last value (alpha) is 1 for full opacity
    colors[:, 3] = 1.0
    
    # Create a dictionary mapping labels to colors
    color_dict = {label: color for label, color in zip(unique_labels, colors)}
    
    print(color_dict)
    
    columns_to_drop = ['Unnamed: 0', 'disease_cohort', 'CompName','TileNum',
    'TileCellID',
    'idxs',
    'DBSCAN_label', 'AccessionNumber', 'TissueType','CD14+CD163+_macrophages_CD4+ICOS+PD1+_Tcell_dist66.36_angle60_sum',
    'CD14+CD163+_macrophages_CD4+_Tcell_dist66.36_angle60_sum',
    'CD14+CD163+_macrophages_CD14+MERTK+_macrophages_dist66.36_angle60_sum',
    'CD14+CD163+_macrophages_CD8+_Tcell_dist66.36_angle60_sum',
    'CD14+CD163+_macrophages_Bcell_dist66.36_angle60_sum',
    'CD14+CD163+_macrophages_cDC1_dist66.36_angle60_sum',
    'CD14+CD163+_macrophages_CD4+PD1+_Tcell_dist66.36_angle60_sum',
    'CD14+CD163+_macrophages_CD4+ICOS+_Tcell_dist66.36_angle60_sum',
    'CD14+CD163+_macrophages_cDC2_dist66.36_angle60_sum',
    'CD14+CD163+_macrophages_pDC_dist66.36_angle60_sum',
    'Bcell_CD14+CD163+_macrophages_dist66.36_angle60_sum',
    'Bcell_CD4+ICOS+PD1+_Tcell_dist66.36_angle60_sum',
    'Bcell_CD4+_Tcell_dist66.36_angle60_sum',
    'Bcell_CD8+_Tcell_dist66.36_angle60_sum',
    'Bcell_CD4+PD1+_Tcell_dist66.36_angle60_sum',
    'Bcell_CD4+ICOS+_Tcell_dist66.36_angle60_sum',
    'Bcell_cDC1_dist66.36_angle60_sum',
    'Bcell_CD14+MERTK+_macrophages_dist66.36_angle60_sum',
    'Bcell_pDC_dist66.36_angle60_sum',
    'Bcell_cDC2_dist66.36_angle60_sum',
    'CD14+MERTK+_macrophages_CD8+_Tcell_dist66.36_angle60_sum',
    'CD14+MERTK+_macrophages_Bcell_dist66.36_angle60_sum',
    'CD14+MERTK+_macrophages_CD4+PD1+_Tcell_dist66.36_angle60_sum',
    'CD14+MERTK+_macrophages_CD14+CD163+_macrophages_dist66.36_angle60_sum',
    'CD14+MERTK+_macrophages_CD4+_Tcell_dist66.36_angle60_sum',
    'CD14+MERTK+_macrophages_cDC1_dist66.36_angle60_sum',
    'CD14+MERTK+_macrophages_CD4+ICOS+PD1+_Tcell_dist66.36_angle60_sum',
    'CD14+MERTK+_macrophages_CD4+ICOS+_Tcell_dist66.36_angle60_sum',
    'CD14+MERTK+_macrophages_pDC_dist66.36_angle60_sum',
    'CD14+MERTK+_macrophages_cDC2_dist66.36_angle60_sum',
    'total_cognate',
    'cognate_ratio',
    'Bcell_cognate_ratio',
    'CD14+CD163+MERTK+_macrophages_cognate_ratio',
    'CD14+CD163+_macrophages_cognate_ratio',
    'CD14+MERTK+_macrophages_cognate_ratio',
    'CD14+_macrophages_cognate_ratio',
    'CD16+_macrophages_cognate_ratio',
    'CD3+_Tcell_cognate_ratio',
    'CD4+CD8+_Tcell_cognate_ratio',
    'CD4+Foxp3+_Tcell_cognate_ratio',
    'CD4+ICOS+PD1+_Tcell_cognate_ratio',
    'CD4+ICOS+_Tcell_cognate_ratio',
    'CD4+PD1+_Tcell_cognate_ratio',
    'CD4+_Tcell_cognate_ratio',
    'CD8+Foxp3+_Tcell_cognate_ratio',
    'CD8+ICOS+PD1+_Tcell_cognate_ratio',
    'CD8+ICOS+_Tcell_cognate_ratio',
    'CD8+PD1+_Tcell_cognate_ratio',
    'CD8+_Tcell_cognate_ratio',
    'Distal_collecting_tubules_cognate_ratio',
    'Endothelial_cells_cognate_ratio',
    'HLAII+_Monocytes_cognate_ratio',
    'HLAII-_Monocytes_cognate_ratio',
    'Inflamed_tubule_cognate_ratio',
    'Mo-Macrophage_cognate_ratio',
    'NK_Tcell_cognate_ratio',
    'NK_cells_cognate_ratio',
    'Neutrophils_cognate_ratio',
    'Plasmablasts_cognate_ratio',
    'Proximal_tubules_cognate_ratio',
    'RBCs_cognate_ratio',
    'TCRgd_Tcell_cognate_ratio',
    'cDC1_cognate_ratio',
    'cDC2_cognate_ratio',
    'other_cognate_ratio',
    'pDC_cognate_ratio',
    'plasma_cells_cognate_ratio',
    'cognateAreaRatio_skewness',
    'cognateAreaRatio_kurtosis',
    'cognateAreaRatio_std',
    'cognateAreaRatio_var',
    'cognateAreaRatio_mean',
    'cognateAreaRatio_max',
    'cognateAreaRatio_min',
    'cognateAreaRatio_25thPercentile',
    'cognateAreaRatio_50thPercentile',
    'cognateAreaRatio_75thPercentile','other_proportion','other_count','RBCs_proportion',
    'RBCs_count','Proximal_tubules_proportion','Proximal_tubules_count','Inflamed_tubule_proportion',
    'Inflamed_tubule_count','Endothelial_cells_proportion','Endothelial_cells_count','Distal_collecting_tubules_proportion',
    'Distal_collecting_tubules_count']
    

    heatmapDfPVals,heatmapDfTStat = cluster_phenotype(km,
                      DBSCANDf.drop(columns=columns_to_drop),
            color_dict,
            False,
            args.write_dir)
    
    #%%
    create_heatmap(heatmap_df=heatmapDfTStat, write_dir=args.write_dir, 
                   filename='DBSCAN_cluster_phenotype_heatmap_cellcounts.tif', title='Heatmap of log2FC Cell Count',
                   columns_to_drop=None,include_counts=True, include_proportions=False, show_annotations=False,exclude_classes=['Endothelial_cells','RBCs','Distal_collecting_tubules','other','Inflamed_tubule','Proximal_tubules'])
    create_heatmap(heatmap_df=heatmapDfTStat, write_dir=args.write_dir, 
                   filename='DBSCAN_cluster_phenotype_heatmap_cellproportion.tif', title='Heatmap of log2FC Cell Count',
                   columns_to_drop=None,include_counts=False, include_proportions=True, show_annotations=False,exclude_classes=['Endothelial_cells','RBCs','Distal_collecting_tubules','other','Inflamed_tubule','Proximal_tubules'])
    #%%
    cmap = sns.color_palette("RdBu", as_cmap=True)
    
    fig, ax = plt.subplots(figsize=(16, 16))
    cg = sns.heatmap(heatmapDfTStat,cmap=cmap.reversed(), fmt='.2f',yticklabels=1,annot=True, annot_kws={"size": 10, "weight": "extra bold"},vmax=25,vmin=-25)
    # hm.annotate(np.vectorize(add_asterisk)(heatmapExport), xycoords='data', fontsize=10, weight='bold', ha='center', va='center')
    plt.title(f'Heatmap of log2FC Cell Count',size=14, weight="extra bold")
    plt.xlabel('Biopsy Cohorts Compared',size=14, weight="extra bold")
    plt.ylabel('Cell Types',size=14, weight="extra bold")
    cg.set_yticklabels(cg.get_yticklabels(), fontsize = 10,weight="extra bold")
    
    plt.tight_layout()
    plt.savefig(args.write_dir+'DBSCAN_cluster_phenotype_heatmap.tif',dpi=200)
    plt.show()
    plt.close()

 
#%%    
#DBSCAN cluster by type
    DBSCANDf['DBSCAN_cluster_type'] = km.labels_
    DBSCAN_cluster_by_group_plots(km,DBSCANDf,'disease_cohort',args.write_dir,300,bar_order=['Normal_Kidney','Lupus_Nephritis','Renal_Allograft'])
    
#%%%

    
if __name__=='__main__':
    main()
end_time = time.perf_counter()
print(f"Time script took to run: {end_time-start_time} seconds ({(end_time-start_time)/60} minutes) .")











