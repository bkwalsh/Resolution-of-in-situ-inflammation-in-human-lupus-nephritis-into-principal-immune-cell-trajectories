# CODEX Project

## Description
This is the code for the paper titled 'Resolution of in situ inflammation in human lupus nephritis into principal immune cell
trajectories'

## Installation
environment.yaml --contains all the packages and versions used. Use this to create a virtual environment with the appropriate packages

## Dataset description
1) cell_information.csv --contains all the metadata for the cells, ie coordinates, samples, accession number
2) cell_expression_all.csv --contains all uncorrected raw MFI values for the cells for all CODEX markers
3) for_manual_gating_corrected.csv --contains the neurocombat batch corrected data, prior to formatting
4) for_manual_gating_corrected_formatted.csv --corrected and formated dataset prior to cell annotation
5) cell_expression_classified_manual_gating.csv --annotated cells with MFI and other relevant metadata
6) otsu_thresholds.csv --stores the MFI used at each gate FYI
7) colorDict.pkl --color dictionary for the annotated cell classes
8) legends.pkl --legends that go with the color dictionary
9) NKAreaDict.pkl --area of kidney control samples in mm^2 
10)LNAreaDict.pkl --area of lupus nephritis samples in mm^2
11) ARAreaDict.pkl --area of renal allograft rejection samples in mm^2
12) NK_clinical_data.csv -- clinical data for kidney control samples
13) LN_clinical_data.csv -- clinical data for lupus nephritis samples
14) RA_clinical_data.csv -- clinical data for renal allograft rejection samples
15) NK_LN.csv -- MWU test results comparing kidney control vs. lupus nephritis
16) NK_AR.csv -- MWU test results comparing kidney control vs. renal allograft rejection 
17) LN_AR.csv -- MWU test results comparing lupus nephritis vs. renal allograft rejection
18) DBSCAN_all_immune_combined.csv --joint file of the DBSCAN-segmented cell aggregates (immune cells only)

## Code description
1) batch_effect_correction_neurocombat.py   -- This script is used to correct batching effects of cell expression data
2) manual_data_reformat.py --This script is for reformatting the data to make it smaller, extracts only the columns we actually use
3) manual_cell_gating_otsu.py --This is the script that annotates the cells using the FACS-analogous method with multiotsu thresholding
4) cluster_color_dict_create.py --script creates the color dictionary used for the annotated cell class plots
5) phenotyping_UMAP_plots_csv_class_label.py --script creates the UMAP plots for visualizing the annotated cells as well as MFIs on the UMAPs
6) cohort_area_dictionary_create.py --calculates the area in mm^2 for the samples and stores as a pkl per cohort, this is done so that when doing density calculations I dont need to de novo calculate the area of the sample
7) clinical_features_analysis.py --simple summary analysis of the clinical features of the cohorts
8) cell_label_MFI_subanalysis.py --plots the joint and individual heatmaps of the cell MFI per annotated cell class
9) cell_label_density_analysis_calculate.py --does the MWU test comparing the annotated cell class density between cohorts
10) cell_label_density_analysis_plots.py --plots the MWU test
11) patient_heterogeneity_analysis.py --main workhorse script for generating most of the plots of the paper: [INSERT]
12) combined_structure_feature_extraction_final.py --uses parallel processing and tissue masks to calculate the area of the kidney compartments (interstitium, tubule, glomerulus, peritubules, periglomerular) as well as labeling which compartment the cells fall into using the global centroid coordinates
13) kidney_compartment_analysis_plots_final.py --plots the heatmaps of the annotated cell density in the kidney compartments
14) DBSCAN_spatial_segmentation.py  --uses DBSCAN to spatial segment the annotated immune cells in the patient cohorts using the global cell coordinates
15) DBSCAN_csv_compile.py --preprocesses and joins the DBSCAN segmentation files into a single csv for boostrapping estimate of optimal number of clusters. Calculates the count and proportion of the annotated cells as DBSCAN segmentation features
16) DBSCAN_optimal_cluster_calculate.py --script that runs the bootstrapping experiment for determining the optimal number of latent DBSCAN cluster classes
17) DBSCAN_optimal_cluster_plots.py --plots the result of the boostrappiong experiment, sum of squared distances (SSD) and delta SSD, elbow method is used to determine optimal k
18) DBSCAN_Kmeans_clustering_optimal_K.py --script used to apply kmeans clustering using the optimally determined k clusters to the DBSCAN-segmented annotated cell aggregates
19) DBSCAN_Kmeans_cluster_phenotype.py --script used to produce the leave-one-out z-score heatmap that helps phenotype the unique features of each DBSCAN cluster type. Plots the total counts of the DBSCAN clusters and cohort piechart composition 










