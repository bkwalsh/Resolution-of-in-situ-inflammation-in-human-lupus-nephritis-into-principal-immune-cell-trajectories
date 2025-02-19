###the packages we use
import time
start_time = time.perf_counter()
import os
import argparse
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.patches as mpatches


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


def reorder_dataframe(df, class_order):
    """Reorder dataframe based on a given class order."""
    # Create a dictionary to map class labels to their order
    class_order_dict = {class_label: index for index, class_label in enumerate(class_order)}
    
    # Sort the dataframe based on the order in class_order_dict
    df['order'] = df['class_label'].map(class_order_dict)
    df = df.sort_values('order').drop(columns='order')
    
    return df
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


    
    
    args,unparsed = parser.parse_known_args()
  


    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)
    

    df = pd.DataFrame(pd.read_csv(args.cell_expression_dir,index_col=False))
    
    labels  =np.array(df['class_label'])
    
    
    #%%
    for lab in np.unique(df['class_label']):
        print(lab)
    #%%
    a = 1
    colorDict = {
        'CD8+Foxp3+_Tcell':[30/255, 229/255, 247/255, a],###Done
        'CD8+ICOS+PD1+_Tcell':[2/255, 115/255, 96/255, a],###Done
        'CD8+ICOS+_Tcell':[21/255, 161/255, 173/255, a],###Done
        'CD8+PD1+_Tcell':[13/255, 186/255, 157/255, a],###Done
        'CD8+_Tcell':[138/255, 255/255, 255/255, a],###Done
        'CD8+_intraepithelial_Tcell':[145/255, 240/255, 240/255, a],###Done,
                 'CD4+CD8+_Tcell':[52/255, 214/255, 49/255, a],###Done
                 'NK_Tcell':[3/255, 135/255, 34/255, a],###Done
                 'TCRgd_Tcell':[45/255, 163/255, 82/255, a],###Done
                 'NK_cells':[82/255, 155/255, 80/255, a],###Done
                 'CD3+_Tcell':[111/255, 252/255, 109/255, a], ###Done
                 'CD4+_Tcell':[71/255, 27/255, 237/255, a], ###Done
                 'CD4+_intraepithelial_Tcell':[91/255, 35/255, 243/255, a],
                 'CD4+Foxp3+_Tcell':[121/255, 116/255, 237/255, a],###Done
                 'CD4+ICOS+PD1+_Tcell':[56/255, 144/255, 207/255, a],###Done
                 'CD4+ICOS+_Tcell':[36/255, 53/255, 163/255, a],###Done
                 'CD4+PD1+_Tcell':[34/255, 93/255, 181/255, a],###Done
                 'Bcell':[206/255,6/255,209/255, a],###Done
                 'plasma_cells':[114/255, 40/255, 118/255, a],###Done
                 'Plasmablasts':[214/255, 138/255, 230/255, a],###Done
                 'CD14+_macrophages':[247/255, 18/255, 10/255, a],###Done
                 'CD14+CD163+_macrophages':[186/255, 42/255, 37/255, a],###Done
                 'CD14+MERTK+_macrophages':[255/255, 115/255, 115/255, a],###Done
                 'CD14+CD163+MERTK+_macrophages':[255/255, 77/255, 77/255, a],###Done
                 'CD16+_macrophages':[227/255, 23/255, 60/255, a],###Done
                 'HLAII+_Monocytes':[ 247/255,  101/255, 199/255, a], #[255/255, 110/255, 0/255, a]
                 'HLAII-_Monocytes':[ 176/255,  5/255, 119/255, a],###Done  [245/255, 160/255, 66/255, a]
                 'Mo-Macrophage':[ 240/255,  90/255, 195/255, a],###Done [235/255, 94/255, 30/255, a]
                 'cDC1':[250/255, 205/255, 25/255, a],###Done [ 250/255,  5/255, 168/255, a]
                 'cDC2':[245/255, 160/255, 66/255, a],###Done [ 176/255,  5/255, 119/255, a]
                 'pDC':[255/255, 110/255, 0/255, a],###Done [ 247/255,  101/255, 199/255, a]
                 'BDCA2pCD103p_DCs':[235/255, 94/255, 30/255, a],###Done [ 240/255,  90/255, 195/255, a]
                 # 'Monocyte_derived_DCs':[ 170/255,  90/255, 10/255, a],
                 'Neutrophils':[ 250/255,  5/255, 168/255, a],###Done [250/255, 205/255, 25/255, a]
                 'RBCs':[115/255, 41/255,3/255, a],###Done
                 'general_unknown':[230/255,230/255,230/255,a],###Done
                 'Inflamed_tubule':[ 160/255,  121/255, 35/255, a],###Done
                 'Endothelial_cells':[ 112/255,  68/255, 6/255, a],###Done
                 'Distal_collecting_tubules':[ 133/255,  98/255, 50/255, a],###Done
                 'Proximal_tubules':[ 161/255,  124/255, 87/255, a]} ###Done 
    
    #%%
    
    
    labelMap = {
        'CD8+Foxp3+_Tcell':'CD8+Foxp3+ Tcell',
        'CD8+ICOS+PD1+_Tcell':'CD8+ICOS+PD1+ Tcell',
        'CD8+ICOS+_Tcell':'CD8+ICOS+ Tcell',
        'CD8+PD1+_Tcell':'CD8+PD1+ Tcell',
        'CD8+_Tcell':'CD8+ Tcell',
        'CD8+_intraepithelial_Tcell':'CD8+ intraepithelial Tcell',
    'CD4+CD8+_Tcell': 'CD4+CD8+ T Cell',
    'NK_Tcell':'Natural Killer T Cell',
    'TCRgd_Tcell':'TCRgd T Cell',
    'CD3+_Tcell': 'CD3+ T Cell', ##cyan
    'CD4+_Tcell': 'CD4+ T Cell',
    'CD4+Foxp3+_Tcell':'CD4+Foxp3+ Tcell',
    'CD4+ICOS+PD1+_Tcell':'CD4+ICOS+PD1+ Tcell',
    'CD4+ICOS+_Tcell':'CD4+ICOS+ Tcell',
    'CD4+PD1+_Tcell':'CD4+PD1+ Tcell',
    'CD4+_intraepithelial_Tcell':'CD4+ intraepithelial Tcell',
   'Mo-Macrophage':'Mo-Macrophage',
    'Bcell': 'B Cell',
    'plasma_cells':'Plasma Cell',
    'Plasmablasts':'Plasmablasts',
    'CD14+_macrophages':'CD14+ Macrophage',
    'CD14+CD163+_macrophages':'CD14+CD163+ MP',
    'CD14+MERTK+_macrophages':'CD14+MERTK+ MP',
    'CD14+CD163+MERTK+_macrophages':'CD14+CD163+MERTK+ MP',
    'CD16+_macrophages':'CD16+ Macrophage',
    'HLAII+_Monocytes':'HLAII+ Monocytes',
    'HLAII-_Monocytes':'HLAII- Monocytes',
    'Inflamed_tubule':'Inflamed Tubule',
    'Endothelial_cells': 'Endothelial Cell',
    'Distal_collecting_tubules':'Distal Collecting Tubule',
    'Proximal_tubules':'Proximal Tubule',
    'cDC':'Conventional Dendritic Cell',
    'pDC':'Plasmacytoid Dendritic Cell',
    'BDCA2pCD103p_DCs':'BDCA2+CD103+ DCs',
    'NK_cells':'Natural Killer Cell',
    'Neutrophils':'Neurophil',
    'RBCs':'Red Blood Cell',
    'general_unknown':'Non-Classified Cell'}  
 
    
    class_order_plot=['Bcell','Plasmablasts','plasma_cells','CD4+Foxp3+_Tcell',
                      'CD4+ICOS+PD1+_Tcell','CD4+ICOS+_Tcell','CD4+PD1+_Tcell','CD4+_Tcell','CD4+_intraepithelial_Tcell',
                      'CD8+Foxp3+_Tcell','CD8+ICOS+PD1+_Tcell','CD8+ICOS+_Tcell','CD8+PD1+_Tcell',
                      'CD8+_Tcell','CD8+_intraepithelial_Tcell','CD3+_Tcell','CD4+CD8+_Tcell','TCRgd_Tcell','NK_Tcell','NK_cells','CD14+CD163+MERTK+_macrophages',
                     'CD14+CD163+_macrophages','CD14+MERTK+_macrophages','CD14+_macrophages','CD16+_macrophages','HLAII+_Monocytes',
                     'HLAII-_Monocytes','Mo-Macrophage','Neutrophils','BDCA2pCD103p_DCs','cDC1','cDC2','pDC', 'Distal_collecting_tubules',
                     'Endothelial_cells','Inflamed_tubule','Proximal_tubules','RBCs','general_unknown']

    
    fig, ax = plt.subplots(1, 1, figsize=(5, len(colorDict)))
    for i, cellClass in enumerate(class_order_plot):
        key=cellClass
        value=colorDict[key]
        new_label = labelMap.get(key, key)
        new_label = new_label.replace('_',' ')
        rect = mpatches.Rectangle((0, i), 1, 1, color=value)
        ax.add_patch(rect)
        ax.text(0.5, i + 0.5, new_label, weight='extra bold',ha='center', va='center',
                fontsize=17, color='black' if sum(value[:3]) > 1.5 else 'white')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, len(colorDict)])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(args.write_dir+"color_scheme_example_plot.png",dpi=300)
    plt.show()
    plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(5, len(colorDict) + 1))  # Adjusted figure size
    for i, cellClass in enumerate(class_order_plot):
        print(i)
        # i+=1
        key = cellClass
        value = colorDict[key]
        new_label = labelMap.get(key, key).replace('_', ' ')
        rect = mpatches.Rectangle((0, i), 1, 1, color=value)
        ax.add_patch(rect)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, len(colorDict) ])  # Adjusted ylim
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(args.write_dir+"color_scheme_example_plot_no_labs.png",dpi=300)
    plt.show()
    plt.close()
    

    legends = []
    for key in colorDict.keys():
        print(key)
        legends.append(mpatches.Patch(color=colorDict[key], label=key))
    with open(args.write_dir+"colorDict.pkl",'wb') as f:
        pickle.dump(colorDict,f)
    with open(args.write_dir+"legends.pkl",'wb') as f:
        pickle.dump(legends,f)  
  #%%
    
    
if __name__=='__main__':
    main()

end_time = time.perf_counter()
print(f"Time script took to run: {end_time-start_time} seconds ({(end_time-start_time)/60} minutes) .")










