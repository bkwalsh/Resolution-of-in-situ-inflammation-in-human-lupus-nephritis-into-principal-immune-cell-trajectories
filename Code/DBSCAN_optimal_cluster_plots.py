###the packages we use
import time
start_time = time.perf_counter()
import os
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np


    #%%    
def main():
    
    #%%
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r',
            '--read_dir',
            type=str,
            # default='cell_patches_1um',
            help=''
            )

    parser.add_argument('-w',
            '--write_dir',
            type=str,
            # default='channel_expression_csvs',
            help=''
            )
   
    

    args,unparsed = parser.parse_known_args()

    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)
  
    
    files = sorted(os.listdir(args.read_dir))
    files = [x for x in files if '.pkl' in x]
    sumSquaredDistances = []
    deltas = []

    for filename in files:
        with open(args.read_dir+filename,'rb') as f:
            df = pickle.load(f)

        
        sumSquaredDistances.append(df[1])
        
        delta = []
        for i in range(0,len(df[1])-1):
            delta.append(df[1][i+1] - df[1][i])
        deltas.append(delta)
    sumSquaredDistances = np.stack(sumSquaredDistances,axis=0)
    deltas = np.stack(deltas,axis=0)
    
    
    means = sumSquaredDistances.mean(axis=0)
    std = sumSquaredDistances.std(axis=0)

    
    titleFont = 14
    fontsize=14
    linewidth=4
    markersize=8
    
    # Define a color for the plot elements
    plot_color = 'blue'  # You can choose any color you like

    plt.figure(figsize=(10,8))
    plt.title("Sum of squared distances",fontsize = titleFont,weight="extra bold")
    plt.errorbar(x = range(2,25),y =means,yerr=std,marker='s', linewidth=linewidth,markersize=markersize, color=plot_color)
    
    # Iterate over each point to add horizontal lines for std
    for i, (mean, error) in enumerate(zip(means, std)):
        plt.hlines(y=mean - error, xmin=i+1.7, xmax=i+2.3, color=plot_color, linewidth=linewidth)  # lower bound
        plt.hlines(y=mean + error, xmin=i+1.7, xmax=i+2.3, color=plot_color, linewidth=linewidth)  # upper bound

    
    plt.xticks(range(2,25,1))
    plt.grid()
    plt.xlabel("K clusters",fontsize = titleFont,weight="extra bold")
    plt.ylabel("Sum Squared Distance",fontsize = titleFont,weight="extra bold")
    for spine in plt.gca().spines.values():
        spine.set_linewidth(8)  # Set the thickness here
    plt.xticks(fontweight="bold", fontsize=fontsize)
    plt.yticks(fontweight="bold", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(args.write_dir+"sum_distances_squared_plot.tif",dpi=200)
    plt.show()
    plt.close()
    
    deltasMeans = deltas.mean(axis=0)
    deltasStd = deltas.std(axis=0)

    
    plt.figure(figsize=(10,8))
    plt.title("Delta sum of squared distances",fontsize = titleFont,weight="extra bold")
    plt.errorbar(x = range(3,25,1),y =deltasMeans,yerr=deltasStd,marker='s', linewidth=linewidth,markersize=markersize, color=plot_color)
    for i, (mean, error) in enumerate(zip(deltasMeans, deltasStd)):
        i+=1
        plt.hlines(y=mean - error, xmin=i+1.7, xmax=i+2.3, color=plot_color, linewidth=linewidth)  # lower bound
        plt.hlines(y=mean + error, xmin=i+1.7, xmax=i+2.3, color=plot_color, linewidth=linewidth)  # upper bound


    plt.xticks(range(3,25,1))
    plt.grid()
    plt.xlabel("K clusters",fontsize = titleFont,weight="extra bold")
    plt.ylabel("Delta Sum Squared Distance",fontsize = titleFont,weight="extra bold")
    for spine in plt.gca().spines.values():
        spine.set_linewidth(8)  # Set the thickness here
    plt.xticks(fontweight="bold", fontsize=fontsize)
    plt.yticks(fontweight="bold", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(args.write_dir+"delta_sum_distances_squared_plot.tif",dpi=200)
    plt.show()
    plt.close()
    #%%
    
 
if __name__=='__main__':
    main()

end_time = time.perf_counter()
print(f"Time script took to run: {end_time-start_time} seconds ({(end_time-start_time)/60} minutes) .")










