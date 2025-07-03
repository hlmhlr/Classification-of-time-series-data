#---- In this program, we separate the faulty portion and non faulty portion of the signal first. Then we sub-samples these portions to make the dataset.

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
#import pandas as pd
import math
from sklearn import preprocessing
from scipy.io import loadmat
# libraries to standardize the data
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.preprocessing import RobustScaler


fault_classes = {0: 'ab', 1: 'ac', 2: 'ag', 3: 'bc', 4: 'bg', 5: 'cg', 6: 'abc', 7: 'abg', 8: 'acg', 9: 'bcg', 10: 'abcg', 11: 'nf'}
fault_classes_names = {'ab': 0, 'ac': 1,'ag': 2,'bc': 3,'bg': 4,'cg': 5,'abc': 6,'abg': 7,'acg': 8, 'bcg': 9, 'abcg': 10,'nf': 11}


def plot_single_data(i_all, i_all_chunks, y_chunk_label, nc, fig_name):
    fig, axs = plt.subplots(len(i_all_chunks)+1)
    axs[0].plot(i_all)
    axs[0].set_title('original')
    for i in range(len(i_all_chunks)):
        axs[i+1].set_title((i+1)/len(i_all_chunks))
        axs[i+1].plot(i_all_chunks[i])
    # plt.savefig(fig_name, dpi=100)



def vectorized_stride_v2_fault(error_ind, array, fc, max_time=None, sub_window_size=None, stride_size=None):

    # ref: https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
    start = 0 #start = clearing_time_index + 1 - sub_window_size + 1

    sub_windows = (start +
        np.expand_dims(np.arange(sub_window_size), 0) +
        # Create a rightmost vector as [0, V, 2V, ...].
        np.expand_dims(np.arange(max_time + 1, step=stride_size), 0).T
    )

    i_chunks = array[sub_windows]
    y_chunk_label = np.full((1,len(i_chunks)), fault_classes_names[fc])[0]
    return i_chunks, y_chunk_label



def split_(i_all, error_starting_time, error_duration, fc):
    """
    input arguments:
    i_all: input array generated from .mat file
    error duration: 0.1/0.2/0.3
    index of error (ei): 0 if error at 0 to 0.1, 1 if error at 0.1 to 0.2 and so on,
    # 0-0.1==>0, 0.1-0.2==>1, 0.2-0.3==>2, 0.3-0.4==>3, 0.4-0.5==>4, 0.5-0.6==>5 ,0.6-0.7==>6, 0.7-0.8==>7, 0.8-0.9==>8, 0.9-1==>9,
    fault class: ab, ac, etc
    output:
    return the x and y arrays where x is divided in chunks and y is their respective labels
    """
    no_of_chunks = len(i_all)/(len(i_all)*error_duration) #

    error_index = int(error_starting_time *  no_of_chunks) # error_starting_time * no_of_chunks # index of fault at which time period the fault starts.
    #--- to normalize the i_all data here
    #--- reading normalize data https://www.digitalocean.com/community/tutorials/normalize-data-in-python

    i_all = 2*(i_all - i_all.min()) / (i_all.max() - i_all.min()) -1

    #---- separate the i_all into faulty and non faulty signals
    i_all_fault = i_all[error_index*10000:] # it clips the portion of the signal before fault.

    if fc == 'nf':
        i_all_chunks, y_chunk_label  = vectorized_stride_v2_fault(error_starting_time, i_all_fault, fc, max_time=1500, sub_window_size=10500, stride_size=100)
    else:
        i_all_chunks, y_chunk_label  = vectorized_stride_v2_fault(error_starting_time, i_all_fault, fc, max_time=1500, sub_window_size=10500, stride_size=100)

    plot_single_data(i_all,i_all_chunks, y_chunk_label, no_of_chunks, "{}_{}_{}.png".format(error_duration, error_starting_time, fc))

    return np.asarray(i_all_chunks), np.asarray(y_chunk_label)

def first_chars(x):
    return(int(x.split('_')[0]))

def load_mot(mat_lis_5):
    for i in mat_lis_5:
        sf_mat = loadmat(i)
        # print(sf_mat.keys())
        if 'current5' in sf_mat:
            ia= sf_mat['current5']
        elif 'current6' in sf_mat:
            ib= sf_mat['current6']
        elif 'current7' in sf_mat:
            ic= sf_mat['current7']
        elif 'current4' in sf_mat:
            ig= - sf_mat['current4'] # - sign for making ig current in phase with the line current. 
        elif 'tout' in sf_mat:
            to= sf_mat['tout']
        # to /= np.max(np.abs(to),axis=0)
    i_all = np.concatenate((ia,ib,ic,ig), axis=1) # to place numpy arrays horizentally-- coloumn wise
    
    # Plot the resampled data
    folder_path = os.path.split(i)[0]
    index=0
    index_value=['IA(A)', 'IB(A)', 'IC(A)', 'IG(A)']
    plt.figure(figsize=(10, 6))
    for col in [ia,ib,ic,ig]:
                plt.plot(to[10000:40000], col[10000:40000], label=index_value[index])
                # plt.plot(to[:-60001,:], col[:-60001,:], label=index_value[index])
                index+=1         
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Current (A)', fontsize=12)
    plt.title(f'Current vs. Time for {os.path.splitext((os.path.split(i)[-1]))[0].split("_")[1]}', fontsize=14)
    plt.legend()
    plt.grid(True)
    fault_name =ii = (os.path.splitext(os.path.split(i)[-1])[0].split("_")[1])
    output_file = os.path.join(folder_path, f"{fault_name}_fault.png")
    # output_file = os.path.join(folder_path, f"{os.path.splitext(os.path.split(i)[-1])[0]).split("_")[1]}_fault.png")
    # plt.savefig(output_file)
    plt.close()        
    i_all=i_all[:-1,:] # to remove the last row to make it even and divisible in odd chunks
    return i_all, to[:-1,:]

def data_generator(main_folder=None):

    sub_folder_list = os.listdir(main_folder)
    i=0
    sf_file_list=[]
    train_x = []
    train_y=[]
    for fo in sub_folder_list:
        error_duration = float('%.1f'%(float(fo.split('_')[1]) - float(fo.split('_')[0]))) 
        error_starting_time = float(fo.split('_')[0]) # extract information from the folder name

        fo_path = os.path.join(os.getcwd(), main_folder, fo)
        sub_folder_list = os.listdir(fo_path)

        for sf in sorted(sub_folder_list, key = first_chars):
            fault_class = sf.split('_')[1] # to find the fault class from the .mat file name in sf variable
            sf_file = os.path.join(os.getcwd(), fo_path, sf)
            sf_file_list.append(sf_file)
            if len(sf_file_list) == 5:
                i_all, to = load_mot(sf_file_list) # read .mat files and generate the array variable for all currents
                sf_file_list=[] # initialize back the list to keep next five files
                i_all_chunks_all, y_chunk_label_all = split_(i_all, error_starting_time, error_duration, fault_class) # to divide a numpy array in to different chunks and its labeling

                if i==0:
                    train_x1 = i_all_chunks_all
                    train_y1 = y_chunk_label_all
                else:
                    train_x1 = np.concatenate((train_x1, i_all_chunks_all), axis=0)
                    train_y1 = np.concatenate((train_y1,y_chunk_label_all), axis=0)

                i+=1

    #--- convert the train_x in to the shape: (no. of samples, no. of values (rows) in each sample, no. of current variables in each sample)

    train_x = train_x1
    train_y = train_y1
    # to convert y into one-hot encoding
    ohe = preprocessing.OneHotEncoder() # Define the One-hot Encoder
    train_y = train_y.reshape(-1, 1) # Reshape data
    # Fit and transform training data
    ohe.fit(train_y)
    train_y = ohe.transform(train_y).toarray()
    train_y = np.expand_dims(train_y, axis=2)

    print('-------data statistics------')
    print('total number of .mat files')
    print('-------- train x-------')
    print(len(train_x), type(train_x), train_x.shape)
    print('-------train y---------')
    print(len(train_y), type(train_y), train_y.shape)

    return (train_x, train_y)
