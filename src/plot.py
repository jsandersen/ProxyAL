import matplotlib.pyplot as plt
import numpy as np

def plot_f1_merge(dirs, colors, label, dataset, encoding, name, ylim_lim = None):
    for i in range(len(dirs)): 
        
        f1_mic_list_A, f1_mac_list_A, _, _, _, _ = _load_results(dirs[i])
        mean_f1_mic_A, _ = _get_mean_and_std(f1_mic_list_A)
        
        
        prep = _prepare(mean_f1_mic_A, 1)[1:]
        plt.plot(prep[:, 0], prep[:, 1], color=colors[i], label=label[i].replace("_", "*"))

        
        print(label[i], ': ', round(prep[:, 1][-1], 4))
    
    
    for i in range(len(dirs)): 
        
        f1_mic_list_A, f1_mac_list_A, _, _, _, _ = _load_results(dirs[i])
        mean_f1_mac_A, _ = _get_mean_and_std(f1_mac_list_A)
        
        
        prep = _prepare(mean_f1_mac_A, 1)[1:]
        plt.plot(prep[:, 0], prep[:, 1], '-.', color=colors[i])

        
        print(label[i], ': ', round(prep[:, 1][-1], 4))
    
    plt.grid(True)
    plt.locator_params(axis="y", nbins=7)

    
    plt.xlim([0, 500])
    if ylim_lim:
        plt.ylim(ylim_lim)
    
    plt.legend(loc=4)    
    plt.xlabel('Iterations')
    plt.ylabel('F1-Score')
    plt.title(f'{dataset} {encoding}')  
    
    plt.savefig(f'./plots/mean_micro_f1_{dataset}_{encoding}_{name}_merge_plot.pdf', bbox_inches='tight')  
    plt.show()
    

def plot_f1(dirs, colors, label, dataset, encoding, name, ylim_micro = None, ylim_macro = None):
    for i in range(len(dirs)): 
        
        f1_mic_list_A, f1_mac_list_A, _, _, _, _ = _load_results(dirs[i])
        mean_f1_mic_A, _ = _get_mean_and_std(f1_mic_list_A)
        
        
        prep = _prepare(mean_f1_mic_A, 1)[1:]
        plt.plot(prep[:, 0], prep[:, 1], color=colors[i], label=label[i])

        
        print(label[i], ': ', round(prep[:, 1][-1], 4))
    
    plt.title('Mean Micro-F1')
    import numpy as np
    #if ylim_micro:
    #    plt.ylim(ylim_micro)
    plt.locator_params(axis="y", nbins=7)
    plt.grid(True)
    
    plt.xlim([0, 500])
    plt.legend()
    plt.savefig(f'./plots/mean_micro_f1_{dataset}_{encoding}_{name}_plot.pdf', bbox_inches='tight')  
    
    plt.show()
    
    for i in range(len(dirs)): 
        f1_mic_list_A, f1_mac_list_A, _, _, _, _ = _load_results(dirs[i])
        _, std_f1_mic_A = _get_mean_and_std(f1_mic_list_A)
        
        
        prep = _prepare(std_f1_mic_A, 1)[1:]
        plt.plot(prep[:, 0], prep[:, 1], color=colors[i], label=label[i])

        
        print(label[i], ': ', round(prep[:, 1][-1], 4))
    
    plt.title('Std. Micro-F1')
    plt.grid(True)
    plt.xlim([0, 500])
    plt.legend()    
    plt.savefig(f'./plots/std_micro_f1_{dataset}_{encoding}_{name}_plot.pdf', bbox_inches='tight') 
           
    plt.show()
    
    for i in range(len(dirs)): 
        
        f1_mic_list_A, f1_mac_list_A, _, _, _, _ = _load_results(dirs[i])
        mean_f1_mac_A, _ = _get_mean_and_std(f1_mac_list_A)
        
        
        prep = _prepare(mean_f1_mac_A, 1)[1:]
        plt.plot(prep[:, 0], prep[:, 1], color=colors[i], label=label[i])

        
        print(label[i], ': ', round(prep[:, 1][-1], 4))
    
    plt.title('Mean Marco-F1')  
    plt.grid(True)
    plt.locator_params(axis="y", nbins=7)
    #if ylim_macro:
    #    plt.ylim(ylim_macro)
    plt.xlim([0, 500])
    plt.legend()    
    plt.savefig(f'./plots/mean_macro_f1_{dataset}_{encoding}_{name}_plot.pdf', bbox_inches='tight') 
        
    plt.show()
    
    for i in range(len(dirs)): 
        f1_mic_list_A, f1_mac_list_A, _, _, _, _ = _load_results(dirs[i])
        _, std_f1_mac_A = _get_mean_and_std(f1_mac_list_A)
        
        prep = _prepare(std_f1_mac_A, 1)[1:]
        plt.plot(prep[:, 0], prep[:, 1], color=colors[i], label=label[i])
        
        print(label[i], ': ', round(prep[:, 1][-1], 4))
    
    plt.title('Std. Marco-F1')
    plt.xlim([0, 500])
    plt.legend()        
    plt.savefig(f'./plots/std_macro_f1_{dataset}_{encoding}_{name}_plot.pdf', bbox_inches='tight') 
    plt.show()
        
def _load_results(root_dir):
    f1_mic_list = np.load('%s/f1_micro.npy' % root_dir)
    f1_mac_list = np.load('%s/f1_macro.npy' % root_dir)
    c_list = np.load('%s/class_distributions.npy' % root_dir, allow_pickle=True)
    times_train_list = np.load('%s/times_training.npy' % root_dir)
    times_inf_list = np.load('%s/times_inference.npy' % root_dir)
    used_index_meta_list = np.load('%s/used_training_index.npy' % root_dir, allow_pickle=True)
    
    return f1_mic_list, f1_mac_list, c_list, times_train_list, times_inf_list, used_index_meta_list

def _get_mean_and_std(x):
    mean = np.array(x).mean(axis=0)
    std = np.array(x).std(axis=0)
    return mean, std

def _prepare(x, batch_site):
    res = []
    for i in range(len(x)):
        res.append([i*batch_site, x[i]])
    return np.array(res)


# libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_balancy(load_dir, dataset, encoding, name):
    c_list = np.load('%s/class_distributions.npy' % load_dir, allow_pickle=True)
    plot_c(c_list, dataset, encoding, name)

def plot_c(c_list, dataset, encoding, name):
    meta_l = []
    for k in range(5):
        l = [[] for i in range(len(c_list[0, k]))]

        for i in c_list[k]:
            for j in range(len(i)):
                l[j].append(i[j])

        meta_l.append(l)
    meta_l=np.array(meta_l).mean(axis=0)

    groups = {f'group_{i}':meta_l[i] for i in range(len(l))}

    # Make data
    data = pd.DataFrame(groups)

    # We need to transform the data from raw data to percentage (fraction)
    data_perc = data.divide(data.sum(axis=1), axis=0)
    groups_list = [data_perc[f'group_{i}'] for i in range(len(l))]

    # Make the plot
    plt.stackplot(range(501),  *groups_list, labels=['A','B','C'])
    #plt.legend(loc='upper left')
    plt.margins(0,0)
    plt.title(f'{dataset} {encoding}')  
    plt.xlabel('Iterations')
    plt.ylabel('Class Distribution (%)')
    plt.savefig(f'./plots/mean_micro_f1_{dataset}_{encoding}_{name}_c_plot.pdf', bbox_inches='tight')  
    plt.show()