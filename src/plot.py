import matplotlib.pyplot as plt
import numpy as np

def plot_f1(dirs, colors, label, dataset, encoding, name):
    for i in range(len(dirs)): 
        
        f1_mic_list_A, f1_mac_list_A, _, _, _, _ = _load_results(dirs[i])
        mean_f1_mic_A, _ = _get_mean_and_std(f1_mic_list_A)
        
        
        prep = _prepare(mean_f1_mic_A, 1)[1:]
        plt.plot(prep[:, 0], prep[:, 1], color=colors[i], label=label[i])

        
        print(label[i], ': ', round(prep[:, 1][-1], 4))
    
    plt.title('Mean Mirco-F1')
    plt.xlim([0, 500])
    plt.legend()
    plt.savefig(f'./plots/mean_micro_f1_{dataset}_{encoding}_{name}_plot.pdf')  
        
    plt.show()
    
    for i in range(len(dirs)): 
        f1_mic_list_A, f1_mac_list_A, _, _, _, _ = _load_results(dirs[i])
        _, std_f1_mic_A = _get_mean_and_std(f1_mic_list_A)
        
        
        prep = _prepare(std_f1_mic_A, 1)[1:]
        plt.plot(prep[:, 0], prep[:, 1], color=colors[i], label=label[i])

        
        print(label[i], ': ', round(prep[:, 1][-1], 4))
    
    plt.title('Std. Mirco-F1')
    plt.xlim([0, 500])
    plt.legend()    
    plt.savefig(f'./plots/std_micro_f1_{dataset}_{encoding}_{name}_plot.pdf') 
            
    plt.show()
    
    for i in range(len(dirs)): 
        
        f1_mic_list_A, f1_mac_list_A, _, _, _, _ = _load_results(dirs[i])
        mean_f1_mac_A, _ = _get_mean_and_std(f1_mac_list_A)
        
        
        prep = _prepare(mean_f1_mac_A, 1)[1:]
        plt.plot(prep[:, 0], prep[:, 1], color=colors[i], label=label[i])

        
        print(label[i], ': ', round(prep[:, 1][-1], 4))
    
    plt.title('Mean Marco-F1')   
    plt.xlim([0, 500])
    plt.legend()    
    plt.savefig(f'./plots/mean_macro_f1_{dataset}_{encoding}_{name}_plot.pdf') 
        
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
    plt.savefig(f'./plots/std_macro_f1_{dataset}_{encoding}_{name}_plot.pdf') 
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