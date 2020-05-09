import numpy as np
"""
# load data for the experiment on CB6133
def load_data(debug=False, num_step=10, num_data=300):
    train_idx = 5600
    test_idx = 5877
    val_idx = 6133
    npf = np.load("./data/protein_data.npz", 'r')
    data, labels = npf["arr_0"], npf["arr_1"]
    train_data = {}; val_data = {}; test_data = {}
    if not debug:
        train_data["X"] = data[0:train_idx]
        train_data["y"] = labels[0:train_idx]
        val_data["X"] = data[test_idx:val_idx]
        val_data["y"] = labels[test_idx:val_idx]
    else:
        train_data["X"] = data[0:train_idx][:num_data,:num_step]
        train_data["y"] = labels[0:train_idx][:num_data, :num_step]
        val_data["X"] = data[0:train_idx][:num_data,:num_step]
        val_data["y"] = labels[0:train_idx][:num_data, :num_step]  
    test_data["X"] = data[train_idx+5:test_idx]
    test_data["y"] = labels[train_idx+5:test_idx]
    return train_data, val_data, test_data
"""
# load data for the experiment on CB513
def load_data(debug=False, num_step=10, num_data=300):
    npf = np.load("./data/pdb_filter.npz", 'r')
    data, labels = npf["arr_0"], npf["arr_1"]
    train_data = {}; val_data = {}; test_data = {}
    if not debug:
        train_data["X"] = data
        train_data["y"] = labels
    else:
        train_data["X"] = data[:num_data,:num_step]
        train_data["y"] = labels[:num_data, :num_step]
    
    npf = np.load("./data/cb_513.npz", 'r')
    data, labels = npf["arr_0"], npf["arr_1"]
    val_data["X"] = data
    val_data["y"] = labels

    test_data["X"] = data
    test_data["y"] = labels
    return train_data, val_data, test_data
    
def get_sample_data(data, batch_size):
    idxes = np.random.randint(0, data["X"].shape[0], [batch_size])
    X = data["X"][idxes]
    y = data["y"][idxes]
    return X, y
