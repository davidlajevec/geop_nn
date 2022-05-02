import numpy as np

def load_data():
    return np.load('data/all_data.npy', allow_pickle=True, mmap_mode ='r')

def data_processing(geop, periodic = True):
    #geop_merid_squeez = np.zeros((len(geop), 60, 360))
    geop_merid_squeez = np.zeros((len(geop), 45, 360))
    # 181 -> 180 every 3rd point latitudinal
    for i in range(179):
        if i%4 == 0:
            geop_merid_squeez[:,i//4,:] = (geop[:,i,:]+geop[:,i+1,:])/2
    # every 3rd point longitudinal
    #geop_squeezed = np.zeros((len(geop), 60, 120))       
    geop_squeezed = np.zeros((len(geop), 45, 90))       
    for i in range(360):
        if i%4 == 0:
            geop_squeezed[:,:,i//4] = geop_merid_squeez[:,:,i]
    if not periodic:
        return np.expand_dims(geop_squeezed, axis=3)

    # normalization [0 ... 1]
    M = 59178.25        # max geopotential in all datasets
    m = 42737.34375      # min geopotential in all datasets
    geop_norm = (geop_squeezed-m)/(M-m)

    periodic_geop = np.zeros((len(geop), 45+3, 90+2))
    periodic_geop[:,1:-2,0] = geop_norm[:,:,-1]
    periodic_geop[:,1:-2,-1] = geop_norm[:,:,0]  
    periodic_geop[:,1:-2,1:-1] = geop_norm
    return np.expand_dims(periodic_geop, axis=3)

def train_valid_data(skip_day=5,split=0.1):
    data = data_processing(load_data()[:-365+skip_day]).astype('float32')
    split_ind = int((1-split)*len(data))
    train_X, train_y = data[:split_ind], data[skip_day:split_ind+skip_day]
    val_X, val_y = data[split_ind:-skip_day], data[split_ind+skip_day:]
    return train_X, train_y, val_X, val_y 

def test_data(skip_day=5):
    data = data_processing(load_data()[-365:]).astype('float32')
    test_X, test_y = data[:-skip_day], data[skip_day:]
    return test_X, test_y

if __name__ == "__main__":
    test_data()