y = []
import h5py
from os import listdir

dir_list = ["newdataset"]
N=0
for directory in dir_list:
    for filename in listdir(directory):
        try:
            # load the image
            f = h5py.File(directory + '\\' + filename, 'r') #Open mat file for reading
      
      
            cjdata = f['cjdata'] #<HDF5 group "/cjdata" (5 members)>
      
      
            label = int(cjdata.get('label')[0,0])
            y.append([label])
        except:
            pass
              
print(y)