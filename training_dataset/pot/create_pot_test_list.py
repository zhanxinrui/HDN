from os.path import join
from os import listdir
import json
import numpy as np
num_types = 1
test_list = []
start_type = 1
for i in range(25,31):
    for j in range(start_type,num_types+1):
        print('i,j',i,j)
        test_list.append('V%02d_%d\n'%(i,j))

# for i in range(17, 31):
#     for j in range(start_type, num_types + 1):
#         print('i,j', i, j)
#         test_list.append('V%02d_%d\n' % (i, j))


with open('./POT_train/testing_set.txt','w') as f:
    f.writelines(test_list)
    # for v in test_list:
    #     f.write(v)
