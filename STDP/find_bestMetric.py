import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from os import walk


if __name__ == "__main__":
    print sys.argv[1]
    if len(sys.argv) < 2:
        print 'useage: find_bestMetric path'

    # directory = sys.argv[2]
    path = sys.argv[1]


    d = []
    for (dirpath, dirnames, filenames) in walk(path):
        d.extend(dirnames)
        break

    # print d

    for i,sub_d in enumerate(d):

        print 'directory: ', sub_d
        if i < 9:
            continue
        sub_path = path  + sub_d + '/patterns/'
        # print 'reading files from ', sub_path, '...'

        f = []
        for (dirpath, dirnames, filenames) in walk(sub_path):
            f.extend(filenames)
            break
        # print f

        if len(f) == 0:
            print 'EMPTY'
            continue
        # print len(f)
        # print f

        metrics = np.zeros(len(f))
        for i,n in enumerate(f):
            metrics[i] = (np.load(sub_path + n)).item(0)['metric'][0]
        max_index = np.argmax(metrics)
        max_paramfile = f[max_index]
        max_param = (np.load(sub_path + max_paramfile).item(0))
        print 'best_metric: ', metrics[max_index]
        print 'corresponding parameter_file and parameters: '
        print max_paramfile
        print max_param
        print
    # print trial.item(0)['metric'][0]
    # print len(np.array(trial))
#for each file in directory

#load content (dictionary)
#check the metric element
