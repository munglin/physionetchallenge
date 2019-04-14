#!/usr/bin/env python3

import sys
import numpy as np
import os, shutil, zipfile

def get_sepsis_score(data):
    x_mean = np.array([
        83.8996, 97.0520,  36.8055,  126.2240, 86.2907,
        66.2070, 18.7280,  33.7373,  -3.1923,  22.5352,
        0.4597,  7.3889,   39.5049,  96.8883,  103.4265,
        22.4952, 87.5214,  7.7210,   106.1982, 1.5961,
        0.6943,  131.5327, 2.0262,   2.0509,   3.5130,
        4.0541,  1.3423,   5.2734,   32.1134,  10.5383,
        38.9974, 10.5585,  286.5404, 198.6777])
    x_std = np.array([
        17.6494, 3.0163,  0.6895,   24.2988, 16.6459,
        14.0771, 4.7035,  11.0158,  3.7845,  3.1567,
        6.2684,  0.0710,  9.1087,   3.3971,  430.3638,
        19.0690, 81.7152, 2.3992,   4.9761,  2.0648,
        1.9926,  45.4816, 1.6008,   0.3793,  1.3092,
        0.5844,  2.5511,  20.4142,  6.4362,  2.2302,
        29.8928, 7.0606,  137.3886, 96.8997])
    c_mean = np.array([60.8711, 0.5435, 0.0615, 0.0727, -59.6769, 28.4551])
    c_std = np.array([16.1887, 0.4981, 0.7968, 0.8029, 160.8846, 29.5367])

    x = data[:, 0:34]
    c = data[:, 34:40]
    x_norm = np.nan_to_num((x - x_mean) / x_std)
    c_norm = np.nan_to_num((c - c_mean) / c_std)

    beta = np.array([
        0.02,  0.01, 0.21,  0.04, 0.004,
        -0.09, 0.07, -0.03, -0.09, 0.11,
        0.76,  0.03, 0.03,  -0.01, 0.01,
        0.1,  0.03, -0.10, -0.05, 0.005,
        -0.01, 0.05, 0.004,  -0.0007, 0.005,
        0.004,  0.07, -0.02, 0.05,  -0.8,
        0.03,  0.1, 0.62,  -0.004, 0.01,
        0.008,  0.03, -0.2, -0.01, 0.19])
    rho = 14
    nu = 0.5

    from keras.models import load_model
    model = load_model('my_model.h5')
    
    Xtest = data.iloc[:,[0,1,3,4,6,34,35,38]]
    mean = np.array([83.8051,97.1538,123.321,82.5631,18.538,61.6434,0.55945])
    sd = np.array([14.6318,2.0955,17.83,12.5797,3.40572,16.4827,0.4965])
    Xtest = (Xtest - mean)/sd
        
    scores = model.predict(Xtest)
    labels = (scores > 0.50)
    return (scores, labels)

def read_challenge_data(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # ignore SepsisLabel column if present
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        data = data[:, :-1]

    return data

if __name__ == '__main__':
    # get input files
    tmp_input_dir = 'tmp_inputs'
    input_files = zipfile.ZipFile(sys.argv[1], 'r')
    input_files.extractall(tmp_input_dir)
    input_files.close()
    input_files = sorted(f for f in input_files.namelist() if os.path.isfile(os.path.join(tmp_input_dir, f)))

    # make temporary output directory
    tmp_output_dir = 'tmp_outputs'
    try:
        os.mkdir(tmp_output_dir)
    except FileExistsError:
        pass

    n = len(input_files)
    output_zip = zipfile.ZipFile(sys.argv[2], 'w')

    # make predictions for each input file
    for i in range(n):
        # read data
        input_file = os.path.join(tmp_input_dir, input_files[i])
        data = read_challenge_data(input_file)

        # make predictions
        if data.size != 0:
            (scores, labels) = get_sepsis_score(data)

        # write results
        file_name = os.path.split(input_files[i])[-1]
        output_file = os.path.join(tmp_output_dir, file_name)
        with open(output_file, 'w') as f:
            f.write('PredictedProbability|PredictedLabel\n')
            if data.size != 0:
                for (s, l) in zip(scores, labels):
                    f.write('%g|%d\n' % (s, l))
        output_zip.write(output_file)

    # perform clean-up
    output_zip.close()
    shutil.rmtree(tmp_input_dir)
    shutil.rmtree(tmp_output_dir)
