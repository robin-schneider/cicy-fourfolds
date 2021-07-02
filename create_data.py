"""
Downloads the cicy four-fold data and creates numpy arrays to work with.
"""
import numpy as np
import ast
import os as os

def find_percentile(h, perc=0.01):
    hmin, hmax = np.min(h), np.max(h)
    down, up = hmin, hmax
    for i in range(hmin, hmax+1):
        if np.sum(h < i)/len(h) > perc:
            down = i-1
            break
    for i in range(hmax+1, hmin, -1):
        if np.sum(h > i)/len(h) > perc:
            up = i+1
            break
    return (h < down) + (h > up)

if __name__ == '__main__':

    if not os.path.exists('cicy4list.txt'):
        import zipfile
        import wget
        # download
        url = 'http://www-thphys.physics.ox.ac.uk/projects/CalabiYau/Cicy4folds/cicy4list.zip'
        # unzip
        wget.download(url, 'cicy4list.zip')
        with zipfile.ZipFile('cicy4list.zip', 'r') as zip_ref:
            zip_ref.extractall('')

    # create arrays; dimensions are from the paper
    # https://arxiv.org/pdf/1405.2073.pdf
    h = np.zeros((921497, 4), dtype=np.int)
    m = np.zeros((921497, 16, 20), dtype=np.int)
    direct = np.zeros((921497), dtype=np.bool)

    #read in data
    with open('cicy4list.txt', 'r') as file:
        data = file.read()

    # manipulate strings
    data = data.replace('True}, \n', 'True};')
    data = data.replace('True},', 'True};')
    data = data.replace('False}, \n', 'False};')
    data = data.replace('False},', 'False};')
    data = data.replace('Null}, \n', 'Null};')
    data = data.replace('Null},', 'Null};')
    data = data.replace('{', '[').replace('}', ']')
    data = data.replace('Null', 'False')
    data = data.replace('\n', '')

    # split into one list per entry
    data = data.split(';')

    # iterate over lists and save relevant data
    # this will be somewhat memory expensive
    for i, line in enumerate(data):
        line = line.strip()
        if i == 0:
            line = line[1:]
        if i >= 921496:
            line = line[:-1]
        lline = ast.literal_eval(line)
        direct[i] = lline[5]
        h[i] += np.array(lline[6:10])
        m[i, 0:lline[1], 0:lline[2]] += np.array(lline[3])

    # make data folder
    if not os.path.exists('data'):
        os.makedirs('data')

    # rescale conf to the interval [0,1]
    m = m/np.max(m)

    # save data
    np.save(os.path.join('data', 'conf'), m)
    np.save(os.path.join('data', 'hodge'), h)
    np.save(os.path.join('data', 'direct'), direct)

    hodge = h[~direct]

    # make tails
    h11tails = find_percentile(hodge[:,0], 0.005)
    h21tails = find_percentile(hodge[:,1], 0.005)
    h31tails = find_percentile(hodge[:,2], 0.005)
    h22tails = find_percentile(hodge[:,3], 0.005)

    print('percentage in tails {}'.format(
        np.sum(h11tails + h21tails + h31tails + h22tails)/len(hodge)))
    
    # save tails
    np.save(os.path.join('data', 'conf_tails'),
        h11tails + h21tails + h31tails + h22tails)