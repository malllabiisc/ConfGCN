import os, sys, pdb, numpy as np, scipy.sparse as sp, random
import argparse, codecs, pickle, time, json, uuid
import networkx as nx
import logging, logging.config

from collections import defaultdict as ddict
from pprint import pprint

np.set_printoptions(precision=4)

def set_gpu(gpus):
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def shape(tensor):
	s = tensor.get_shape()
	return tuple([s[i].value for i in range(0, len(s))])


def debug_nn(res_list, feed_dict):
	import tensorflow as tf
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	summ_writer = tf.summary.FileWriter("tf_board/debug_nn", sess.graph)
	res = sess.run(res_list, feed_dict = feed_dict)
	return res

def get_logger(name, log_dir, config_dir):
	config_dict = json.load(open( config_dir + 'log_config.json'))
	config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

"""
Most of the functions below are taken from https://github.com/tkipf/gcn
"""

def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
                index.append(int(line.strip()))
        return index


def sample_mask(idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)


def load_data(dataset_str, args):
        """
        Loads input data from gcn/data directory

        ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
                (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
                object;
        ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

        All objects above must be saved using python pickle module.

        :param dataset_str: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        if 'nell' in dataset_str:
                data_dict = pickle.load(open('./data/{}_data.pkl'.format(dataset_str), 'rb'), encoding='latin1')
                x, y, tx, ty, allx, ally, graph = data_dict['x'], data_dict['y'], data_dict['tx'], data_dict['ty'], data_dict['allx'], data_dict['ally'], data_dict['graph']

                index = list(range(allx.shape[0])) + data_dict['test.index']
                remap = {x: x for x in range(allx.shape[0])}
                remap.update({i+allx.shape[0]: x for i, x in enumerate(data_dict['test.index'])})
                remap_inv = {v: k for k, v in remap.items()}

                graph_new = ddict(list)
                for key, val in graph.items():
                        if key not in remap_inv: continue
                        graph_new[remap_inv[key]] = [remap_inv[v] for v in val if v in remap_inv]

                graph = graph_new
                test_idx_reorder = [remap_inv[x] for x in data_dict['test.index']]
        else:
                for i in range(len(names)):
                        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                                if sys.version_info > (3, 0):
                                        objects.append(pickle.load(f, encoding='latin1'))
                                else:
                                        objects.append(pickle.load(f))

                x, y, tx, ty, allx, ally, graph = tuple(objects)
                test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))

        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
                # Fix citeseer dataset (there are some isolated nodes in the graph)
                # Find isolated nodes, add them as zero-vecs into the right position
                test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
                tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
                tx_extended[test_idx_range-min(test_idx_range), :] = tx
                tx = tx_extended
                ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
                ty_extended[test_idx_range-min(test_idx_range), :] = ty
                ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask   = sample_mask(idx_val, labels.shape[0])
        test_mask  = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val   = np.zeros(labels.shape)
        y_test  = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def sparse_to_tuple(sparse_mx):
        """Convert sparse matrix to tuple representation."""
        def to_tuple(mx):
                if not sp.isspmatrix_coo(mx):
                        mx = mx.tocoo()
                coords = np.vstack((mx.row, mx.col)).transpose()
                values = mx.data
                shape = mx.shape
                return coords, values, shape

        if isinstance(sparse_mx, list):
                for i in range(len(sparse_mx)):
                        sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
                sparse_mx = to_tuple(sparse_mx)

        return sparse_mx


def preprocess_features(features, noTuple=False):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)

        if noTuple:     return features
        else:           return sparse_to_tuple(features)

def normalize_adj(adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj, noTuple=False):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
        
        if noTuple:     return adj_normalized
        else:           return sparse_to_tuple(adj_normalized)


def get_ind_from_adj(adj):
        lens = [len(list(np.nonzero(row)[0])) for row in adj]
        ind  = np.zeros((adj.shape[0], np.max(lens)), dtype=np.int64)
        mask = np.zeros((adj.shape[0], np.max(lens)), dtype=np.float32)

        for i, row in enumerate(adj):
                J = np.nonzero(row)[1]
                for pos, j in enumerate(J):
                        ind[i][pos]  = j
                        mask[i][pos] = 1

        return ind, mask