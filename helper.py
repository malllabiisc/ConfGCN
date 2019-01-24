from pymongo import MongoClient
import os, sys, pdb, numpy as np, random, argparse, codecs, pickle, time, json, queue, re
from pprint import pprint
from collections import defaultdict as ddict
from joblib import Parallel, delayed
import numpy as np, sys, unicodedata, requests, os, random, pdb, requests, json, itertools, argparse, pickle
from random import randint
import networkx as nx
import scipy.sparse as sp
from pprint import pprint
import logging, logging.config, itertools, pathlib
from sklearn.metrics import precision_recall_fscore_support
import gzip, queue, threading
from threading import Thread
from scipy.stats import describe as des

np.set_printoptions(precision=4)

def mergeList(list_of_list):
	return list(itertools.chain.from_iterable(list_of_list))

def checkFile(filename):
	return pathlib.Path(filename).is_file()

def getWord2vec(wrd_list):
	dim = 300
	embeds = np.zeros((len(wrd_list), dim), np.float32)
	embed_map = {}

	res = db_word2vec.find({"_id": {"$in": wrd_list}})
	for ele in res:
		embed_map[ele['_id']] = ele['vec']

	count = 0
	for wrd in wrd_list:
		if wrd in embed_map: 	embeds[count, :] = np.float32(embed_map[wrd])
		else: 			embeds[count, :] = np.random.randn(dim)
		count += 1

	return embeds


def getEmbeddings(embed_loc, wrd_list, embed_dims):
	embed_list = []

	wrd2embed = {}
	for line in open(embed_loc):
		data = line.strip().split(' ')
		wrd, embed = data[0], data[1:]
		embed = list(map(float, embed))
		wrd2embed[wrd] = embed

	for wrd in wrd_list:
		if wrd in wrd2embed: 	embed_list.append(wrd2embed[wrd])
		else: 	
			print('Word not in embeddings dump')
			embed_list.append(np.random.randn(embed_dims))

	return np.array(embed_list, dtype=np.float32)


def len_key(tp):
	return len(tp[1])

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

def stanford_tokenize(text):
	res = callnlpServer(text)
	toks = [ele['word'] for ele in res['tokens']]
	return toks


def is_number(wrd):
	isNumber = re.compile(r'\d+.*')
	return isNumber.search(wrd)

isNumber = re.compile(r'\d+.*')
def norm_word(word):
	# if isNumber.search(word.lower()):
	# 	return '---num---'
	# if re.sub(r'\W+', '', word) == '':
	# 	return '---punc---'
	# else:
	return word.lower()
		

def is_int(s):
	try:
		int(s)
		return True
	except ValueError:
		return False

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

def partition(lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def getChunks(inp_list, chunk_size):
	return [inp_list[x:x+chunk_size] for x in range(0, len(inp_list), chunk_size)]

def mergeList(list_of_list):
	return list(itertools.chain.from_iterable(list_of_list))


pdb_multi  = '!import code; code.interact(local=vars())'
pdb_global = 'globals().update(locals())'