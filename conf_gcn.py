from helper import *
import tensorflow as tf

class ConfGCN(object):

	def load_data(self):
		"""
		Reads the data from pickle file

		Parameters
		----------
		self.p.data: 		Name of the dataset -- citeseer/cora/pubmed/coraml'

		Returns
		-------
		self.adj:		Adjacency list of the graph
		self.features:		Given initial node features
		self.y_train:		Labels corresponding to labeled training nodes
		self.y_valid:		Labels of nodes in the validcation data
		self.y_test:		Labels of nodes in the test data
		self.train_mask:	Contains 1 for nodes which are part of training data. Same holds for valid and test.
		self.num_nodes:		Number of nodes in the graph
		self.adj_ind:		Storing graph edge information as adjacency list
		self.adj_ind_mask:	Mask for padded indices in adj_ind
		self.input_dim:		Input node feature size
		self.output_dim:	Number of classes to which nodes can belong
		"""

		print("loading data")
		self.adj, self.features, self.y_train, self.y_val, self.y_test, self.train_mask, self.val_mask, self.test_mask = load_data(self.p.data, self.p)
		self.features 	  = preprocess_features(self.features, noTuple=False)
		self.adj  	  = preprocess_adj(self.adj, noTuple=True).todense()
		self.adj_ind, self.adj_ind_mask = get_ind_from_adj(self.adj)

		self.num_nodes    = self.features[2][0]
		self.input_dim    = self.features[2][1]
		self.output_dim   = self.y_train.shape[1]

		# Label mask
		self.label_cond = np.zeros((self.num_nodes), np.bool)
		for i in range(self.num_nodes):
			if np.sum(self.y_train[i]) != 0:
				self.label_cond[i] = 1

		self.placeholders = {
			'features': 		tf.sparse_placeholder(tf.float32, shape=tf.constant(self.features[2], dtype=tf.int64)), # features[2] = shape of the input
			'labels': 		tf.placeholder(tf.float32, 	  shape=(None, self.y_train.shape[1])),		   	# batch x 7(num_classes)
			'labels_mask': 		tf.placeholder(tf.int32),
			'adj_ind':              tf.placeholder(tf.int32),
			'adj_ind_mask':         tf.placeholder(tf.float32),
			'dropout': 		tf.placeholder_with_default(0.,   shape=()),						# Dropout
			'num_features_nonzero': tf.placeholder(tf.int32)  								# helper variable for sparse dropout
		}

	def create_feed_dict(self, split='train'):
		"""
		Creates a feed dictionary for the batch

		Parameters
		----------
		split:		data split -- train/test/valid

		Returns
		-------
		feed:		Feed dictionary to be fed during sess.run
		"""
		feed = {}

		feed[self.placeholders['features']] 			= self.features
		feed[self.placeholders['adj_ind']]                      = self.adj_ind
		feed[self.placeholders['adj_ind_mask']]                 = self.adj_ind_mask
		feed[self.placeholders['num_features_nonzero']] 	= self.features[1].shape

		if split == 'train':
			feed[self.placeholders['labels']] 		= self.y_train
			feed[self.placeholders['labels_mask']] 		= self.train_mask
			feed[self.placeholders['dropout']] 		= self.p.drop
		elif split == 'test':
			feed[self.placeholders['labels']] 		= self.y_test
			feed[self.placeholders['labels_mask']] 		= self.test_mask
			feed[self.placeholders['dropout']] 		= 0.0
		else:
			feed[self.placeholders['labels']] 		= self.y_val
			feed[self.placeholders['labels_mask']] 		= self.val_mask
			feed[self.placeholders['dropout']] 		= 0.0

		return feed

	def sparse_dropout(self, x, keep_prob, noise_shape):
		"""
		Dropout for sparse tensors.

		Parameters
		----------
		x:		Input data
		keep_prob:	Keep probability
		noise_shape:	Size of each entry of x

		Returns
		-------
		pre_out:	x after dropout

		"""
		random_tensor  = keep_prob
		random_tensor += tf.random_uniform(noise_shape)
		dropout_mask   = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
		pre_out        = tf.sparse_retain(x, dropout_mask)
		return pre_out * (1./keep_prob)

	def matmul(self, a, b, is_sparse=False):
		"""
		Performs matrix multiplication between a and b, based on whether a is sparse or not.

		Parameters
		----------
		a, b:		Tensors to multiply
		is_sparse: 	Whether 'a' is sparse or not

		Returns
		-------
		Matrix multiplication output of 'a' and 'b'

		"""
		if is_sparse: 	return tf.sparse_tensor_dense_matmul(a, b)
		else: 		return tf.matmul(a, b)

	def dropout(self, inp, dropout, num_feat_nonzero=0, is_sparse=False):
		"""
		Performs dropout on given tensor inp based on whether inp is sparse or not

		Parameters
		----------
		inp:		Tensors on which dropout needs to be performed
		dropout:	Dropout rate
		num_feat_nonzero: Size of each entry of inp
		is_sparse: 	Whether 'inp' is sparse or not

		Returns
		-------
		inp after dropout

		"""
		if is_sparse: 	return self.sparse_dropout(inp, 1 - dropout, num_feat_nonzero)
		else:		return tf.nn.dropout(inp, 1-dropout)

	def GCNLayer(self, gcn_in, adj_ind, adj_ind_mask, input_dim, output_dim, act, dropout, num_features_nonzero, input_sparse=False, name='GCN'):
		"""
		GCN Layer Implementation for ConfGCN

		Parameters
		----------
		gcn_in:		Input to GCN Layer
		adj_ind:	Adjacency list 
		adj_ind_mask:	Mask corresponding to adj_ind
		input_dim:	gcn_dim dimension
		output_dim:	Final output dimension of GCN layer
		act:		Activation function to use
		dropout:	Dropout rate for GCN input
		num_feat_nonzero: Size of each entry of gcn_dim
		input_sparse:	Whether input is sparse or not
		name 		Name of the layer (used for creating variables, keep it different for different layers)

		Returns
		-------
		out		Output of GCN Layer
		"""

		with tf.variable_scope('{}_vars'.format(name)) as scope:
			wts  = tf.get_variable('weights', [input_dim, output_dim],       initializer=tf.contrib.layers.xavier_initializer(), regularizer=self.regularizer)
			bias = tf.get_variable('bias',    [output_dim],                  initializer=tf.contrib.layers.xavier_initializer(), regularizer=self.regularizer)

		gcn_in   = self.dropout(gcn_in, dropout, num_features_nonzero, is_sparse=input_sparse)
		node_act = self.matmul(gcn_in,  wts, is_sparse=input_sparse)

		f_vecs   = tf.nn.embedding_lookup(self.mu, adj_ind)
		f_diff   = f_vecs - tf.expand_dims(self.mu, axis=1)
		f_diff   = f_diff * tf.expand_dims(adj_ind_mask, axis=2)

		sig_vecs = tf.nn.embedding_lookup(self.sig, adj_ind)
		sig_sum  = sig_vecs + tf.expand_dims(self.sig, axis=1)
		sig_sum  = sig_sum  * tf.expand_dims(adj_ind_mask, axis=2)

		dist     = tf.reduce_sum(f_diff * (f_diff * sig_sum), axis=2) + self.p.bias
		dist     = 1 / dist
		dist     = tf.exp(dist - tf.reduce_max(dist, axis=1, keepdims=True)) * adj_ind_mask
		dist     = dist / tf.reduce_sum(dist, axis=1, keepdims=True)

		act_vecs = tf.nn.embedding_lookup(node_act, adj_ind)
		act_vecs = act_vecs * tf.expand_dims(adj_ind_mask, axis=2)

		final_act = tf.reduce_sum(act_vecs * tf.expand_dims(dist, axis=2), axis=1)
		gcn_out   = final_act

		return gcn_out

	def add_model(self):
		"""
		Creates the Computational Graph

		Parameters
		----------

		Returns
		-------
		nn_out:		Logits for each node in the graph
		"""


		self.layers, self.activations = [], []

		with tf.variable_scope('main_variables') as scope:
			self.mu  = tf.get_variable('mu',  [self.num_nodes,  self.output_dim], initializer=tf.contrib.layers.xavier_initializer(), regularizer=self.regularizer)	# Label distribution for each node
			self.sig = tf.get_variable('sig', [self.num_nodes,  self.output_dim], initializer=tf.constant_initializer(1.0), regularizer=self.regularizer)		# Inverse of co-variance matrix

		self.mu  = tf.nn.softmax(self.mu, axis = 1)	# Makes mu into a distribution
		self.sig = tf.nn.elu(self.sig)			# Imposes soft non-negative constraint on co-variance matrix

		gcn1_out = self.GCNLayer(
				gcn_in                  = self.placeholders['features'],
				adj_ind                 = self.placeholders['adj_ind'],
				adj_ind_mask            = self.placeholders['adj_ind_mask'],
				input_dim               = self.input_dim,
				output_dim              = self.p.gcn_dim,
				act                     = tf.nn.relu,
				dropout                 = self.placeholders['dropout'],
				num_features_nonzero    = self.placeholders['num_features_nonzero'],
				input_sparse            = True,
				name                    = 'GCN_1'
			)

		gcn2_out = self.GCNLayer(
				gcn_in                  = gcn1_out,
				adj_ind                 = self.placeholders['adj_ind'],
				adj_ind_mask            = self.placeholders['adj_ind_mask'],
				input_dim               = self.p.gcn_dim,
				output_dim              = self.output_dim,
				act                     = lambda x: x,
				dropout                 = self.placeholders['dropout'],
				num_features_nonzero    = self.placeholders['num_features_nonzero'],
				input_sparse            = False,
				name                    = 'GCN_2'
			)

		nn_out = gcn2_out
		return nn_out

	def get_accuracy(self, nn_out):
		"""
		Computed accuracy of the predicted nodes

		Parameters
		----------
		nn_out:		Output of the model

		Returns
		-------
		accuracy:	accuracy for the entire batch
		"""

		correct_prediction 	 = tf.equal(tf.argmax(nn_out, 1), tf.argmax(self.placeholders['labels'], 1))	# Identity position where prediction matches labels
		accuracy_all 		 = tf.cast(correct_prediction, tf.float32)					# Cast result to float
		mask 			 = tf.cast(self.placeholders['labels_mask'], dtype=tf.float32)			# Cast mask to float
		mask 			/= tf.reduce_mean(mask)								# Compute mean of mask
		accuracy_all 		*= mask 									# Apply mask on computed accuracy

		return tf.reduce_mean(accuracy_all)


	def loss_smooth(self, adj_ind, adj_ind_mask):
		"""
		Computes L_{smooth} term as defined in the paper

		Parameters
		----------
		adj_ind:	Adjacency list
		adj_ind_mask:	Mask corresponding to adjacency list

		Returns
		-------
		Returns  L_{smooth} loss
		"""
		mu_vecs  = tf.nn.embedding_lookup(self.mu, adj_ind)
		mu_diff  = (mu_vecs - tf.expand_dims(self.mu, axis=1)) * tf.expand_dims(adj_ind_mask, axis=2)

		sig_vecs = tf.nn.embedding_lookup(self.sig, adj_ind)
		sig_sum  = (sig_vecs + tf.expand_dims(self.sig, axis=1)) * tf.expand_dims(adj_ind_mask, axis=2)

		loss     = tf.reduce_sum(mu_diff * (mu_diff * sig_sum))

		return loss

	def loss_label(self):
		"""
		Computes L_{label} term as defined in the paper

		Parameters
		----------

		Returns
		-------
		Returns  L_{label} loss
		"""
		node_ind = tf.squeeze(tf.where(tf.not_equal(self.placeholders['labels_mask'], 0)), axis=1)

		mu_vecs  = tf.gather(self.mu, node_ind)
		y_actual = tf.gather(self.placeholders['labels'], node_ind)
		mu_diff  = y_actual - mu_vecs

		sig_vecs = tf.gather(self.sig, node_ind) + self.p.gamma
		loss     = tf.reduce_sum(mu_diff * ((mu_diff * sig_vecs)))

		return loss

	def loss_const(self, nn_out):
		"""
		Computes L_{const} term as defined in the paper

		Parameters
		----------
		nn_out:		Logits for each node in the graph

		Returns
		-------
		Returns  L_{const} loss
		"""
		pred  = tf.nn.softmax(nn_out)
		loss  = tf.square(pred - self.mu)
		loss  = loss * tf.expand_dims(tf.cast(self.placeholders['labels_mask'], tf.float32), axis=1)
		return tf.reduce_sum(loss)

	def loss_reg(self):
		"""
		Computes L_{regularizatino} term as defined in the paper

		Parameters
		----------

		Returns
		-------
		Returns  L_{regularizatino} loss
		"""
		return tf.reduce_sum(tf.where(self.sig < 0, -self.sig, tf.zeros_like(self.sig)))

	def add_loss_op(self, nn_out):
		"""
		Computes loss based on logits and actual labels

		Parameters
		----------
		nn_out:		Logits for each bag in the batch

		Returns
		-------
		loss:		Computes loss based on prediction and actual labels of the bags
		"""
		loss  = 0

		temp       = tf.nn.softmax_cross_entropy_with_logits(logits=nn_out, labels=self.placeholders['labels']) 	# Compute cross entropy loss
		mask       = tf.cast(self.placeholders['labels_mask'], dtype=tf.float32)					# Cast masking from boolean to float

		loss += self.p.l_cross 		* tf.reduce_sum(temp * mask) / tf.reduce_sum(mask)
		loss += 1/4 * self.p.l_smooth 	* self.loss_smooth(self.placeholders['adj_ind'], self.placeholders['adj_ind_mask'])
		loss += 1/2 * self.p.l_label 	* self.loss_label()
		loss += self.p.l_const 		* self.loss_const(nn_out)
		loss += self.p.l_reg 		* self.loss_reg()

		if self.regularizer != None:
			loss += tf.contrib.layers.apply_regularization(self.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

		return loss

	def add_optimizer(self, loss, isAdam=True):
		"""
		Add optimizer for training variables

		Parameters
		----------
		loss:		Computed loss

		Returns
		-------
		train_op:	Training optimizer
		"""
		with tf.name_scope('Optimizer'):
			if isAdam:  optimizer = tf.train.AdamOptimizer(self.p.lr)
			else:       optimizer = tf.train.GradientDescentOptimizer(self.p.lr)
			train_op  = optimizer.minimize(loss)

		return train_op

	def __init__(self, params):
		"""
		Constructor for the main function. Loads data and creates computation graph. 

		Parameters
		----------
		params:		Hyperparameters of the model

		Returns
		-------
		"""
		self.p  = params
		self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)
		self.logger.info(vars(self.p)); pprint(vars(self.p))

		if self.p.l2 == 0.0:    self.regularizer = None
		else:           	self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2)

		self.load_data()

		nn_out    	= self.add_model()
		self.loss 	= self.add_loss_op(nn_out)
		self.accuracy 	= self.get_accuracy(nn_out)

		self.train_op = self.add_optimizer(self.loss)

		self.merged_summ = tf.summary.merge_all()
		self.summ_writer = None


	def evaluate(self, sess, split='valid'):
		"""
		Evaluates the learned embeddings on valid/test data

		Parameters
		----------
		sess:		Session of tensorflow
		split:		Dataset split -- valid/test

		Returns
		-------
		loss:		Loss over the entire data
		acc:		Overall accuracy
		"""
		feed_dict 	= self.create_feed_dict(split=split)  				# Defines the feed_dict to be fed to NN
		loss, acc 	= sess.run([model.loss, model.accuracy], feed_dict=feed_dict) 	# Computer loss and accuracy
		return loss, acc								# return loss, accuracy

	def run_epoch(self, sess, epoch, shuffle=True):
		"""
		Runs one epoch of training

		Parameters
		----------
		sess:		Session of tensorflow
		epoch:		Epoch number
		shuffle:	Shuffle data while before creates batches

		Returns
		-------
		"""

		feed_dict = self.create_feed_dict(split='train')

		outs = sess.run([self.train_op, self.loss, self.accuracy], feed_dict=feed_dict)	# Training step
		cost, acc = self.evaluate(sess, split='valid')					# Computer Validation performance

		# Saving best model on Validation dataset
		if acc > self.best_val:
			self.best_val = acc
			self.saver.save(sess=sess, save_path=self.save_path)

		self.logger.info('E:{} {}  train_accuracy: {:.3f}\tvalid_accuracy: {:.3f}\tBest Validation Accuracy: {:.3f}'. format(epoch + 1, self.p.name, outs[2]*100, acc*100, self.best_val*100))

	def fit(self, sess):
		"""
		Trains the model and finally evaluates on test

		Parameters
		----------
		sess:		Tensorflow session object

		Returns
		-------
		"""
		self.summ_writer = tf.summary.FileWriter("tf_board/ConfGCN/" + self.p.name, sess.graph)
		self.saver     	 = tf.train.Saver()
		save_dir  	 = 'checkpoints/' + self.p.name + '/'
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		self.save_path 	 = os.path.join(save_dir, 'best_model')
		self.best_val    = 0.0

		for epoch in range(self.p.epochs):
			train_loss = self.run_epoch(sess, epoch)

		self.saver.restore(sess, self.save_path)
		test_cost, test_acc  = self.evaluate(sess, split='test')

		self.logger.info('\n\n===================\nFinal performance on {}: \nTest Accuracy: {:.2f} %\n==================='.format(self.p.data, test_acc*100))


if __name__== "__main__":

	parser = argparse.ArgumentParser(description='Confidence-based GCN')

	parser.add_argument('-data',    default='citeseer',				help='Dataset to use')
	parser.add_argument('-gpu',     default='0',					help='GPU to use')
	parser.add_argument('-name',    default='test',				help='Name of the run')

	parser.add_argument('-lr',      default=0.01,		type=float,		help='Learning rate')
	parser.add_argument('-epochs',  default=250,		type=int,		help='Max epochs')
	parser.add_argument('-l2',      default=0.01,		type=float,		help='L2 regularization')
	parser.add_argument('-opt',     default='adam',             			help='Optimizer to use for training')
	parser.add_argument('-gcn_dim', default=16,		type=int,       	help='GCN hidden dimension')
	parser.add_argument('-drop',    default=0.3,		type=float,     	help='Dropout for full connected layer')

	parser.add_argument('-l_cross', default=1, 		type=float,		help='L_cross value')
	parser.add_argument('-l_smooth',default=1, 		type=float,		help='L_smooth value')
	parser.add_argument('-l_label', default=0, 		type=float,		help='L_label value')
	parser.add_argument('-l_const', default=10, 		type=float,		help='L_const value')
	parser.add_argument('-l_reg', 	default=1, 		type=float,		help='L_reg value')
	parser.add_argument('-gamma', 	default=3, 		type=float,		help='Gamma value')
	parser.add_argument('-bias', 	default=0.1, 		type=float,		help='bias value')

	parser.add_argument('-restore',	 	 action='store_true',        		help='Restore from the previous best saved model')
	parser.add_argument('-eval',	  	 action='store_true',        		help='Set evaluation only mode')
	parser.add_argument('-manual_param',	 action='store_true',        		help='Set evaluation only mode')

	parser.add_argument('-logdir',  dest="log_dir",        	default='./log/',      	help='Log directory')
	parser.add_argument('-config',  dest="config_dir",     	default='./config/',    help='Config directory')

	args = parser.parse_args()

	# Not changing name when restoring previously saved model
	if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S") + '_' + str(uuid.uuid4())[:8]

	if not args.manual_param:
		params = json.load(open('./config/hyperparams.json'))
		for key, val in params[args.data].items():
			exec('args.{}={}'.format(key, val))

	# Evaluation only model (no training)
	if args.eval: args.epochs = 0

	# Set GPU
	set_gpu(args.gpu)

	# Create model
	model = ConfGCN(args)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		model.fit(sess)					# Start training
