def week_clustering(n_clusters=8, method='GMM', aggregation_period='D'):
	
	print 'Performing clustering of weekly behaviours...'
	
	'''Init stage'''
	if method not in ['GMM', 'Kmeans']:
		raise ValueError('Wrong method or no method specified.')
	if aggregation_period not in ['D', '8h', '6h']:
		raise ValueError('Wrong period or no period specified.')

	print 'Loading data...'
	EPS = sys.float_info.epsilon 
	with open(resources.clustering_matrix+aggregation_period+'.pickle', 'r') as f:
		clustering_matrix, code_index = pickle.load(f)
	code_index = np.asarray(code_index)
	# np.unique returns an ordered array and we don't want that
	_, idx = np.unique(code_index, return_index=True)
	unique_codes = code_index[np.sort(idx)]
	n_codes = len(unique_codes)
	dims = clustering_matrix.shape

	'''Normalize data'''
	mx = np.mean(clustering_matrix, axis=1)
	stdx = np.std(clustering_matrix, axis=1, dtype=np.float64)+EPS
	clustering_matrix = np.divide((clustering_matrix-np.array([mx,]*dims[1]).transpose()), 
                               np.array([stdx,]*dims[1]).transpose())

	'''Perform clustering'''
	if method=='Kmeans':
		# Kmeans clustering. MAKE SURE Kmeans MODULE IS IMPORTED.
		print 'Performing Kmeans clustering...'
		clst = Kmeans(n_clusters=n_clusters, precompute_distances=True, max_iter=500, n_init=20)
		clst.fit(clustering_matrix)
		labels = clst.predict(clustering_matrix)
	
	if method=='GMM':
		print 'Performing GMM clustering...'
		# GMM clustering. MAKE SURE GMM MODULE IS IMPORTED.
		clst = GMM(n_components=n_clusters, covariance_type='full', random_state=0)
		clst.fit(clustering_matrix)
		labels = clst.predict(clustering_matrix)
		prob, weights = clst.score_samples(clustering_matrix)

	weight_means = np.zeros((n_codes, K))
	for i in range(n_codes):
   		aux = weights_dframe.ix[unique_codes[i]]
   		weight_means[i, :] = np.mean(aux.values, axis=0)

	