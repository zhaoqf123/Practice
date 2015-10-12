## This is the main script to crack this challenge
import pandas as pd
import numpy as np
from datetime import datetime as dt
from sklearn import metrics

import DataProcessing as dp #This is customized module




class files(object):
	path_root = '/Volumes/Data/Downloads/Thunder/Uber'
	path_data = '/201408_trip_data.csv'
	path = path_root+path_data
	path_new = path[:-4]+'_new.csv'
	path_new_2 = path[:-4]+'_new_2.csv'
	path_out_clustering = path[:-4] + '_cluster_out.csv'
	path_out_classication = path[:-4] + '_classification_out.csv'


def preprocessing():
	'''This function applys classification analysis on the data.
	It will classify similar trips into chunks.
	The main aim is to group similar users into same chunks to provide some personalised service or recommendation
	'''
	tp = dp.datetimeparse # customised class to parse time
	path = files.path
	path_new = files.path_new
	path_new_2 = files.path_new_2
## Step 1. Read data into dataframe and explore the data
	df = pd.read_csv(path)
#!	print df.info()
#!	print df.describe()
#!	print df['Zip Code'].unique()
	header = df.columns.tolist()
	'''
## Step 2. Engineer new features, and select some old features, finally combine them for processing
	# all old features -> ['Trip ID', 'Duration', 'Start Date', 'Start Station', 'Start Terminal', 'End Date', 'End Station', 'End Terminal', 'Bike #', 'Subscriber Type', 'Zip Code']
	old_feature_selected = ['Trip ID', 'Duration', 'Start Terminal', 'End Terminal', 'Bike #', 'Subscriber Type']
	special_columns = dt(2014, 3, 1, 0, 0, 0, 0) # This column is to calculate the time difference between the column and this specific time
	new_df_1 = tp.split(df['Start Date'], time_str_format = "%m/%d/%Y %H:%M", columns = ['month', 'day', 'hour', 'minute', 'weekday'], special_columns = special_columns)
	new_df_2 = tp.split(df['End Date'], time_str_format = "%m/%d/%Y %H:%M", columns = ['month', 'day', 'hour', 'minute', 'weekday'], special_columns = special_columns)
	new_df = pd.concat([new_df_1, new_df_2], axis = 1)
	final_df = pd.concat([df[old_feature_selected], new_df], axis = 1)
	final_df.to_csv(path_new, index = False)
	'''
## Step 3. Apply one-hot encoder on certain features
	df = pd.read_csv(path_new)
	clns = ['Start Terminal', 'End Terminal', 'Bike #', 'Subscriber Type']
	df_new = pd.get_dummies(df, prefix = clns, prefix_sep = '_', dummy_na = False, columns = clns, sparse = False)
	df_new.to_csv(path_new_2, index = False)


###--------------------------------------------------------------------------------------------------------------------------------------------###
def predict_xgb(X, y):
	'''Use Xtreme Gradient Boosting to do train the model and do prediction
	'''
	import xgboost as xgb
## Step 1. Split training data into training and local validation data
	train_index, test_index, pred_index = dp.stratification.one_cln3(y, train = 0.6, cross = 0.2)

	train_train = X[train_index, :]
	train_test = X[test_index, :]
	train_pred = X[pred_index, :]
	y_train = y[train_index]
	y_test = y[test_index]
	y_pred_truth = y[pred_index]

	dtrain = xgb.DMatrix(train_train, label = y_train) # Create xgb data object for training
	dtest = xgb.DMatrix(train_test, label = y_test) # Create xgb data object for local testing
## Step 2. Set parameter and train the model
	param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}#, 'eval_metric':'auc'}
	num_round = 3 # The maximum number of iterations
	watchlist = [(dtest, 'eval'), (dtrain, 'train')]
	bst = xgb.train(param, dtrain, num_round, watchlist)
## Step 3. Do prediction
	X_pred = xgb.DMatrix(train_pred)
	preds = bst.predict(X_pred)
## Step 4. Evaluate the prediction by comparing it with the results
#	print 'error = %f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=y_pred_truth[i]) /float(len(preds)))
	y_pred_pred = (preds>0.5).astype(int) # Convert probability into binary 0 or 1
	recall = metrics.recall_score(y_pred_truth, y_pred_pred)
	precision = metrics.precision_score(y_pred_truth, y_pred_pred)
	f1 = metrics.f1_score(y_pred_truth, y_pred_pred)
	auc = metrics.roc_auc_score(y_pred_truth, y_pred_pred)
	print "Recall is %f, Precision is %f, f1 score is %f, and AUC is %f" % (recall, precision, f1, auc)
## Step 5. Dump model
	# dump model with feature map
	bst.dump_model(files.path_root + '/dump' + '/dump.nice.txt', files.path_root + '/dump' + '/featmap.txt')
	return preds

def classification():
	'''This function applys classification analysis on the data.
	It will predict whether a given trip is done by a customer or a subscriber.
	The main aim is to identify potential subscriber, identify the most important features that
	determine whether a person can be a subscriber or not, and use the reason to turn them into subscriber
	'''
## Step 1. Read the preprocessed data, and then drop some columns and modify certain columns
	path_new_2 = files.path_new_2
	path_new = files.path_new
	path_out_classication = files.path_out_classication
	clns_to_drop = ['Trip ID', 'Subscriber Type_Customer', 'Subscriber Type_Subscriber'] #['Trip ID', 'Start Date_month', 'Start Date_total_seconds', 'End Date_month', 'End Date_total_seconds']
	df = pd.read_csv(path_new_2)
	df_X = df.drop(clns_to_drop, axis = 1)
	df_y = df['Subscriber Type_Subscriber']

	df_X['Duration'] = df_X['Duration'].apply(lambda x: x/60.0) # Transform the duration into unit of min
	df_X['Start Date_total_seconds'] = df_X['Start Date_total_seconds'].apply(lambda x: x/86400.0) # Transform the total time into unit of day
	df_X['End Date_total_seconds'] = df_X['End Date_total_seconds'].apply(lambda x: x/86400.0) # Transform the total time into unit of day

	header = df_X.columns.tolist()
	df_header = pd.DataFrame(header, columns = ['cln_names'])
	df_header.to_csv(files.path_root + '/headers_dump.csv')
## Step 2. Construct sklearn or xgboost constructor, and do some prediction
	X = df_X.values
	y = df_y.values
	predict_xgb(X, y)
###--------------------------------------------------------------------------------------------------------------------------------------------###


	
##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@##
def survey_n_clusters(data, n_kernels, path_out_clustering, df_original):
	from sklearn.cluster import KMeans
#	from sklearn.metrics import silhouette_score
	path_out_clustering = path_out_clustering[:-4] + '_' + str(n_kernels) + 'clusters.csv'
	model = KMeans(n_clusters=n_kernels, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
	label = model.fit_predict(data)

#	silhouette_avg = silhouette_score(data, label)
#	print("For n_clusters =", n_kernels, "The average silhouette_score is : ", silhouette_avg)
#	The above evaluation throw feedback "Killed: 9"
## Step 2.1 Output the labeled data, combining with the original data that is before one-hot-encoder
	print ("For n_clusters =", n_kernels, "The summation of all labels is : ", sum(label))
	print ("For n_clusters =", n_kernels, "The total number of non-zero labels is : ", sum(label > 0))
	print ("For n_clusters =", n_kernels, "The built-in score is : ", model.score(data))

	label = label.reshape(len(label),1)
	df_out = pd.DataFrame(label, columns = ['label'])
	df_out = pd.concat([df_original, df_out], axis = 1)
	df_out.to_csv(path_out_clustering, index = False)

def clustering():
	'''This function applys unsupervised learning - clustering, to analyse the data in order to find churn.
	The churn can be used to provide some personalised service or recommendation.
	'''
## Step 1. Read the preprocessed data, then drop some columns	
	path_new_2 = files.path_new_2
	path_new = files.path_new
	path_out_clustering = files.path_out_clustering
	clns_to_drop = ['Trip ID', 'Start Date_month', 'Start Date_total_seconds', 'End Date_month', 'End Date_total_seconds']
	df = pd.read_csv(path_new_2)
	df = df.drop(clns_to_drop, axis = 1)
	df['Duration'] = df['Duration'].apply(lambda x: x/3600.0) # Transform the duration into hours unit -> a kind of similarity
	header = df.columns.tolist()
## Step 2. Construct sklearn constructor, and do some prediction
	data = df.values

	df_original = pd.read_csv(path_new) # This is used to attach the original data with the predicted clusters
	range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
	for i in range_n_clusters:
		survey_n_clusters(data, i, path_out_clustering, df_original)
##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@##


if __name__ == '__main__':
## Step 0. Do some preprocessing on the data
#	preprocessing()
## Step 1. Do some classification/clustering analysis on the data
#	clustering()
	classification()





