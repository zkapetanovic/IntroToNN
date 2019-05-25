import pickle
import numpy as np


def unpickle(filename):
    fo = open(filename, 'rb')
    data = pickle.load(fo, encoding='bytes')
    fo.close()
    
    return data

#Step 1: define a function to load traing batch data from directory
def load_training_batch(folder_path,batch_id):

    ###load batch using pickle###
    filename = folder_path + 'data_batch_' + str(batch_id)
    print("Loading File: " + filename)
    data_dict = unpickle(filename)

  
    return data_dict


#Step 2: define a function to load testing data from directory
def load_testing_batch(folder_path):
    
	###load batch using pickle###
    filename = folder_path + 'test_batch'
    print("Loading File: " + filename)
    data_dict = unpickle(filename)

    
    return data_dict

#Step 3: define a function that returns a list that contains label names (order is matter)
"""
	airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""
def load_label_names():
    meta_data = unpickle('cifar-10-batches-py/batches.meta')
    label_names = np.array(meta_data[b'label_names'])
    
    return label_names

#Step 4: define a function that reshapes the features to have shape (10000, 32, 32, 3)
def features_reshape(features):
    feat_reshape = features.reshape((len(features), 3,32, 32)).transpose(0,2,3,1)
    
    return feat_reshape

'''
#Step 5 (Optional): A function to display the stats of specific batch data.
def display_data_stat(folder_path,batch_id,data_id):
	"""
	Args:
		folder_path: directory that contains data files
		batch_id: the specific number of batch you want to explore.
		data_id: the specific number of data example you want to visualize
	Return:
		None

	Descrption: 
		1)You can print out the number of images for every class. 
		2)Visualize the image
		3)Print out the minimum and maximum values of pixel 
	"""
	pass
'''

#Step 6: define a function that does min-max normalization on input
def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    norm = (x - min_val)/(max_val-min_val)
    
    return norm

#Step 7: define a function that does one hot encoding on input
def one_hot_encoding(x):
    encoded = np.zeros((len(x), 10))
    
    for j,k in enumerate(x):
        
        encoded[j][k] = 1
    
    return encoded

#Step 8: define a function that perform normalization, one-hot encoding and save data using pickle
def preprocess_and_save(features,labels,filename):
    
    features = normalize(features)
    labels = one_hot_encoding(labels)
    
    pickle.dump((features, labels), open(filename, 'wb'))

    return "Preprocess and save complete."


#Step 9:define a function that preprocesss all training batch data and test data. 
#Use 10% of your total training data as your validation set
#In the end you should have 5 preprocessed training data, 1 preprocessed validation data and 1 preprocessed test data
def preprocess_data(folder_path):
    num_batches = 5
    valid_features = []
    valid_labels = []
    
    ##### Preprocess training data #####
    for batch in range(1, num_batches+1):
        batch_i = load_training_batch(folder_path, batch)
        
        features = batch_i[b'data']
        features = features_reshape(features)
        labels = batch_i[b'labels']
        
        index = int(len(features) * 0.1)
        
        # Preprocess 90% of training data set of batch_i
        filename = 'preprocessed_batch_' + str(batch) + '.p'
        preprocess_and_save(features[:-index], labels[:-index], filename)
        
        # Preprocess 10% for validation
        valid_features.extend(features[-index:])
        valid_labels.extend(labels[-index:])
        
    ##### Preprocess and save validation data #####
    preprocess_and_save(np.array(valid_features), np.array(valid_labels), 'preprocessed_validation.p')
    
    ##### Preprocess and save test data #####
    fo = open(folder_path + '/test_batch', mode='rb')
    test_batch = pickle.load(fo, encoding='bytes')
    fo.close()
    
    test_features = features_reshape(test_batch[b'data'])
    
    
    test_labels = test_batch[b'labels']
    
    preprocess_and_save(np.array(test_features), np.array(test_labels), 'preprocessed_training.p')
    
    
   
#Step 10: define a function to yield mini_batch
def mini_batch(features,labels,mini_batch_size):
    
    for start in range(0, len(features), mini_batch_size):
        stop = min(start + mini_batch_size, len(features))
        yield features[start:stop], labels[start:stop]
    
    
#Step 11: define a function to load preprocessed training batch, the function will call the mini_batch() function
def load_preprocessed_training_batch(batch_id, batch_size):
    
    filename = 'preprocessed_batch_' + str(batch_id) + '.p'
    features,labels = pickle.load(open(filename, mode='rb'))
    
    return mini_batch(features,labels, batch_size)

#Step 12: load preprocessed validation batch
def load_preprocessed_validation_batch(batch_id, batch_size):
    
	filename = 'preprocessed_validation.p'
	features,labels = pickle.load(open(filename, mode='rb'))
    
	return features,labels

#Step 13: load preprocessed test batch
def load_preprocessed_test_batch(test_mini_batch_size):
    
    filename = 'preprocessed_training.p'
    features,label = pickle.load(open(filename, mode='rb'))
    
    return mini_batch(features,labels,test_mini_batch_size)
