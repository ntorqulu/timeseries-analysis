import sys
import numpy as np
import network as network_module
import data as data_module
import os

#Retrieving arguments from the command line
test_name = str(sys.argv[1])
filter_name = str(sys.argv[2])
classifier = str(sys.argv[3])
num_nodes = int(sys.argv[4])
input_probability = np.float64(sys.argv[5])
reservoir_probability = np.float64(sys.argv[6])

data_obj = data_module.Data(80) #80% training 20% testing

net = network_module.Network()

#Setting the right data for all the possible combinations of problems and classifiers

script_dir = os.path.dirname(__file__)

if test_name == '5s':
	data_file = os.path.join(script_dir, 'dataSorted_allOrientations.mat')
	data_obj.import_data(data_file)
	if classifier == 'lin':
		data_obj.build_train_labels_lin()
		data_obj.build_test_labels_lin()
        
	elif classifier == 'log':
		data_obj.build_train_labels_log()
		data_obj.build_test_labels_log()

	else:
		print("This classifier is not supported for this test.")
		sys.exit(1)

	data_obj.build_training_matrix()
	data_obj.build_test_matrix()
	net.L = 5

elif test_name == 'lvr':
	if classifier == 'log' or classifier == '1nn':
		data_file = os.path.join(script_dir, 'dataSorted_leftAndRight.mat')
		data_obj.import_data(data_file)
		data_obj.leftvsright_mixed()
		net.L = 1

	else: 
		print("This classifier is not supported for this test.")
		sys.exit(1)

else:
	print("This test does not exist.")
	sys.exit(1)

#Filtering the data
if filter_name not in data_obj.spectral_bands.keys():
	print("The specified frequency band is not supported")
	sys.exit(1)

data_obj.training_data = data_obj.filter_data(data_obj.training_data,filter_name)
data_obj.test_data = data_obj.filter_data(data_obj.test_data,filter_name)


#Computing the absolute value of the data, to get rid of negative numbers
data_obj.training_data = np.abs(data_obj.training_data)
data_obj.test_data = np.abs(data_obj.test_data)

########################
# Define the network parameters
########################

net.T = data_obj.training_data.shape[1] #Number of training time steps
net.n_min = 2540 #Number time steps dismissed
net.K = 128 #Input layer size
net.N = num_nodes #Reservoir layer size


net.u = data_obj.training_data
net.y_teach = data_obj.training_results

net.setup_network(data_obj,num_nodes,input_probability,reservoir_probability,data_obj.data.shape[-1])

net.train_network(data_obj.data.shape[-1],classifier,data_obj.num_columns, data_obj.num_trials_train, data_obj.train_labels, net.N) 

net.mean_test_matrix = np.zeros([net.N,data_obj.num_trials_test,data_obj.data.shape[-1]])

net.test_network(data_obj.test_data, data_obj.num_columns,data_obj.num_trials_test, net.N, data_obj.data.shape[-1], t_autonom=data_obj.test_data.shape[1])

if classifier == 'lin':
	print(f'Performance for {test_name} using {classifier} : {data_obj.accuracy_lin(net.regressor.predict(net.mean_test_matrix.T),data_obj.test_labels)}')

elif classifier == 'log':
	print(f'Performance for {test_name} using {classifier} : {net.regressor.score(net.mean_test_matrix.T,data_obj.test_labels.T)}')

elif classifier == '1nn':
	print(f'Performance for {test_name} using {classifier} : {net.regressor.score(net.mean_test_matrix.T,data_obj.test_labels)}')
