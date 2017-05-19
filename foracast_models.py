import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
#######NEURAL NETWORK########
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
#from pybrain.tools.xml import NetworkWriter
#from pybrain.tools.xml import NetworkReader
#from pybrain.tools.neuralnets import NNregression
############################
#matplotlib.use('TkAgg')
from sklearn.svm import SVC
from sklearn import svm
from sklearn import decomposition
from sklearn import datasets

class basic_forecast(object):

	def __init__(self):
		pass
	def set_parameters(self,forecast_parameters,n_past_years,r_stat,cotg):
		'S,starting_year, delta_validation, delta_test, n_past_years'
		self.r_stat = r_stat
		self.forecast_parameters = forecast_parameters
		self.n_past_years = n_past_years
		self.cotg = cotg
	def  forecast(self, cotg):
		''' Method to calculate the '''
		def DRL_n_years(self):
			j=0
			year2=[]
			predict2=[]
			NR102 = []
			for j in range(len(self.r_stat)-self.n_past_years):
			    NR102.append(self.cotg.Cota[(self.cotg.Ano2==self.r_stat.index.min()+j+self.n_past_years)].describe(percentiles=linspace(0,1,21))['10%'])
			    predict2.append(self.cotg.Cota[(self.cotg.Ano2>=self.r_stat.index.min()+j)&(self.cotg.Ano2<self.r_stat.index.min()+j+self.n_past_years)].describe(percentiles=linspace(0,1,21))['10%'])
			    year2.append(self.r_stat.index.min()+j+self.n_past_years)
			erro = abs(np.array(NR102 )- np.array(predict2))
			erro20 = pd.DataFrame(erro)
			erro20.index = year2
			self.predict2 = predict2
			self.NR102 = NR102
			BM20 = pd.DataFrame([year2, NR102,predict2,abs(np.array(NR102)-np.array(predict2)).tolist()],index=['year','NR10','predict','erro']).T
			BM20.index=BM20.year
			self.bm = BM20
		DRL_n_years(self)
		self.test_results = self.forecast_parameters
		test_results = []
		BM_Error = []
		BM_Prediction = []
		BM_nr = []
		for i in range(len(self.forecast_parameters)):
			aux = self.bm[self.forecast_parameters[1][i]+self.forecast_parameters[2][i]+10:10+self.forecast_parameters[1][i]+self.forecast_parameters[2][i]+self.forecast_parameters[3][i]-1]
			test_results.append(aux)
			BM_Error.append(aux.erro)
			BM_Prediction.append(aux.predict)
			BM_nr.append(aux.NR10)
		self.test_results['BM_Error'] = pd.Series(BM_Error)
		self.test_results['BM_Prediction'] = pd.Series(BM_Prediction)
		self.test_results['BM_nr'] = pd.Series(BM_nr)
		self.results_dataframe = pd.concat(test_results)
		
class auto_regression(object):
	
	def __init__(self):
		pass
	def set_parameters(self,forecast_parameters,n_past_years,r_stat,cotg):
		'S,starting_year, delta_validation, delta_test, n_past_years'
		self.r_stat = r_stat
		self.forecast_parameters = forecast_parameters
		self.n_past_years = n_past_years
		self.cotg = cotg
	def forecast(self):
		# Define the training variables
		def set_training_values(self,i):	
			self.training_values_x = self.r_stat['10%'].ix[self.r_stat.index.min()+1: self.forecast_parameters.ix[i:i,3:4].values[0][0]+self.forecast_parameters.ix[i:i,1:2].values[0][0]-2]
			self.training_values_y = self.r_stat['10%'].ix[self.r_stat.index.min()+2:self.forecast_parameters.ix[i:i,3:4].values[0][0]+self.forecast_parameters.ix[i:i,1:2].values[0][0]-1]

		# Defines function to find the coefficients that best fit the data
		
		def train_coefficients(self):

			self.input_matrix = np.vstack((np.ones(len(self.training_values_x)),self.training_values_x.as_matrix().T)).T
			kron = (np.kron(np.eye(len(self.training_values_y)),np.ones((1,1))))
			kron[0][0] = 0
			kron_lambda =0
			self.coefficients = (np.mat(self.input_matrix.T)*np.linalg.pinv(np.mat(self.input_matrix)*np.mat(self.input_matrix.T)+kron_lambda*kron))*np.mat(self.training_values_y.values).T
			self.training_results = pd.DataFrame(np.zeros(len(self.training_values_y)))*np.NAN
			self.training_results['Years'] = self.training_values_y.index
			self.training_results['DRL'] = self.training_values_y.tolist()
			self.training_results['Prediction'] = np.squeeze(np.mat(self.input_matrix) * self.coefficients).tolist()[0]
			self.training_results['Error'] = np.sqrt((self.training_results['DRL'] - self.training_results['Prediction'])**2)
		
		def test_coefficients(self,i):
			# Need to be updated to the new format without validation period.
			# Slice [Ano_Inicio+10:Ano_Inicio+10+Delta]  Delta = 10; To be corrected need to input the right forecast_parameter and correct training values slicing as well.
			self.test_values_x = self.r_stat['10%'].ix[self.forecast_parameters.ix[i:i,1:2].values[0][0]+self.forecast_parameters.ix[i:i,3:4].values[0][0]-1:
                                                 self.forecast_parameters.ix[i:i,1:2].values[0][0]+self.forecast_parameters.ix[i:i,3:4].values[0][0]+self.forecast_parameters.ix[i:i,3:4].values[0][0]-2]
			self.test_values_y = self.r_stat['10%'].ix[self.forecast_parameters.ix[i:i,1:2].values[0][0]+self.forecast_parameters.ix[i:i,3:4].values[0][0]:
                                                 self.forecast_parameters.ix[i:i,1:2].values[0][0]+self.forecast_parameters.ix[i:i,3:4].values[0][0]+self.forecast_parameters.ix[i:i,3:4].values[0][0]-1]
		 	self.input_matrix_test = np.vstack((np.ones(len(self.test_values_x)),self.test_values_x.as_matrix().T)).T
		 	self.test_results = pd.DataFrame(np.zeros(len(self.test_values_y)))*np.NAN
		 	prediction_aux = np.mat(self.input_matrix_test)*self.coefficients
		 	self.test_results['Years'] = self.test_values_y.index
		 	self.test_results['DRL'] = self.test_values_y.tolist()
		 	self.test_results['Prediction'] = np.array(prediction_aux.T.tolist()[0])
		 	self.test_results['Error'] = np.sqrt((self.test_results['DRL'] - self.test_results['Prediction'])**2)

		def plot_results(self):
			fig = plt.figure(figsize=(4,2))
			self.test_results['DRL'].plot(marker='*',markersize=10)
			self.test_results['Prediction'].plot(marker='*',markersize=10)
			#self.test_results['Error'].plot(kind='bar')
			#plt.yticks(range(0,100,10))
			plt.legend(loc=1)
			plt.title(str(self.forecast_parameters.ix[i:i,1:2].values[0][0]+self.forecast_parameters.ix[i:i,3:4].values[0][0])+
                      ':'+str(self.forecast_parameters.ix[i:i,1:2].values[0][0]+self.forecast_parameters.ix[i:i,3:4].values[0][0]+
                      self.forecast_parameters.ix[i:i,3:4].values[0][0]-1) + '---E:'+str(round(self.test_results['Error'].mean())))

			plt.show()


		self.coefficients_summary = []
		self.training_results_summary = []
		self.test_results_summary = []
		self.error_summary = []
		self.predictions_summary = []
		self.tests_result = self.forecast_parameters

		for i in range(len(self.forecast_parameters)):
			
			set_training_values(self,i)
			train_coefficients(self)
			test_coefficients(self,i)
			plot_results(self)
			
			self.coefficients_summary.append(self.coefficients)
			self.training_results_summary.append(self.training_results)
			self.test_results_summary.append(self.test_results)
			self.error_summary.append(self.test_results.Error)
			self.predictions_summary.append(self.test_results.Prediction)

		self.forecast_results = pd.concat(self.test_results_summary)
		self.tests_result = self.forecast_parameters
		self.tests_result['AR_Error'] = self.error_summary
		self.tests_result['AR_Prediction'] = self.predictions_summary


class multiple_regression():

	def __init__(self):
		pass
	
	def set_month_values(self,aggfunc):

		self.monthly_values = self.cotg[self.cotg.Cota.notnull()].pivot_table(values = 'Cota', index=['Ano2'], columns=['Mes'],aggfunc=aggfunc)
		self.monthly_values = self.monthly_values[[7, 8, 9, 10, 11,12,1, 2, 3, 4, 5, 6]]
		self.monthly_values.columns = range(12)

	def set_parameters(self,forecast_parameters,n_past_years,r_stat,cotg,number_of_months):
		'S,starting_year, delta_validation, delta_test, n_past_years'
		self.r_stat = r_stat
		self.forecast_parameters = forecast_parameters
		self.n_past_years = n_past_years
		self.cotg = cotg
		self.number_of_months = number_of_months
		self.set_month_values(np.max)
		

	def forecast(self):

		def set_training_values(self,i):
			self.training_values_x = self.monthly_values.ix[self.monthly_values.index.min()+1:
                                     self.forecast_parameters.ix[i:i,1:2].values[0][0]+self.forecast_parameters.ix[i:i,3:4].values[0][0]-1,
			                         11-self.number_of_months:]
			self.training_values_y = self.r_stat['10%'].ix[self.monthly_values.index.min()+2:
                                     self.forecast_parameters.ix[i:i,1:2].values[0][0]+self.forecast_parameters.ix[i:i,3:4].values[0][0]]

		def train_coefficients(self):
			
			self.input_matrix = np.vstack((np.ones(len(self.training_values_x)),self.training_values_x.as_matrix().T)).T
			kron = (np.kron(np.eye(len(self.training_values_y )),np.ones((1,1))))
			kron[0][0] = 0
			kron_lambda = 0
			self.coefficients = (np.mat(self.input_matrix.T)*np.linalg.pinv(np.mat(self.input_matrix)*np.mat(self.input_matrix.T)+kron_lambda*kron))*np.mat(self.training_values_y.values).T
			self.training_results = pd.DataFrame(np.zeros(len(self.training_values_y )))*np.NAN
			self.training_results['Years'] = self.training_values_y.index
			self.training_results['DRL'] = self.training_values_y.tolist()
			self.training_results['Prediction'] = np.squeeze(np.mat(self.input_matrix) * self.coefficients).tolist()[0]
			self.training_results['Error'] = np.sqrt((self.training_results['DRL'] - self.training_results['Prediction'])**2)

		def set_test_values(self,i):
			
			self.test_values_x = self.monthly_values.ix[self.forecast_parameters.ix[i:i,1:2].values[0][0]+self.forecast_parameters.ix[i:i,3:4].values[0][0]-1:
                                                           self.forecast_parameters.ix[i:i,1:2].values[0][0]+self.forecast_parameters.ix[i:i,3:4].values[0][0]+
                                                           self.forecast_parameters.ix[i:i,3:4].values[0][0]-2,
                                                           11-self.number_of_months:]

			self.test_values_y = self.r_stat['10%'].ix[self.forecast_parameters.ix[i:i,1:2].values[0][0]+self.forecast_parameters.ix[i:i,3:4].values[0][0]:
                                                      self.forecast_parameters.ix[i:i,1:2].values[0][0]+self.forecast_parameters.ix[i:i,3:4].values[0][0]+
                                                      self.forecast_parameters.ix[i:i,3:4].values[0][0]-1]

		def test_coefficients(self,i):
		 	self.input_matrix_test = np.vstack((np.ones(len(self.test_values_x)),self.test_values_x.as_matrix().T)).T
		 	self.test_results = pd.DataFrame(np.zeros(len(self.test_values_y)))*np.NAN
		 	prediction_aux = np.mat(self.input_matrix_test)*self.coefficients
		 	self.test_results['Years'] = self.test_values_y.index.tolist()
		 	self.test_results['DRL'] = self.test_values_y.tolist()
		 	self.test_results['Prediction'] = np.array(prediction_aux.T.tolist()[0])
		 	self.test_results['Error'] = np.sqrt((self.test_results['DRL'] - self.test_results['Prediction'])**2)

		def plot_results(self):
			fig = plt.figure(figsize=(4,2))
			self.test_results['DRL'].plot(marker='*',markersize=10)
			self.test_results['Prediction'].plot(marker='*',markersize=10)
			#self.test_results['Error'].plot(kind='bar')
			#plt.yticks(range(0,100,10))
			plt.legend(loc=1)
			plt.title(str(self.forecast_parameters.ix[i:i,1:2].values[0][0]+self.forecast_parameters.ix[i:i,3:4].values[0][0])+
                      ':'+str(self.forecast_parameters.ix[i:i,1:2].values[0][0]+self.forecast_parameters.ix[i:i,3:4].values[0][0]+
                      self.forecast_parameters.ix[i:i,3:4].values[0][0]-1) + '---E:'+str(round(self.test_results['Error'].mean())))
			plt.show()


		self.coefficients_summary = []
		self.training_results_summary = []
		self.test_results_summary = []
		self.error_summary = []
		self.predictions_summary = []
		self.tests_result = self.forecast_parameters
		

		for i in range(len(self.forecast_parameters)):
			
			set_training_values(self,i)
			train_coefficients(self)
			set_test_values(self,i)
			test_coefficients(self,i)
			plot_results(self)
			
			self.coefficients_summary.append(self.coefficients)
			self.training_results_summary.append(self.training_results)
			self.test_results_summary.append(self.test_results)
			self.error_summary.append(self.test_results.Error)
			self.predictions_summary.append(self.test_results.Prediction)

		self.forecast_results = pd.concat(self.test_results_summary)
		self.tests_result = self.forecast_parameters
		self.tests_result['MR_Error'] = self.error_summary
		self.tests_result['MR_Prediction'] = self.predictions_summary


class artificial_neural_network(object):


	def __init__(self):
		pass

	def set_month_values(self,aggfunc):

		self.monthly_values = self.cotg[self.cotg.Cota.notnull()].pivot_table(values = 'Cota', index=['Ano2'], columns=['Mes'],aggfunc=aggfunc)
		self.monthly_values = self.monthly_values[[7, 8, 9, 10, 11,12,1, 2, 3, 4, 5, 6]]
		self.monthly_values.columns = range(12)


	def set_parameters(self,forecast_parameters,r_stat,cotg,number_of_months):
		'S,starting_year, delta_validation, delta_test, n_past_years'
		self.r_stat = r_stat
		self.forecast_parameters = forecast_parameters
		self.cotg = cotg
		self.number_of_months = number_of_months
		self.neurons_structure = [3,8,8,8]
		self.set_month_values(np.max)

	def normalize(self,series):

		if isinstance(series,pd.DataFrame):
			series_max = series.T.stack().max()
			series_min = series.T.stack().min()
			series_mean = series.T.stack().mean()
			series_std = series.T.stack().std()
		else:
			series_max = series.T.max()
			series_min = series.T.min()
			series_mean = series.T.mean()
			series_std = series.T.std()

		return (series-series_min)/(series_max-series_min)

	def set_training_values(self,training_values_x,training_values_y):

		self.training_values_x = training_values_x
		self.training_values_y = training_values_y
		self.input_matrix_x = self.normalize(self.training_values_x).as_matrix()
		self.input_matrix_y = self.normalize(self.training_values_y).as_matrix()
		self.input_matrix_y = self.input_matrix_y.reshape(-1,1)

		self.ds = SupervisedDataSet(self.input_matrix_x.shape[1],1)
		for x, y in zip(self.input_matrix_x, self.input_matrix_y):
			self.ds.addSample(tuple(x), (y))

	def set_training_validation_ratio(self):
		# PCA - DIMENTION REDUCTION

		pca = decomposition.PCA(n_components=2)
		pca.fit(self.input_matrix_x)

		X = pca.transform(self.input_matrix_x)
		x_test = pca.transform(self.test_values_x_norm.values)

		def dist(X,x_test):
			euc_dist = map(lambda i: np.linalg.norm(np.array([X[i]])-np.array([x_test])),range(len(X)))
			
			return(euc_dist)

		distances = dist(X,x_test)
		ordered_distances = pd.DataFrame(distances).sort(0)
		self.distances = dist(X,x_test)
		self.ordered_distances = pd.DataFrame(distances).sort(0)

		# SLICE VALIDATION AND TRAINING PERIODS

		self.training_set_input = self.input_matrix_x[ordered_distances.index[5:]]
		self.training_set_target = self.input_matrix_y[ordered_distances.index[5:]]
		self.validation_set_input = self.input_matrix_x[ordered_distances.index[:5]]
		self.validation_set_target = self.input_matrix_y[ordered_distances.index[:5]]

		#CREATE DATASETS FOR PYBRAIN MODEL
		self.training_set = SupervisedDataSet(self.training_set_input.shape[1],self.training_set_target.shape[1])
		for x, y in zip(self.training_set_input,self.training_set_target):
			self.training_set.addSample(tuple(x), (y))

		self.validation_set = SupervisedDataSet(self.validation_set_input.shape[1],self.validation_set_target.shape[1])
		for x, y in zip(self.validation_set_input,self.validation_set_target):
			self.validation_set.addSample(tuple(x), (y))

		#print(self.validation_set)

		plt.figure(figsize=(2,2))
		plt.scatter(X[ordered_distances.index[5:]][:, 0], X[ordered_distances.index[5:]][:, 1],c='black')
		#plt.scatter(X[ordered_distances.index[len(X)/10:]], self.training_set_target,c='black',s=20)
		plt.scatter(x_test[:, 0], x_test[:, 1],marker='v',c='red',edgecolor=None,s=100)
		#plt.scatter(x_test,self.test_values_y_norm ,marker='v',c='red',edgecolor=None,s=100)
		#plt.scatter(X[:, 0][ordered_distances.index[:len(X)/10]], X[:, 1][ordered_distances.index[:len(X)/10]],c='yellow')
		plt.scatter(X[ordered_distances.index[:5]][:, 0], X[ordered_distances.index[:5]][:, 1],c='y' )#c=self.validation_set_target.reshape(-1), cmap=plt.cm.viridis)
		#plt.scatter(X[ordered_distances.index[:len(X)/10]][,:0], self.validation_set_target,c='yellow')
		#print((X[ordered_distances.index[:len(X)/10]]).reshape(-1,1))
		#print((self.test_values_y[ordered_distances.index[:len(X)/10]]).reshape(-1,1))
		plt.show()

		# REMOVE THE WEIRD EXTRA ZEROS
		#self.validation_set.data['input'] = np.array([self.validation_set.data['input'][0]])
		#self.validation_set.data['target'] = np.array([self.validation_set.data['target'][0]])

	def set_validation_values(self,validation_values_x,validation_values_y):

		self.validation_values_x = validation_values_x
		self.validation_values_y = validation_values_y

		self.validation_values_x_norm = (self.validation_values_x - self.training_values_x.min())/(self.training_values_x.max() - self.training_values_x.min())
		self.validation_values_y_norm = (self.validation_values_y - self.training_values_y.min())/(self.training_values_y.max() - self.training_values_y.min())

		self.val_data = SupervisedDataSet(self.validation_values_x_norm.as_matrix().shape[1],1)
		for x, y in zip(self.validation_values_x_norm.as_matrix(), self.validation_values_y_norm.as_matrix().reshape(-1,1)):
			self.val_data.addSample(tuple(x), (y))
		#self.val_data.data['input'] = self.val_data.data['input'][0]
		#self.val_data.data['target'] = self.val_data.data['target'][0]

	def set_test_values(self,training_values_x,training_values_y):

		self.test_values_x = training_values_x
		self.test_values_y = training_values_y

		self.test_values_x_norm = (self.test_values_x - self.training_values_x.values.min())/(self.training_values_x.values.max() - self.training_values_x.values.min())
		self.test_values_y_norm = (self.test_values_y - self.training_values_y.values.min())/(self.training_values_y.values.max() - self.training_values_y.values.min())

		self.test_data = SupervisedDataSet(self.test_values_x_norm.as_matrix().shape[1],1)
		for x, y in zip(self.test_values_x_norm.as_matrix(), self.test_values_y_norm.as_matrix().reshape(-1,1)):
			self.test_data.addSample(tuple(x), (y))
		#self.test_data.data['input'] = self.test_data.data['input'][0]
		#self.test_data.data['target'] = self.test_data.data['target'][0]

	def build_network(self,i):

		self.network = buildNetwork(self.input_matrix_x.shape[1],8,1,bias=True)#outclass = LinearLayer#self.neurons_structure[i/4]

	def backpropagation(self):

		self.trainer = BackpropTrainer(self.network,self.ds, verbose = False, learningrate=0.0001)
		#self.training_error2 = self.trainer.trainEpochs(1000)
		self.training_error, self.validation_error = self.trainer.trainUntilConvergence( dataset=self.training_set ,maxEpochs=2000, verbose = False, continueEpochs=200,validationProportion=0.15)#,trainingData=self.training_set, validationData=self.validation_set)#,#trainingData=self.training_set, validationData=self.validation_set)
		#plt.figure()
		#plt.plot(self.training_error)
		#plt.plot(self.validation_error)
		#plt.show()
	def nn_regression(self):
		fig = plt.figure(figsize=(5,5))
		self.nnRegression = NNregression(self.ds, hidden=self.number_of_months - 2, VDS=self.val_data, epoinc=750,Graph=fig)# VDS=self.val_data
		self.nnRegression.setupNN()
		self.nnRegression.initGraphics(ymax=1, xmax=-1)
		self.run  = self.nnRegression.runTraining(convergence=0.00000001)

	def backpropagation_plot(self):
		
		plt.figure(figsize=(2,1))
		plt.plot(self.training_error[:],'b', self.validation_error[:],'r')
		plt.legend(loc=2)
		plt.show()

	def test_coefficients(self):

		self.test_prediction=[]
		for k in self.test_values_x_norm.index:
			self.test_prediction.append(self.network.activate(self.test_values_x_norm.ix[k:k,:].values[0]))

		ymax = self.training_values_y.max()
		ymin = self.training_values_y.min()
		self.test_error = abs(pd.DataFrame((np.array(self.test_prediction)*(ymax-ymin))+ymin)-pd.DataFrame(self.test_values_y.as_matrix()))

	def forecast_plot(self):
		plt.figure(figsize=(5,5))
		ymax = self.training_values_y.max()
		ymin = self.training_values_y.min()
		plt.plot(self.test_values_y.tolist(),marker='*',markersize=10)
		plt.plot((np.array(self.test_prediction)*(ymax-ymin))+ymin,marker='*',markersize=10)
		print(self.test_error,self.test_error.mean())
		#plt.title(str(Ano_Inicio +10)+'-'+str(Ano_Inicio+20-1)+',  Erro_av=:'+str(erro.mean().round(2)))
		plt.show()

class support_vector_regression(artificial_neural_network):

	def __init__(self):
		pass

	def svm_fit(self):
		X = self.input_matrix_x
		y = self.input_matrix_y.reshape(-1)
		self.clf = svm.SVR(kernel='linear',gamma=0.00000001)#, degree=2)
		self.clf.fit(X, y)

	def svm_predict(self):
		self.test_prediction = self.clf.predict(self.validation_values_x_norm.values)
		ymax = self.training_values_y.values.max()
		ymin = self.training_values_y.values.min()
		self.predicted =( np.array(self.test_prediction)*(ymax-ymin))+ymin
		self.test_error = abs(pd.DataFrame((np.array(self.test_prediction)*(ymax-ymin))+ymin)-pd.DataFrame(self.test_values_y.as_matrix()))


























