#Liquin Yu
#hw6

import numpy as np
import math
import matplotlib.pyplot as plt
import csv
from scipy import io
import random
from sklearn.metrics import zero_one_loss


#load data and put into desired format
trainingData = io.loadmat('digit-dataset/train.mat')
trainingImages = np.array(trainingData['train_images'])
trainingImages = trainingImages.transpose()
trainingImages = np.matrix(np.reshape(trainingImages, [60000, 784]))/float(256)
trainingLabels = np.matrix(trainingData['train_labels'])
means = np.mean(np.matrix(trainingImages), axis=0)
std = np.std(np.matrix(trainingImages), axis=0) + 0.000001

#get/set sizes for things we need
n_in = 784
n_hid = 200
n_out = 10
trainingDataSize = np.size(trainingLabels)

#create a 10x1 matrix with values 0 expect for at position index 
def labelToVector(index):
	vector = np.zeros(10)
	vector[index] = 1
	return np.matrix(vector).transpose()

def leastSquareLoss(labelPred, yVector):
	assert (yVector.shape == labelPred.shape)
	return 1/2*np.matrix.sum(np.square(yVector - labelPred))

def leastSquareDerivDelta(labelPred, yVector):
	assert yVector.shape == labelPred.shape
	return np.multiply(labelPred-yVector,np.multiply(labelPred, (1-labelPred)))

def tanhDeriv(dotProdVector):
	return 1-np.square(np.tanh(dotProdVector))

def sigmoid(dotProdVector):
	return 1.0/(1+np.exp(-dotProdVector))

def crossEntropyLoss(labelPred, yVector):
	return -np.matrix.sum(np.multiply(yVector*np.log(labelPred))+np.multiply((1-yVector),np.log(1-labelPred)))

def crossEntropyDerivDelta(labelPred, yVector):
	return np.add(labelPred,-yVector)

def trainNeuralNetwork(images, labels, costFunct, costFunctDerivDelta, numIter):
	#create random weights as a starting point
	weights_hid = 0.001*np.random.randn(n_hid, n_in+1)
	weights_out = 0.001*np.random.randn(n_out, n_hid+1)


	#used to append bias term
	bias = np.matrix(np.ones(1))

	#for plotting
	
	xAxis = [0,40000,80000,120000,160000,200000,240000,280000,320000,360000,400000]
	errorArray = []
	errorArray.append(zero_one_loss(labels,predictNeuralNetwork(weights_hid, weights_out, images)))
	
	#----

	i = 0
	while(i < numIter):
		learningRate = 1/float(i/1000+1000)

		#---pick a random training data point
		randDataIndex = random.sample(range(0,trainingDataSize), 1)
		dataPoint = trainingImages[randDataIndex].transpose()
		dataPointWithBias = np.append(dataPoint, bias, axis = 0)
		dataLabel = trainingLabels[randDataIndex]
		#change label into 10x1 Matrix
		yVector = labelToVector(dataLabel.item(0))

		#forwardpass
		dotProd_hid = np.dot(weights_hid,dataPointWithBias)
		output_hid = np.tanh(dotProd_hid)
		outputWithBias_hid = np.append(output_hid, bias, axis = 0)
		dotProd_out = np.dot(weights_out,outputWithBias_hid)
		output_out = sigmoid(dotProd_out)

		#--backwardpass
		#10x1
		delta = costFunctDerivDelta(output_out,yVector)
		#10x200
		gradientW2_out = np.dot(delta, outputWithBias_hid.transpose())
		assert gradientW2_out.shape == (10,201)
		#remove weight column for bias
		weightsNoBias_out = np.delete(weights_out,np.size(weights_out, axis=1)-1, axis=1)
		gradientW1_out_temp1 = np.dot(delta.transpose(),weightsNoBias_out).transpose()
		assert gradientW1_out_temp1.shape == (200,1)
		gradientW1_out_temp2 = np.multiply(gradientW1_out_temp1, tanhDeriv(dotProd_hid))
		assert gradientW1_out_temp2.shape == (200,1)
		gradientW1_out = np.dot(gradientW1_out_temp2, dataPointWithBias.transpose())
		assert gradientW1_out.shape == (200,785)

		#--update
		weights_out = np.add(weights_out, -(learningRate*gradientW2_out))
		weights_hid = np.add(weights_hid, -(learningRate*gradientW1_out))

		i += 1

		#for plotting
		
		if i in xAxis:
			errorArray.append(zero_one_loss(labels,predictNeuralNetwork(weights_hid, weights_out, images)))
		
		#----

	#for plotting
	
	plt.plot(xAxis, errorArray, 'b-')
	plt.title('Squared Loss Gradient Update')
	plt.xlabel('iterations')
	plt.ylabel('error')
	plt.show()
	
	#------

	return weights_hid, weights_out

def predictNeuralNetwork(weights1, weights2, testImages):
	size = np.size(testImages, axis=0)
	biasVector = np.matrix(np.ones(size)).transpose()

	testImages = np.append(testImages, biasVector, axis = 1)
	dotProd_hid = np.dot(testImages, weights1.transpose())
	output_hid = np.tanh(dotProd_hid)
	outputWithBias_hid = np.append(output_hid, biasVector, axis = 1)
	dotProd_out = np.dot(outputWithBias_hid, weights2.transpose())
	output_out = sigmoid(dotProd_out)
	return np.argmax(output_out, axis = 1)


weights1, weights2 = trainNeuralNetwork(trainingImages, trainingLabels, leastSquareLoss, leastSquareDerivDelta, 400000)

#For Kaggle
'''
testData = io.loadmat('digit-dataset/test.mat')
testImages = np.array(testData['test_images'])
testImages = testImages.transpose()
size = len(testImages)
testImages = np.matrix(np.reshape(testImages, [size, 784]))/float(256)

pred = predictNeuralNetwork(weights1, weights2, testImages)

testCSV = csv.writer(open('imageKagglePredictions.csv', 'wt'))
testCSV.writerow(['Id', 'Category'])
for i in range(0,len(pred)):
	testCSV.writerow([i+1,int(pred[i])])
'''
#-----


pred = predictNeuralNetwork(weights1, weights2, trainingImages)
print zero_one_loss(trainingLabels,pred)




