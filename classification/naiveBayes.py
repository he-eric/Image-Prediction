# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import random
import math

K = .3

def countLabel(trainingLabels):
  labelCounter = util.Counter()
  labelCounter.incrementAll(trainingLabels, 1)
  return labelCounter

def sortDatum(trainingData, trainingLabels, legalLabels):
  """
  Sort datums into a dict
    Key ===== Value
    Map labels face -> { datum1_features, datum2_features...}
    Map labels with digit -> {...}
  """
  sortedDatums = {}
  for i in range(len(legalLabels)):
    sortedDatums[i] = []
  intLabelIndex = 0
  for datum in trainingData:
    intLabel = trainingLabels[intLabelIndex]
    sortedDatums[intLabel].append(datum)
    intLabelIndex+=1
  return sortedDatums

def generateFeatureCounterForDigits(labelDatums, features, value):
  """
  Given a list of datums belonging to the same digit, create a counter for the value given
  In other words, return a data structure that stores the count for each feature

  Return:
    EG: Digit 7
    ________________________________
   |__Value___________ 1____________|
   | Feature1 |________0____________|
   | Feature2 |________5____________|
   | Feature3 |________3____________|
   | ....     |________...__________|
   | FeatureN |________x____________|
  """
  featureCounter = util.Counter()
  for feature in features:
    featureCounter[feature]
  for datum in labelDatums:
    for feature in datum:
      if datum[feature] == value:
        featureCounter[feature] += 1
  return featureCounter

def generateFeatureCounterForFaces(labelDatums, features, value1, value2):
  featureCounter = util.Counter()
  for feature in features:
    featureCounter[feature] = 1
  for datum in labelDatums:
    for feature in datum:
      if datum[feature] == value1:
        featureCounter[feature] += 1
  return featureCounter

def generateProbTableForDigits(labelDatums, features):
  featureCounter0 = generateFeatureCounterForDigits(labelDatums, features, 0)
  featureCounter1 = generateFeatureCounterForDigits(labelDatums, features, 1)
  featureCounter2 = generateFeatureCounterForDigits(labelDatums, features, 2)
  featureCounter1.__radd__(featureCounter2)
  for feature in features:
    if featureCounter0[feature] == 0:
      featureCounter0[feature] = 0
    if featureCounter1[feature] == 0:
      featureCounter1[feature] = 0
    # print str(featureCounter0[feature]) + ", " + str(featureCounter1[feature])
    featureCounter0[feature] = (featureCounter0[feature]+K) / float(len(labelDatums)+K)
    featureCounter1[feature] = (featureCounter1[feature]+K) / float(len(labelDatums)+K)
    # print str(featureCounter0[feature]) + ", " + str(featureCounter1[feature])
  return [featureCounter0, featureCounter1]

def generateProbTableForFaces(labelDatums, features):
  featureCounter5 = generateFeatureCounterForFaces(labelDatums, features, 0, 5)
  featureCounter10 = generateFeatureCounterForFaces(labelDatums, features, 5, 10)
  featureCounter15 = generateFeatureCounterForFaces(labelDatums, features, 10, 15)
  featureCounter20 = generateFeatureCounterForFaces(labelDatums, features, 15, 20)
  featureCounter25 = generateFeatureCounterForFaces(labelDatums, features, 20, 25)
  for feature in features:
    featureCounter5[feature] = (featureCounter5[feature] / float(len(labelDatums)))
    featureCounter10[feature] = (featureCounter10[feature] / float(len(labelDatums)))
    featureCounter15[feature] = (featureCounter15[feature] / float(len(labelDatums)))
    featureCounter20[feature] = (featureCounter20[feature] / float(len(labelDatums)))
    featureCounter25[feature] = (featureCounter25[feature] / float(len(labelDatums)))
  return [featureCounter5, featureCounter10, featureCounter15, featureCounter20, featureCounter25]

def calculateLabelLogProbForDigits(datum, label, probTables):
  floatLabelLogProb = 0.0
  labelProbTable = probTables[label]
  for feature in datum:
    intPixelValue = datum[feature]
    if (intPixelValue == 0):
      floatProb = labelProbTable[0][feature]
      floatLogProb = abs(math.log(floatProb, 10))
      floatLabelLogProb = floatLabelLogProb + floatLogProb
    else:
      floatProb = labelProbTable[1][feature]
      floatLogProb = abs(math.log(floatProb, 10))
      floatLabelLogProb = floatLabelLogProb + floatLogProb
  return floatLabelLogProb

def calculateLabelLogProbForFaces(datum, label, probTables):
  floatLabelLogProb = 1.0
  labelProbTable = probTables[label]
  for feature in datum:
    intPixelCount = datum[feature]
    if intPixelCount < 5:
      floatProb = labelProbTable[0][feature]
      floatLogProb = abs(math.log(floatProb, 10))
      floatLabelLogProb = floatLabelLogProb * floatProb
    elif intPixelCount < 10:
      floatProb = labelProbTable[1][feature]
      floatLogProb = abs(math.log(floatProb, 10))
      floatLabelLogProb = floatLabelLogProb * floatProb
    elif intPixelCount < 15:
      floatProb = labelProbTable[2][feature]
      floatLogProb = abs(math.log(floatProb, 10))
      floatLabelLogProb = floatLabelLogProb * floatProb
    elif intPixelCount < 20:
      floatProb = labelProbTable[3][feature]
      floatLogProb = abs(math.log(floatProb, 10))
      floatLabelLogProb = floatLabelLogProb * floatProb
    elif intPixelCount <= 25:
      floatProb = labelProbTable[4][feature]
      floatLogProb = abs(math.log(floatProb, 10))
      floatLabelLogProb = floatLabelLogProb * floatProb
  return floatLabelLogProb

def printProbTables(probTables):
  x = 0
  for probTable in probTables:
    for counter in probTable:
      for features in counter:
        print features
        print counter[features]
        x+=1
        if x == 5:
          break
      x = 0
      print ""

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.size = 0
    self.trainingLabels = None
    self.probabilityTables = []
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));

    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.

    For digits 0-9 create a table
    For digit 0
    Count all labels with 0
    Make a table for digit 0
    Make 784 features as rows for the table
    Make 2 values as columns for the table 0 or 1
    For each pixel/feature calculate the probability
      For each datum, look at pixel i, see if it is 0 or 1
      Number of digit 0s with pixel i as 0 / Number of digit 0s = probability for value 0 at pixel i
      Number of digit 0s with pixel i as 1 / Number of digit 0s = probability for value 1 at pixel i

    For faces...
    ...
    """

    "*** YOUR CODE HERE ***"
    self.size = len(trainingData)
    self.trainingLabels = trainingLabels
    sortedDatums = sortDatum(trainingData, trainingLabels, self.legalLabels)
    if len(self.legalLabels) == 10:
      for key in sortedDatums:
        table = generateProbTableForDigits(sortedDatums[key], self.features)
        self.probabilityTables.append(table)
    else:
      for key in sortedDatums:
        table = generateProbTableForFaces(sortedDatums[key], self.features)
        self.probabilityTables.append(table)

        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.

    For digits
    For digit 0
      Get the value at each pixel 0 or 1
      Reference the table created from training
      Get the probability of each pixel
      Multiply them all
      Put into logJoint
    Do for all digits
    """
    logJoint = util.Counter()
    "*** YOUR CODE HERE ***"
    labelCounter = countLabel(self.trainingLabels)
    if len(self.legalLabels) == 10:
      # print "######################################"
      for i in range(len(self.legalLabels)):
        floatPriorProb = float(labelCounter[i]) / self.size
        logJoint[i] = -(math.log(floatPriorProb,10) + calculateLabelLogProbForDigits(datum, i, self.probabilityTables))
      #   print "logJoint[" + str(i) + "] = " + str(logJoint[i])
      # print "######################################"
    else:
      for i in range(len(self.legalLabels)):
        floatPriorProb = float(labelCounter[i]) / self.size
        # print self.size
        # print labelCounter[i]
        # print floatPriorProb
        logJoint[i] = floatPriorProb * calculateLabelLogProbForDigits(datum, i, self.probabilityTables)
        #print logJoint[i]
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
    

    
      
