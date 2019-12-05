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
  # INITIALIZE sortedDatums AS A EMPTY DICTIONARY
  sortedDatums = {}

  # INITIALIZE sortedDatums WITH KEYS AND ASSIGN EMPTY LISTS TO EACH KEY
  for i in range(len(legalLabels)):
    sortedDatums[i] = []

  # APPEND EACH DATUM IN THE TRAINING DATA TO THE CORRECT LIST USING THE DATUM'S LABEL AS KEY
  intLabelIndex = 0
  for datum in trainingData:
    intLabel = trainingLabels[intLabelIndex]
    sortedDatums[intLabel].append(datum)
    intLabelIndex+=1

  # RETURN A DICTIONARY OF SORTED DATUMS
  return sortedDatums

# labelDatums - LIST OF DATUMS WITH LABEL 'X'
# features - LIST OF ALL POSSIBLE FEATURES
# value - VALUE TO MEASURE
def generateFeatureCounter(labelDatums, features, value):
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
  # INITIALIZE THE COUNTER
  featureCounter = util.Counter()

  # INITIALIZE THE 'KEYS'
  for feature in features:
    featureCounter[feature]

  # START COUNTING THE NUMBER OF FEATURES WITH GIVEN VALUE
  for datum in labelDatums:
    for feature in datum:
      if datum[feature] == value:
        featureCounter[feature] += 1

  # RETURN A DICTIONARY/COUNTER THAT COUNTS THE NUMBER OF DATUMS FOR EACH FEATURE WITH THE GIVEN VALUE
  return featureCounter

# labelDatums - LIST OF DATUMS CORRESPONDING TO A SPECIFIC LABEL 'X'
# features - LIST OF ALL POSSIBLE FEATURES
def generateProbTable(labelDatums, features):
  # GENERATE THE COUNT FOR EACH FEATURE WITH VALUE 0
  featureCounter0 = generateFeatureCounter(labelDatums, features, 0)
  # GENERATE THE COUNT FOR EACH FEATURE WITH VALUE 1
  featureCounter1 = generateFeatureCounter(labelDatums, features, 1)
  # GENERATE THE COUNT FOR EACH FEATURE WITH VALUE 2
  featureCounter2 = generateFeatureCounter(labelDatums, features, 2)
  # COMBINE THE COUNT FOR FEATURES WITH VALUES 1 OR 2
  featureCounter1.__radd__(featureCounter2)

  # CALCULATE THE PROBABILITY
  # PROBABILITY OF FEATURE = #_OF_DATUMS_WITH_FEATURE_WITH_VALUE_NOT_0 / #_TOTAL_NUMBER_OF_DATUMS_WITH_LABEL_X
  for feature in features:

    # ACCOUNT FOR 0 PROBABILITY
    if featureCounter0[feature] == 0:
      featureCounter0[feature] = 1
    if featureCounter1[feature] == 0:
      featureCounter1[feature] = 1

    featureCounter0[feature] = (featureCounter0[feature]+K) / float(len(labelDatums)+K)
    featureCounter1[feature] = (featureCounter1[feature]+K) / float(len(labelDatums)+K)

  # RETURN A LIST OF 2 COUNTERS
  # ONLY FOR DATUMS WITH LABEL 'X'
  # FIRST COUNTER - PROBABILITY OF FEATURE WITH VALUE 0
  # SECOND COUNTER - PROBABILITY OF FEATURE WITH VALUE NOT 0
  return [featureCounter0, featureCounter1]

# datum - TESTING DATUM
# label - PROBABILITY OF THIS DATUM BEING TH LABEL 'X'
# probTables - LIST OF PROBABILITY TABLES
def calculateLabelLogProb(datum, label, probTables):
  # INITIALIZE THE PROB TO 0.0
  floatLabelLogProb = 0.0

  # GET THE RIGHT PROBABILITY TABLE IN THE LIST OF PROBABILITY TABLES
  labelProbTable = probTables[label]

  # ADD ALL LOG PROBABILITIES
  for feature in datum:
    # VALUE AT THIS FEATURE
    intPixelValue = datum[feature]
    if (intPixelValue == 0):
      # IF VALUE AT THIS FEATURE/PIXEL EQUALS 0
      # GET THE PROBILITY OF LABEL 'X' WITH THIS FEATURE/PIXEL EQUAL TO 0
      floatProb = labelProbTable[0][feature]

      # TRANSFORM THE PROBABILITY WITH LOG
      floatLogProb = math.log(floatProb, 10)

      # SUM THE LOG PROBABILITIES
      floatLabelLogProb = floatLabelLogProb + floatLogProb
    else:
      # IF VALUE AT THIS FEATURE/PIXEL EQUALS NOT 0
      # GET THE PROBILITY OF LABEL 'X' WITH THIS FEATURE/PIXEL EQUAL TO NOT 0
      floatProb = labelProbTable[1][feature]

      # TRANSFORM THE PROBABILITY WITH LOG
      floatLogProb = math.log(floatProb, 10)

      # SUM THE LOG PROBABILITIES
      floatLabelLogProb = floatLabelLogProb + floatLogProb

  # RETURN THE LOG PROBABILITY THAT THE DATUM IS LABEL 'X'
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

    "*** YOUR CODE HERE ***"
    # SET THE SIZE OF THE TRAINING DATA
    self.size = len(trainingData)
    # SET THE TRAINING LABELS
    self.trainingLabels = trainingLabels
    # SORT THE TRAINING DATA
    # sortedDatums is a DICTIONARY
    # KEYS are LABELS --- VALUES are a LIST OF DIGITS/FACES corresponding to the label
    sortedDatums = sortDatum(trainingData, trainingLabels, self.legalLabels)

    if len(self.legalLabels) > 1:
      # GENERATE A PROB TABLE FOR EACH LABEL (EG: 9 tables for digits; 2 tables for faces)
      for key in sortedDatums:
        table = generateProbTable(sortedDatums[key], self.features)
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

  # datum - DIGIT/FACE TO BE PREDICTED
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    "*** YOUR CODE HERE ***"
    labelCounter = countLabel(self.trainingLabels)
    if len(self.legalLabels) > 1:

      # CALCULATE THE PROBABILITY OF THE DATUM FOR EACH LABEL
      for i in range(len(self.legalLabels)):

        # GET THE PRIOR PROBABILITY
        floatPriorProb = float(labelCounter[i]) / self.size

        # ADD THE LOG PRIOR PROBABILITY TO THE LOG
        logJoint[i] = (math.log(floatPriorProb,10) + calculateLabelLogProb(datum, i, self.probabilityTables))

    # RETURN A LIST OF PROBABILITIES
    # [PROB(DATUM=0), PROB(DATUM=1), PROB(DATUM=2), .... PROB(DATUM=9]
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
    

    
      
