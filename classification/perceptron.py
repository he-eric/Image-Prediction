# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Perceptron implementation
import util
import random
PRINT = True

def defineWeights(weights, legalLabels):
  for label in legalLabels:
    weights[label] = util.Counter()  # this is the data-structure you should use

def defineBias(bias, legalLabels):
  for label in legalLabels:
    bias.append(random.uniform(-1, 1))

def randomWeights(weights, features):
  for keys in features:
    weights[keys] = random.uniform(-1, 1)

def randomBias(bias, legalLabels):
  for label in legalLabels:
    bias[label] = random.uniform(-1,1)

class PerceptronClassifier:
  """
  Perceptron classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "perceptron"
    self.max_iterations = 5
    self.weights = {}
    self.bias = []
    defineWeights(self.weights, self.legalLabels)
    defineBias(self.bias, self.legalLabels)

  def setWeights(self, weights):
    assert len(weights) == len(self.legalLabels);
    self.weights == weights;

  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details. 
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """

    defineWeights(self.weights, self.legalLabels)
    randomBias(self.bias, self.legalLabels)
    self.features = trainingData[0].keys() # could be useful later
    # for i in self.legalLabels:
    #   randomWeights(self.weights[i], self.features)
    # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
    # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
    counter = util.Counter()
    for iteration in range(self.max_iterations):
      print "Starting iteration ", iteration, "..."
      for i in range(len(trainingData)):
        for j in self.legalLabels:
          # counter = dict
          # associate a label with a f(x)
          counter[j] = trainingData[i].__mul__(self.weights[j]) #+ self.bias[j]
        if not trainingLabels[i] == counter.argMax():
          self.weights[trainingLabels[i]].__radd__(trainingData[i])
          self.weights[counter.argMax()].__sub__(trainingData[i])
          self.bias[trainingLabels[i]]+=1.0
          self.bias[counter.argMax()]-=1.0

  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  def findHighWeightFeatures(self, label):
    """
    Returns a list of the 100 features with the greatest weight for some label
    """
    featuresWeights = self.weights[label].values()
    #"*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    return featuresWeights

