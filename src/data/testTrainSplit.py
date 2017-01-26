import pandas as pd
from sklearn.cross_validation import train_test_split
import pickle

candidates = pd.read_csv('../data/candidates.csv')

positives = candidates[candidates['class']==1].index  
negatives = candidates[candidates['class']==0].index

## Under Sample Negative Indexes
np.random.seed(42)
negIndexes = np.random.choice(negatives, len(positives)*5, replace = False)

candidatesDf = candidates.iloc[list(positives)+list(negIndexes)]

X = candidatesDf.iloc[:,:-1]
y = candidatesDf.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
