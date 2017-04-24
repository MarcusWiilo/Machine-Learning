import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier

# innitial test: home, search, logged => bought
# home , search
# home, logged
# search, logged
# search: 75,00% (8 tests)

df = pd.read_csv('search2.csv')

X_df = df[['home', 'search', 'logged']]
Y_df = df['bought']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

percent_of_training = 0.8
percent_of_test = 0.1

# TODO: Verify if  the ranges are ok with the int transformation

size_of_training = int(percent_of_training * len(Y))
size_of_test = int(percent_of_test * len(Y))
size_of_validation = len(Y) - size_of_training - size_of_test

train_data = X[0:size_of_training]
train_marcations = Y[0:size_of_training]

end_of_test = size_of_training + size_of_test
test_data = X[size_of_training:end_of_test]
test_marcations = Y[size_of_training:end_of_test]

validation_of_data = X[end_of_test:]
validation_of_marcation = Y[end_of_test:]

def fit_and_predict(nome, model, train_data, train_marcations, test_data, test_marcations):

	model.fit(train_data, train_marcations)

	results = model.predict(test_data)
	hits = (results == test_marcations)

	total_of_hits = sum(hits)
	total_of_elements = len(test_data)
	hit_rate = 100.0 * total_of_hits / total_of_elements

	msg = "Hit rate of {0}: {1}".format(nome, hit_rate)
	print(msg)

model = MultinomialNB()
fit_and_predict("MultinomialNB", model, train_data, train_marcations, test_data, test_marcations)

model = AdaBoostClassifier()
fit_and_predict("AdaBoostClassifier", model, train_data, train_marcations, test_data, test_marcations)

# the effectivenes of the algorithm that kicks all in one values

hit_base = max(Counter(test_marcations).itervalues())
hit_rate_base = 100.0 * hit_base / len(test_marcations)

print("Hit rate base: %f" % hit_rate_base)
print("Total of tests: %d " % len(test_data))

