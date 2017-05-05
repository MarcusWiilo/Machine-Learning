import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from collections import Counter


df = pd.read_csv('search.csv')

X_df = df[['home', 'search', 'logged']]
Y_df = df['bought']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

percent_of_training = 0.9

# TODO: Verify if  the ranges are ok with the int transformation

size_of_training = int(percent_of_training * len(Y))
size_of_test = int(len(Y) - size_of_training)

train_data = X[:size_of_training]
train_marcations = Y[:size_of_training]

test_data = X[-size_of_test:]
test_marcations = Y[-size_of_test:]

model = MultinomialNB(alpha=1.0, class_prior=None)
model.fit(train_data, train_marcations)

results = model.predict(test_data)
hits = (results == test_marcations)

total_of_hits = sum(hits)
total_of_elements = len(test_data)
hit_rate = 100.0 * total_of_hits / total_of_elements

print("Hit rate of algorithm: %f" % hit_rate)
print(total_of_elements)

# the effectivenes of the algorithm that kicks all in one values

hit_base = max(Counter(test_marcations).itervalues())
hit_rate_base = 100.0 * hit_base / len(test_marcations)
print("Hit rate base: %f" % hit_rate_base)

