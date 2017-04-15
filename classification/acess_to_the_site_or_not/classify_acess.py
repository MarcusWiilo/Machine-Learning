# My initial approach
# Separate 90% of data for training and 10% for test: 55,55%

from sklearn.naive_bayes import MultinomialNB
from data import loading_acess

X,Y = loading_acess()

train_data = X[:90]
train_marcation = Y[:90]

test_data = X[-9:]
test_marcation = Y[-9:]

model = MultinomialNB()
model.fit(train_data, train_marcation)

results = model.predict(test_data)
differences = results - test_marcation

hits = [d for d in differences if d ==0]
total_of_hits = len(hits)
total_of_elements = len(test_data)
hit_rate = 100.0 * total_of_hits / total_of_elements

print(hit_rate)
print(total_of_elements)