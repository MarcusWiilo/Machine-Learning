from sklearn.naive_bayes import MultinomialNB
from data import loading_acess

X,Y = loading_acess()

model = MultinomialNB()
model.fit(X, Y)

results = model.predict(X)
differences = results - Y

hits = [d for d in differences if d ==0]
total_of_hits = len(hits)
total_of_elements = len(X)
hit_rate = 100.0 * total_of_hits / total_of_elements

print(total_of_hits)
print(total_of_elements)