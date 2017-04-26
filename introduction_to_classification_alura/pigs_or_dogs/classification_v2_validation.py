from sklearn.naive_bayes import MultinomialNB

# it´s fat? have a short legs? say au au?
pig1 = [1, 1, 0]
pig2 = [1, 1, 0]
pig3 = [1, 1, 0]
dog1 = [1, 1, 1]
dog2 = [0, 1, 1]
dog3 = [0, 1, 1]

data = [pig1, pig2, pig3, dog1, dog2, dog3]

marcation = [1, 1, 1, -1, -1, -1]

model = MultinomialNB()
model.fit(data, marcation)

misterius1 = [1, 1, 1]
misterius2 = [1, 0, 0]
misterius3 = [0, 0, 1]
test = [misterius1, misterius2, misterius3]

marcation_test = [-1, 1, -1]

results = model.predict(test)
print(results)

differences = results - marcation_test
print(differences)

hits = [d for d in differences if d==0]

total_of_hits = len(hits)
total_of_elements = len(test)

hit_rate = 100.0 * total_of_hits / total_of_elements
print(hit_rate)