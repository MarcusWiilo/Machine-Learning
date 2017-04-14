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
test = [misterius1, misterius2]

print(model.predict(test))