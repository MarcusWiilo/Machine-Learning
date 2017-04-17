import csv

def loading_search():

	X = [];
	Y = [];

	file = open('search.csv', 'rb')
	reader = csv.reader(file)
	reader.next()
	for home,search,logged,bought in reader:
		data = [int(home), search, int(logged)]
		X.append(data)
		Y.append(int(bought))

	return X,Y

