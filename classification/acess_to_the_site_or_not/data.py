import csv

def loading_acess():

	X = []
	Y = []

	file = open('acess.csv', 'rb')
	reader = csv.reader(file)

	reader.next()

	for home,how_it_works,contact,buy in reader:

		data = [int(home), int(how_it_works), int(contact)]
		X.append(data)
		Y.append(int(buy))

	return X, Y