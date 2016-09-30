import pandas as pandas
import numpy as numpy

def clean_data(file_path):
	data = pandas.read_csv(file_path, header=0)

	# Replace null ages with median
	median_age = data.Age.dropna().median()
	if len(data.Age[data.Age.isnull()]) > 0:
		data.loc[(data.Age.isnull()), 'Age'] = median_age

	# Convert all strings to int values
	data.Gender = data.Sex.map({'male' : 0, 'female' : 1}).astype(int)

	# Replace all null embarked to most common place
	if len(data.Embarked[data.Embarked.isnull()]) > 0:
		data.Embarked[data.Embarked.isnull()] = data.Embarked.dropna().mode().values

    # Convert all strings to int values by using a dictionary
	Ports = list(enumerate(numpy.unique(data['Embarked'])))    
	PortsDict = {name : i for i, name in Ports}              
	data.Embarked = data.Embarked.map(lambda x: PortsDict[x]).astype(int) 

	# Replace all null fares with median of respective class fares
	if len(data.Fare[data.Fare.isnull()]) > 0:
		median_fare = numpy.zeros(3)
		for f in range(0,3):                                              
			median_fare[f] = data[data.Pclass == f+1 ]['Fare'].dropna().median()
		for f in range(0,3):                                              
			data.loc[(data.Fare.isnull()) & (data.Pclass == f+1 ), 'Fare'] = median_fare[f]

 	passengerIds = data['PassengerId'].values

	# Remove all string attributes which are not required
	data = data.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
  
	return passengerIds, data.values

