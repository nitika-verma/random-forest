from featureEngineering import clean_data
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":

	passengerIdsTrain, trainData = clean_data("train.csv")
	passengerIdsTest, testData = clean_data("test.csv")

	randomForest = RandomForestClassifier(n_estimators=200)
	randomForest = randomForest.fit(trainData[0::,1::], trainData[0::,0])

	output = randomForest.predict(testData).astype(int)

	print "Output Data"
	print passengerIdsTest, output
