from sklearn import tree
from sklearn import neighbors
import pandas as pd

clf_tree = tree.DecisionTreeClassifier()
clf_neighbors = neighbors.KNeighborsClassifier(20)

filename = '../datasets/gender_classifier/Howell_data1.csv'

df = pd.read_csv(filename)

datasize = (len(df.index))
split = int(0.75* datasize)

train = ([],[])
test = ([],[])
# [index, height, weight, age, male]
for row in df.itertuples():
	if row[0] <= split:
		train[0].append([row[1],row[2],row[3]])
		train[1].append(row[4])
	else:
		test[0].append([row[1],row[2],row[3]])
		test[1].append(row[4])
	

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

clf_tree = clf_tree.fit(train[0], train[1])
clf_neighbors = clf_neighbors.fit(train[0], train[1])

prediction = clf_tree.predict(test[0])
prediction2 = clf_neighbors.predict(test[0])



globalCount=0
count = 0
for e1, e2 in zip(prediction, test[1]):
	globalCount+=1
	if e1 != e2:
		count+=1

count2 = 0
for e1, e2 in zip(prediction2, test[1]):
	if e1 != e2:
		count2+=1
		
print("Incorrect pracentage {}%".format((count * 100)/globalCount))
print("Incorrect pracentage {}%".format((count2 * 100)/globalCount))


