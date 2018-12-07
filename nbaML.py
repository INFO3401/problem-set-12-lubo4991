import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


def loadData(datafile):
    with open(datafile, 'r') as csvfile:
        data = pd.read_csv(csvfile)
        
    #Inspect the data
    print(data.columns.values)
    
    return data

def runKNN(dataset, prediction, ignore):
#Set up our dataset
    
    X = dataset.drop(columns=[prediction, ignore])
    Y = dataset[prediction].values
    
    # SPlit the data into training and testing set
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)
    
    #run K-NN algorithm
    knn = KNeighborsClassifier(n_neighbors=5)
    
    #Train the model
    knn.fit(X_train, Y_train)
    
    #Test the model
    score = knn.score(X_test, Y_test)
    print("Predicts" + prediction + "with" + str(score) + "accuracy")
    print("Chance is:" + str(1.0/len(dataset.groupby(prediction))))
    return knn

def classifyPlayer(targetRow, data, model, prediction, ignore):
    X = targetRow.drop(columns=[prediction, ignore])

    #Determine the five closest neighbors
    neighbors = model.kneighbors(X, n_neighbors=5, return_distance=False)
    
    #Print out the neighbors data
    for neighbor in neighbors[0]:
        print(data.iloc[neighbor])

def runKMeans(dataset, ignore):
    #Set up our dataset
    X = dataset.drop(columns=ignore)
    #Run K-means algorithm
    kmeans = KMeans(n_clusters=5)
    #Train the model
    kmeans.fit(X)
    #Add the predtions to the dataframe
    dataset['cluster'] = pd.Series(kmeans.predict(X), index=dataset.index)
    #Print a scatterplot matrix
    scatterMatrix = sns.pairplot(dataset.drop(columns=ignore), hue='cluster', palette='Set2')
    
    scatterMatrix.savefig("kmeansCluster.png")
    
    return kmeans
    

    


#Test your code


nbaData=loadData("nba_2013_clean.csv")
knnModel = runKNN(nbaData, "pos", "player")
classifyPlayer(nbaData.loc[nbaData['player']=='Kobe Bryant'], nbaData, knnModel, 'pos', 'player')

kmeansModel = runKMeans(nbaData, ['pos', 'player'])

