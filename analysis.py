import pandas as pd 
import numpy as np 
import seaborn as sbs
import matplotlib.pyplot as plt
from pdb import set_trace as stop
from sklearn.ensemble import RandomForestClassifier

def showMissData(df):
    # how much data is missing?
    for colName in df.columns : 
        print("{0}\t{1}\t{2}").format( \
            colName, \
            len(df) - len(df[colName].dropna()), \
            float(len(df)-len(df[colName].dropna()))/len(df))

def fixMissAge(df):
    #sns.boxplot(y='Age', x='Survived', hue='Pclass', data=train )
    # Based on age, class, and survival status of the remaining passengers, 
    # we can infer the age of the missing values based on survival and Pclass
    # NOTE: maybe we could use sex instead of (or in addition to?) survival status here. 
    ageMedians = { \
        "male":   {1:0, 2:0, 3:0}, \
        "female": {1:0, 2:0, 3:0}  }
    # todo: refactor this into 1 pair of nested loops
    for sex in ageMedians.keys(): 
        for pclass in ageMedians[sex].keys():
            ageMedians[sex][pclass] = df[ (df['Sex']==sex) & (df['Pclass']==pclass) ].Age.median()
    for sex in ageMedians.keys():
        for pclass in ageMedians[sex].keys():
            subdf = df[ (df['Age'].isnull()) & (df['Sex']==sex) & (df['Pclass']==pclass) ]
            df.loc[subdf.index,'Age'] = ageMedians[sex][pclass]
    return df

def prepDF(csvIn):
    df = pd.DataFrame.from_csv(csvIn,header=0, index_col=0)   
    showMissData(df)
    # resolve columns that have missing data: 
    # 1: Cabin missing from 77% of rows in Training Set; drop it
    df.drop('Cabin',axis=1,inplace=True)
    # 2: Age missing from 19% of rows in Training Set ; fix it?
    df = fixMissAge(df)
    # 3: Embarked missing from 2 entires; where could they be from? 
    #sns.boxplot(y='Age', x='Pclass', hue='Embarked', data=train )
    #train.groupby("Embarked").apply(len)
    # looks like S is the most common departure site, which agrees with the distribution of Age/pclass
    df.loc[df[df["Embarked"].isnull()].index,'Embarked'] = "S"    
    # training set all cleaned 
    # convert embarked and sex to factors
    mapEmbarked = {val:count for count,val in enumerate(df.Embarked.unique())}
    df.loc[df.index, 'Embarked'] = df['Embarked'].map(mapEmbarked)
    mapSex = {val:count for count,val in enumerate(df.Sex.unique())}
    df.loc[df.index,'Sex'] = df['Sex'].map(mapSex)
    # drop non-numeric columns
    df.drop(['Ticket','Name'],axis=1,inplace=True)
    return df 

def calcError(trainDF, n=10):
    '''
    loop over the training dataframe n times 
        taking a subset, 
        training the model, and
        calculating the error rate 
    '''
    errArr = []
    i=0
    while i <= n:
        subFrac = np.random.ranf()
        if subFrac < 0.6:
            subFrac = 1-subFrac
        elif subFrac < 0.5 or subFrac > 0.95:
            continue 
        # grab some rows to be sub-training set 
        subTrainInd = trainDF.sample(int(len(trainDF)*subFrac)).index.tolist()
        # grab the rest to be sub-test set, with known answers  
        subTestInd  = set(trainDF.index) - set(subTrainInd)
        # now assemble the sub-dataframes 
        subTrainDF  = trainDF.loc[subTrainInd]
        subTestDF   = trainDF.loc[subTestInd]
        forest = RandomForestClassifier(n_estimators=10)
        forest.fit(subTrainDF.drop('Survived',axis=1), subTrainDF.Survived)
        # %err  calculated as the number of cases we got wrong (including over- and under-estimates), 
        #  divided by the total number of cases 
        errArr.append(abs( subTestDF.Survived - forest.predict(subTestDF.drop('Survived',axis=1)) ).sum()*100.0/float(len(subTestDF)))
        i+=1
    return np.mean(errArr), np.std(errArr),n 


def main():
    train = prepDF('train.csv')
    test = prepDF('test.csv')
    # what does the correlation between variables look like?
    # sbs.heatmap(train.corr())
    # plt.show() 
    # Mostly shows obvious/expected correlations
    #    better class correlated with high ticket price (0.5)
    #    better class correlated with survival (0.3) 
    #    sex correlated with survival (0.5)
    #    correlations between age, siblings, and parents likely stem from families   
    
    # Given this method, how accurate should we expect to be? 
    avgErr, stdvErr, n  = calcError(train, 50)
    print("calculated an average of {}% error with stdev {}% over {} iterations".format(avgErr, stdvErr,n ))
    # at this point, getting consistently ~19% of the answers wrong

    #forest = RandomForestClassifier(n_estimators=100)
    # need to fix 1 missing fare
    # forest.fit(train.drop('Survived',axis=1), train.Survived)
    
    
if __name__ == "__main__": 
    main()
    
