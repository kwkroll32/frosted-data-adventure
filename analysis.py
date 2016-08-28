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

def setTitle(df):
    df["Title"] = df.Name.apply(lambda x: x.split(',')[1]).apply(lambda x: x.split(".")[0].strip())
    norm = {"Mr":None, "Mrs":None, "Ms":None}
    norm['Mr'] = df[df["Title"].apply(lambda x: x.strip() in ['Don', 'Rev', 'Sir', 'Col', 'Capt','Jonkheer','Major'] )].index
    # male doctors get Mr, female doctors get "Lady" so they can be dealt with next 
    df.loc[df[(df["Title"] == "Dr") & (df["Sex"]=="male")].index, "Title"] = "Mr"
    df.loc[df[(df["Title"] == "Dr") & (df["Sex"]=="female")].index, "Title"] = "Lady"
    # those with parentheses are married, those without are Ms.
    # NOTE: this could get more accurate by incorporating sibsp
    norm['Mrs'] = df[df.apply(lambda x: (x["Title"] in ['Mme','Lady','Mlle','the Countess', 'Dona']) and ( '(' in x['Name']) ,axis=1)].index
    norm['Ms']  = df[df.apply(lambda x: (x["Title"] in ['Mme','Lady','Mlle','the Countess', 'Dona']) and ( '(' not in x['Name']) ,axis=1)].index
    for title in norm.keys():
        df.loc[norm[title],'Title'] = title 
    return df

def fixMissAge(df):
    #sns.boxplot(y='Age', x='Survived', hue='Pclass', data=train )
    # Based on age, class, and title of the remaining passengers, 
    # we can infer the age of the missing values 
    ageMedians = { \
        "Mr":     {1:0, 2:0, 3:0}, \
        "Master": {1:0, 2:0, 3:0}, \
        "Mrs":    {1:0, 2:0, 3:0}, \
        "Ms":     {1:0, 2:0, 3:0}, \
        "Miss":   {1:0, 2:0, 3:0}  }
    # todo: refactor this into 1 pair of nested loops
    for title in ageMedians.keys(): 
        for pclass in ageMedians[title].keys():
            ageMedians[title][pclass] = df[ (df['Title']==title) & (df['Pclass']==pclass) ].Age.median()
    for title in ageMedians.keys():
        for pclass in ageMedians[title].keys():
            subdf = df[ (df['Age'].isnull()) & (df['Title']==title) & (df['Pclass']==pclass) ]
            df.loc[subdf.index,'Age'] = ageMedians[title][pclass]
    return df

def setFamSize(df):
    '''
    determine family size; children, parents, spouses, and/or siblings
    factors: none: 0, passenger + 1, passenger + 2, or passenger + 3 or more  
    '''
    df.loc[df.index, "surname"] = df['Name'].str.split(',').apply(lambda x: x[0])
    fam = df[ (df['SibSp']!=0) | (df['Parch']!=0) ].index
    singletons = set(df.index) - set(fam)
    # sbs.countplot(x=fam.apply(lambda x: x['Parch'] + x['SibSp'],axis=1)) 
    # can be divided into 3 groups: +1s, +2s, and >+3s, 
    #                        i.e. pairs, 1+2, and families larger than 3
    df.loc[fam,'famSize'] = df.loc[fam].apply(lambda x: x['Parch'] + x['SibSp'],axis=1)
    df.loc[singletons,'famSize'] = 0
    df['famSize'] = df['famSize'].astype(int)
    # map family sizes into the 3 factors 
    # could build dict programatically to be more robust to other data sets
    convertSize = {0:0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 10: 3}
    df.loc[df.index,'famSize'] = df['famSize'].map(convertSize)
    return df
    
def fixMissingCabin(df):
    '''
    determine distribution of cabins occupied by each class 
    assumes that surnames are already established (run setFamRel before running this)
    '''    
    # assuming that families shared the same ticket number
    # fam.groupby(["Ticket","surname"]).apply(lambda x: x["Cabin"].unique())
    # according to the above groupby, families tended to share a cabin, or at least be nearby
    cabins = df.groupby(["Ticket","surname"]).apply(lambda x: x["Cabin"].str[0]).reset_index()[['PassengerId','Cabin']]
    cabins.columns = pd.Index(['PassengerId','CabLett'])
    df = df.merge(cabins, left_index=True, right_on="PassengerId")
    # many cabins are still null at this point
    # but cabin is a function of class, i.e. higher class correlated with higher letters
    # sbs.countplot(x='CabLett',hue='Pclass',data=df,order=sorted(df['CabLett'].unique()))
    # calculate the proportion of cabins present at each class level
    #print(df.pivot_table(index='CabLett', columns='Pclass',aggfunc='count')['PassengerId'])
    classProbDict = df.pivot_table(index='CabLett', columns='Pclass',aggfunc='count')['PassengerId'].apply(lambda x: {m:float(x[m])/x.sum() for m in x.index} ).to_dict()
    # apply the proportions to as-yet unknown cabins 
    for pclass in classProbDict.keys():
        tot = len(df[(df['CabLett'].isnull()) & (df['Pclass'] == pclass)])
        for cabin in classProbDict[pclass].keys():        
            if classProbDict[pclass][cabin] > 0 and not df[(df['CabLett'].isnull()) & (df['Pclass']==pclass)].empty:
                lSub = df[(df['CabLett'].isnull()) & (df['Pclass']==pclass)].sample(int(np.round(classProbDict[pclass][cabin]*tot))).index
                print("class {0}\tCabin {1}\tSelected {2} at rate {3}".format(pclass, cabin, len(lSub),classProbDict[pclass][cabin]))
                df.loc[lSub,'CabLett'] = cabin 
    #print(df.pivot_table(index='CabLett', columns='Pclass',aggfunc='count')['PassengerId'])
    df.drop('Cabin',axis=1,inplace=True)
    return df 

def intifyDF(df):
    '''
    change all text variables to int classes
    TODO: bin continuous variables like age and fare? unsure of whether this is good or bad   
    '''
    for column in ['Title','CabLett','Embarked','Sex']:
        for i,val in enumerate(df[column].unique()):
            df.loc[df[df[column] == val].index,column] = i
    # convert embarked and sex to factors
    # mapEmbarked = {val:count for count,val in enumerate(df.Embarked.unique())}
    # df.loc[df.index, 'Embarked'] = df['Embarked'].map(mapEmbarked)
    # mapSex = {val:count for count,val in enumerate(df.Sex.unique())}
    # df.loc[df.index,'Sex'] = df['Sex'].map(mapSex)
    # drop non-numeric columns
    df.drop(['Ticket','Name','surname'],axis=1,inplace=True)
    return df

def prepDF(df):
    # df = pd.DataFrame.from_csv(csvIn,header=0, index_col=0)   
    showMissData(df)
    # resolve columns that have missing data: 
    # 1: Split title into a new column
    df = setTitle(df)  
    # 2: Age missing from 19% of rows in Training Set ; fix it?
    df = fixMissAge(df)
    # 3: Embarked missing from 2 entires; where could they be from? 
    #sns.boxplot(y='Age', x='Pclass', hue='Embarked', data=train )
    #train.groupby("Embarked").apply(len)
    # looks like S is the most common departure site, which agrees with the distribution of Age/pclass
    df.loc[df[df["Embarked"].isnull()].index,'Embarked'] = "S"     
    df = setFamSize(df)
    df = fixMissingCabin(df)
    return intifyDF(df) 

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
    train = pd.DataFrame.from_csv('train.csv')
    test  = pd.DataFrame.from_csv('test.csv')
    df = prepDF(train.append(test))
    # what does the correlation between variables look like?
    # sbs.heatmap(df.corr())
    # plt.show() 
    # Mostly shows obvious/expected correlations
    #    better class correlated with high ticket price (0.5)
    #    better class correlated with survival (0.3) 
    #    sex correlated with survival (0.5)
    #    correlations between age, siblings, and parents likely stem from families   
    
    # Given this method, how accurate should we expect to be? 
    avgErr, stdvErr, n  = calcError(df.dropna(subset=["Survived"]), 50)
    print("calculated an average of {}% error with stdev {}% over {} iterations".format(avgErr, stdvErr,n ))
    # at this point, getting between 17% and 19% of the answers wrong
    
    # the real test dataset
    #forest = RandomForestClassifier(n_estimators=100)
    # forest.fit(train.drop('Survived',axis=1), train.Survived)
    
if __name__ == "__main__": 
    main()