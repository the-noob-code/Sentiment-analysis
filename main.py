from cleanup import *
from naive_bayes_classifier import *
from stats import *

if __name__ == "__main__":
    df1 = pd.read_csv("Train.csv", dtype = 'string')
    df1_clean = cleanup(df1)
    train_data = train(df1_clean)
    df2 = pd.read_csv("Test.csv", dtype = 'string')
    df2_clean = cleanup(df2)
    result = test(df2_clean, train_data[0], train_data[1], train_data[2], train_data[3], train_data[4])
    df3 = pd.read_csv("Test.csv")
    df3['Calculated']=result
    df3.to_csv('op.csv')
    print("Generating output file op.csv ...")
    stats=contigency_matrix(df2['label'].values,result)
    print("=========Contigency matrix=========")
    print("\t        gold positive | gold negative ")
    print("system positive    ",stats[0],"       |      ",stats[1],"\nsystem negative    ",stats[2],"       |      ",stats[3])
    print("\nAccuracy=",stats[6],"\nPrecision=",stats[4],"\nRecall=",stats[5],"\nF1 score=",stats[7])
