import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def dataloader(url):
    print("Loading Data")
    dataset = pd.read_pickle(url)
    labels = dataset.iloc[:, 16:]
    dataset = dataset.iloc[:, 4:16]

    _, weightindex = weight_metric()
    labels = labels[weightindex].to_numpy() #[:27]
    del weightindex
    gc.collect()


    print("Test Train Split 1")
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)
    del dataset
    del labels


    # train-validation-test split = 60/20/20
    print("Test Train Split 2")
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    print("Loading Leads")
    X_train = lead_loader(X_train)
    X_val = lead_loader(X_val)
    X_test = lead_loader(X_test)

    print("Delete Missing Rows")
    X_train, y_train = delete_missing_rows(X_train, y_train)
    X_val, y_val = delete_missing_rows(X_val, y_val)
    X_test, y_test = delete_missing_rows(X_test, y_test)

    gc.collect()

    return X_train, X_test, y_train, y_test, X_val, y_val

# Convert ECG leads from Dataframe to NumPy array (used to convert to tensor later)
def lead_loader(dataset):

    dataset = dataset.to_numpy()
    print("Dataset shape: ", dataset.shape)
    testingleads = np.zeros([dataset.shape[0], dataset.shape[1], 5000])
    
    for i in range(0,12):
        for j in range(0, dataset.shape[0]):
            testingleads[j, i, :] = dataset[j, i]

    del dataset
    gc.collect()

    return testingleads

# Loading weight metric from challenge to use the 27 classes
def weight_metric():
    weights = pd.read_csv("../Data/weights.csv", index_col=0)
    ctcodes = pd.read_csv("../Data/Dx_map.csv")
    ctcodes = ctcodes.iloc[:, 1:]
    replacedict = dict(zip(ctcodes.iloc[:,0], ctcodes.iloc[:,1]))

    weights.columns = weights.columns.astype(int)
    weights.rename(columns=replacedict, index=replacedict, inplace=True)
    weightindex = np.array(weights.index)
    return weights, weightindex


# Delete rows where labels are missing
def delete_missing_rows(leads, labels):
    print("Deleting Rows/n")
    indexvals=[]
    for i in range(0, labels.shape[0]):
        if len(np.unique(labels[i,:], axis=0)) == 2:
            indexvals.append(i)
    leads = np.take(leads, indexvals, axis=0)
    labels = np.take(labels, indexvals, axis=0)

    del indexvals
    gc.collect()

    return leads, labels


# trainingleads, testingleads, traininglabels, testinglabels, validationleads, validationlabels = dataloader("../Data/fullecgdata.pkl")


        