import numpy as np 
from sklearn.model_selection import train_test_split as sk_split
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from time import time 

def random_split(data, ratios, seed=42):
    if isinstance(ratios, float):
        return sk_split(data, train_size=ratios, seed=seed)
    elif isinstance(ratios, list):
        indices = [
            round(len(data) * ratio) for ratio in np.cumsum(ratios)[:-1]
        ]  #:x, x:y, y:
        shuffled_data = data.sample(frac=1, random_state=seed)
        return np.split(shuffled_data, indices, axis=0)



def deprecated_stratify_split(data, ratios, split_column = 'userId', seed = 42):

    #Deja on ordonne le dataset et on le 

    #TODO use chronological and random

    groups = data.groupby(split_column)

    splits = [[] for _ in range(len(ratios))]

    for _, group in groups: #Tres mauvais de travailler avec des boucles !!! Optimiser a la microsoft
        group_split = random_split(group, ratios, seed) #Pas bon si on veut conserver la chronologie
        for i in range(len(ratios)):
            splits[i].append(group_split[i])
    
    return (pd.concat(splits[i]) for i in range(len(ratios)))


def stratify_split(data, ratios, split_column = 'userId', seed = 42):

    groups = data.groupby(split_column)
    data["count"] = groups[split_column].transform("count")
    data["rank"] = groups.cumcount() + 1

    splits = []
    prev_threshold = None
    for threshold in np.cumsum(ratios):
        condition = data["rank"] <= round(threshold * data["count"])
        if prev_threshold is not None:
            condition &= data["rank"] > round(prev_threshold * data["count"])
        splits.append(data[condition].drop(["rank", "count"], axis=1))
        prev_threshold = threshold

    return splits
    
from datetime import datetime, timedelta

def timestamp_to_ISO(timestamp):
    return datetime.strftime(datetime(1970, 1, 1, 0, 0, 0) + timedelta(seconds=timestamp), "%Y-%m-%d %H:%M:%S")

if __name__ == '__main__':


    path = "../datasets/ml-latest-small/ratings.csv"
    src_name = "userId"
    dst_name = "movieId"
    edge_name = "rating"
    edge_threshold = 3.5
    timestamp_name = "timestamp"


    data = pd.read_csv(path)
    

    data[timestamp_name] = data.apply(lambda x : timestamp_to_ISO(x[timestamp_name]), axis = 1)


    #Etape de pre processing deja 

    data = data.drop_duplicates() #Supprime les duplicates 
    
    #On doit aussi retravailler sur les indexes 

    n_users = len(data[src_name].unique())
    n_movies = len(data[dst_name].unique())

    users_encoder = LabelEncoder()
    movies_encoder = LabelEncoder()

    data[src_name] = users_encoder.fit_transform(data[src_name])
    data[dst_name] = movies_encoder.fit_transform(data[dst_name])

    #Split 
    ratios = [1.,0.,0.] #Train/Val/Test 
    train_data, val_data, test_data = stratify_split(data, ratios, split_column=src_name)

    print(val_data)








    


