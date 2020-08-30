from utils import User
import csv

def scrapeTrain(textFile, label, output_csv):
    '''
    Example function for generating CSV files for a folder; you shouldn't 
    need to use this unless you are attempting to retrain the network with
    different datapoints. The textFile argument should be the directory that
    contains text files you want to classify, and the label should be either
    1 or 0 (bot or human).

    Example usage: scrapeTrain("Train/humans.txt", 0, "Train/datas.csv")
    '''
    all_users = list(set(open(textFile).read().split('\n')))
    all_data = []

    for user in all_users:
        print(user)
        try:
            all_data.append([label] + User(user).data[1:])
        except Exception as e:
            print(str(e))
            continue 

    with open(output_csv, 'a') as f:
        writer = csv.writer(f)
        writer.writerows(all_data) 

