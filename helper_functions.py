import pickle
import pandas as pd

def create_dataframe(col_names, company_titles):
    col_names.extend(company_titles)
    data = pd.DataFrame(columns = col_names)

    return data

def bias(a, b):
    return 0.5 - (b/(b+a))


def normalize_weights(list):
    '''
    Normalizes a list between 0 and 1
    '''
    try:
        list = [float((i - min_none(list))) / float((max_none(list) - min_none(list))) if i != None else 0 for i in list]
    except(ZeroDivisionError):
        list = None # if the denominator is 0 the weights are equal
    return list


def min_none(list):
    '''
    Returns the minimum value of a list, ignores None values
    '''
    newlist = min([value for value in list if value != None])
    return newlist


def max_none(list):
    '''
    Returns the maximum value of a list, ignores None values
    '''
    newlist = max([value for value in list if value is not None])
    return newlist


def save_dict(dict, filepath = 'data/company.pkl'):
    '''
    Saves a dictionary to a file
    '''
    a_file = open(filepath, "wb")
    pickle.dump(dict, a_file)
    a_file.close()


def open_dict(filepath = 'data/company.pkl'):
    '''
    Opens a dictionary from a file
    '''
    a_file = open(filepath, "rb")
    dict = pickle.load(a_file)
    a_file.close()
    return dict


def get_nth_key(dictionary: dict, n=0):
    '''
    Gets the nth key of a dictionary

    Parameters
    ----------
    dictionary: dictionary
    n: the nth key to get

    '''
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range")
