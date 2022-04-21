import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

########## PREPERATION ##########
# defining agents as abstact class
class Agent:
    # init-method, the constructor method for agents
    # maybe we dont need the position and index parameters? these will be given in the populate_company function
    def __init__(self, position, index, gender, age, seniority, fire, seniority_position):
        self.position = position
        self.index = index
        self.gender = gender
        self.age = age
        self.seniority = seniority
        self.fire = fire[0] # could potentially be dependent on the sex and the seniority and the position?
        self.seniority_position = seniority_position


# function for creating empty dictionary (company)
def create_company(titles:list, n:list):
    '''
    Creates a dictionary with the given titles as keys and empty lists as values.

    Parameters
    ----------
    titles : list, a list of strings with job titles
    n : list, a list of integers with the number of agents in each job title
    '''
    company = {}
    for i in range(len(titles)):
        key = titles[i] 
        value = [None for i in range(0, n[i])]
        company[key] = value

    return company


# function for creating the first agents and putting them in the company
def populate_company(company: dict, weights: dict):
    '''
    Creates the first agents and puts them in the company dictionary.

    Parameters
    ----------
    company : dictionary, a dictionary with the job titles as keys and empty lists as values
    weights: dictionary, a dictionary containing information on how to generate the attributes of the agents in each jobtitle
    '''
    for i in company.keys():
        for j in range(0, len(company[i])):
            company[i][j] = Agent(position = i, index = j, gender = random.choices(['male', 'female'], weights = weights[i]['weights'], k = 1), age = random.gauss(weights[i]['age'][0], weights[i]['age'][1]), seniority = random.gauss(weights[i]['seniority'][0], weights[i]['seniority'][1]), fire = random.choices([True, False], weights = weights[i]['fire']), seniority_position = random.gauss(weights[i]['seniority_position'][0], weights[i]['seniority_position'][1]))


# function for counting gender of agents in the different hierchical levels of the company
def count_gender(company):
    '''
    Counts the number of males and females of each job-title in the company

    Parameters
    ---------- 
    company : dictionary, a dictionary with the job titles as keys and empty lists as values
    '''
    female_count = np.repeat(0, len(company.keys()))
    male_count = np.repeat(0, len(company.keys()))

    for index, i in enumerate(company.keys()):
        for j in range(0, len(company[i])):
            if company[i][j] is not None:
                if company[i][j].gender == ['female']:
                    female_count[index] += 1
                else:
                    male_count[index] += 1
    return female_count, male_count


########## PLOTTING ##########
# function for plotting the gender distribution in the company
def plot_gender(company, tick):
    '''
    Plots the current gender distribution in each job-title of the company

    Parameters
    ---------- 
    company : dictionary, a dictionary with the job titles as keys and lists of agents as values
    '''
    female, male = count_gender(company)
    
    labels = company.keys()
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(9,6))
    rects1 = ax.bar(x - width/2, female, width, label='Female', color = 'skyblue')
    rects2 = ax.bar(x + width/2, male, width, label='Male', color = 'steelblue')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_title('Count by Job-title and Gender, month = {}'.format(tick + 1))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()

    return ax

# function for plotting the gender distribution development over all ticks
def plot_gender_development(company: dict, months:int, data: pd.DataFrame):
    '''
    Plots the counts of each gender in each job-title in the company for each tick

    Parameters
    ----------
    company : dictionary, a dictionary with the job titles as keys and lists of agents as values
    months: int, the number of months simulated
    data: pandas dataframe, a dataframe with the information about the genders of employees in each job-title
    '''
    fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize=(12,5), sharey=False)
    colorlist = {'female': 'skyblue', 'male': 'steelblue'}
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colorlist.values()]
    plt.legend(markers, colorlist.keys(), numpoints=1) # legend
    fig.suptitle('Counts of each gender in each position', fontsize=16) # title
    fig.text(0.5, 0.04, 'Month', ha='center', va='center') # x-axis label 
    fig.text(0.07, 0.5, 'Count', ha='center', va='center', rotation='vertical') # y-axis label

    for index, ax in enumerate(axs.flatten()):
        jobtitle = list(company.keys())[index]
        #ax.set_xlabel('Month')
        #ax.set_ylabel('Count')
        ax.set_title(jobtitle) # setting title to the job-title

        for gender in ['female', 'male']:
            gender_dat = data.loc[data['gender'] == gender]
            dat = gender_dat[jobtitle]
            ticks = gender_dat['tick']
            ax.plot(ticks, dat, color = colorlist[gender])

# get nth_key
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

def promote_agent(company:dict, i, j, ind_i, weight:dict, bias:float):
    '''
    Promotes an agent if the empty position is not at the lowest level of the company. It the empty position is at the lowest level of the company, a new agent is generated and hired. 

    Parameters
    ----------
    company : dictionary, 
    i : the level at which a position is available
    j : the index at which a position is available at level i
    ind_i : the index of the level at which a position is available
    weight : dictionary, a dictionary containing the information used to generate agents (used to generate new agents at he lowest level)
    bias : float, a bias added to all males to either increase or decrease their chance of getting promoted
    '''
    # if level is the lowest, a new agent is created
    if i == list(company.keys())[-1]:
        company[i][j] = Agent(position = i, index = j, gender = random.choices(['male', 'female'], weights = weight[i]['weights'], k = 1), age = random.gauss(weight[i]['age'][0], weight[i]['age'][0]), seniority = random.gauss(weight[i]['seniority'][0], weight[i]['seniority'][1]), fire = weight[i]['fire'], seniority_position = 0) # SHOULD WE PUT IN A RANDOM ONE MAYBE THEY HAD A SIMILAR POSITION
    
    elif i != list(company.keys())[-1]:
        weights = []
        # determining weight from the attributes of agents on the level lower that the empty spot
        # the weight is determined by the age and the seniority of the agent and normalized by the number of agents on the level lower that the empty spot
        index = get_nth_key(company, (ind_i+1))
        for k in range(0, len(company[index])):
            if company[index][k] is not None:
                if company[index][k].gender[0] == 'male':
                    add_bias = bias
                else:
                    add_bias = 0
                weights.append((company[index][k].age + company[index][k].seniority)/(len(company[index])) + add_bias)
            else:
                weights.append(None)
        #np.argsort(test_list, axis=-1)[-candidates:] -weight[index]['candidates']
        promotion_candidates_index = np.argsort(np.where(~np.isin(weights, [None]), weights, -1e10))[-weight[i]['candidates']:]

        #normalizing weights THIS TAKES A LONG TIME... how can we speed this up?
        weights = np.array(weights)[promotion_candidates_index]
        weights = normalize_weights(weights)

        # an agent from the promotion candidates is promoted using the random choice function, with weights calculated - CHECK THAT THIS WORKS AS INTENDED
        agent = company[index][random.choices(promotion_candidates_index, weights = weights)[0]]

        # setting the seniority in the position as 0
        agent.seniority_position = 0

        # promoting the agent
        company[i][j] = agent
        # removing the agent from the lower level of the company
        company[agent.position][agent.index] = None
        # changing the position of the agent
        company[i][j].postition = i
        # changing the index of the agent
        company[i][j].index = j


def fire_agent(company, i, j):
    '''
    If the fire attribute of an agent is True, the agent is fired.
    '''
    if company[i][j] is not None:
        if company[i][j].fire[0] == True:
            company[i][j] = None


def update_agents(company, i, j, weight:dict):
    '''
    Updates the agents each tick (e.g. adding a month to seniority and age)

    Parameters
    ----------
    company : dictionary, 
    i : the job-title of the agent
    j : the index of the agent at the job-title
    weight : dictionary, a dictionary containing the information used to generate agents
    '''
    # adding a month to the seniority of all agents
    company[i][j].seniority += 1/12
    # adding a month to the position seniority of all agents
    company[i][j].seniority_position += 1/12
    # adding a month to the age of all agents
    company[i][j].age += 1/12
    company[i][j].fire = random.choices([True, False], weights = weight[i]['fire'], k = 1)
    # if the agent has reached the age of 68, the agent is retired
    if company[i][j].age >= 68:
        company[i][j] = None



########## SIMULATION ##########

# NOTES
# check savepath in beginning of simulation
# figure out what data to include

def run_abm(months: int, save_path: str, company_titles: list, titles_n: list, weights: dict, bias: float, plot_each_tick = False):
    '''
    Runs the ABM simulation

    Parameters
    ----------
    months : int, the number of months to simulate
    save_path : str, the path to the csv file
    company_titles : list, a list of strings with the job titles
    titles_n : list, a list of integers with the number of agents in each job title
    plot_each_tick : bool, if True, the gender distribution is plotted after each tick
    weights : dictionary, a dictionary containing the information used to generate agents
    bias : float, a bias for the company's hiring process (positive bias = males are more likely to be promoted)
    '''
    # creating empty dataframe for the results
    col_names = ['tick', 'gender']
    col_names.extend(company_titles)
    data = pd.DataFrame(columns = col_names)
    
    # create company
    company = create_company(company_titles, titles_n)
    
    # populate company using weights
    populate_company(company, weights)

    # plot initial
    if plot_each_tick:
        plot_gender(company, tick=-1)

    for month in range(months):
        # iterating through all agents
        for ind_i, i in enumerate(company.keys()):
            for j in range(0, len(company[i])):
                if company[i][j] is not None:
                    update_agents(company, i, j, weights)
                    fire_agent(company, i, j)

                if company[i][j] == None:
                    promote_agent(company, i, j, ind_i, weight=weights, bias=bias)
            
        # plotting and appending data to data frame                           
        if plot_each_tick:
            plot_gender(company, tick = month)

        f = {'gender': 'female', 'tick': month, 'C-Suite': count_gender(company)[0][0],'SVP': count_gender(company)[0][1], 'VP': count_gender(company)[0][2], 'Senior Manager': count_gender(company)[0][3], 'Manager': count_gender(company)[0][4], 'Entry Level': count_gender(company)[0][5]}
        m = {'gender': 'male', 'tick': month, 'C-Suite': count_gender(company)[1][0],'SVP': count_gender(company)[1][1], 'VP': count_gender(company)[1][2], 'Senior Manager': count_gender(company)[1][3], 'Manager': count_gender(company)[1][4], 'Entry Level': count_gender(company)[1][5] }
        
        # create pandas dataframe from dictionaries f and m
        new_data = pd.DataFrame.from_dict([f, m])
        data = data.append(new_data, ignore_index=False, verify_integrity=False, sort=False)

        print('tick {} done'.format(month))

    # plotting the gender development over time
    plot_gender_development(company, months = months, data = data)
    # saving the data to a csv-file
    data.to_csv(save_path)


                