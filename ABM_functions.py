import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


########## PREPERATION ##########
# defining agents as abstact class
class Agent:
    # init-method, the constructor method for agents
    # maybe we dont need the position and index parameters? these will be given in the populate_company function
    def __init__(self, position, index, gender, age, senority):
        self.position = position
        self.index = index
        self.gender = gender
        self.age = age
        self.senority = senority


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
            company[i][j] = Agent(position = i, index = j, gender = random.choices(['male', 'female'], weights = weights[i]['weights'], k = 1), age = weights[i]['age'], senority = weights[i]['senority'])


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
    fig, axs = plt.subplots(nrows = 1, ncols = 4, figsize=(12,5))
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


def promote_agent(company:dict, i, j, ind_i, weight:dict):
    '''
    Promotes an agent if the empty position is not at the lowest level of the company. It the empty position is at the lowest level of the company, a new agent is generated and hired. 

    Parameters
    ----------
    company : dictionary, 
    i : the level at which a position is available
    j : the index at which a position is available at level i
    ind_i : the index of the level at which a position is available
    weight : dictionary, a dictionary containing the information used to generate agents (used to generate new agents at he lowest level)
    '''
    # if level is the lowest, a new agent is created
    if i == list(company.keys())[-1]:
        company[i][j] = Agent(position = i, index = j, gender = random.choices(['male', 'female'], weights = weight[i]['weights'], k = 1), age = weight[i]['age'], senority = weight[i]['senority'])
    
    elif i != list(company.keys())[-1]:
        weights = []
        # determining weight from the attributes of agents on the level lower that the empty spot
        # the weight is determined by the age and the senority of the agent
        # the weight is normalized by the number of agents on the level lower that the empty spot
        index = get_nth_key(company, (ind_i+1))
        for k in range(0, len(company[index])):
            if company[index][k] is not None:
                weights.append((company[index][k].age + company[index][k].senority)/(len(company[index])))
            else:
                weights.append(0)
        # the agent with the highest weight is promoted (could be changed to random choice?)
        agent = company[index][np.argmax(weights)]


        #else:
        #    agent = None
        #    while agent == None: # REMEMBER: This runs infinetly if all agents are None in the given category... FIX THIS
        #        agent = random.choice(company[list(company.keys())[ind_i+1]])

        # promoting the agent
        company[i][j] = agent
        # removing the agent from the lower level of the company
        company[agent.position][agent.index] = None
        # changing the position of the agent
        company[i][j].postition = i
            # changing the index of the agent
        company[i][j].index = j


def fire_agent(company, i):
    if i == 'Department Head' and len(company[i]) > 19:
        company[i][random.randint(0, len(company[i])-1)] = None
        
    # fire 1 random leader if there are more than 39 leaders
    if i == 'Leader' and len(company[i]) > 39:
        company[i][random.randint(0, len(company[i])-1)] = None
    #fire 1 random senior if there are more than 69 senior
    if i == 'Senior' and len(company[i]) > 69:
        company[i][random.randint(0, len(company[i])-1)] = None

    # fire 1 random junior if there are more than 128 juniors
    if i == 'Junior' and len(company[i]) > 128:
        company[i][random.randint(0, len(company[i])-1)] = None


def update_agents(company, i, j):
    '''
    Updates the agents each tick (e.g. adding a month to senority and age)
    '''
    # adding a month to the senority of all agents
    company[i][j].senority += 1/12
    # adding a month to the age of all agents
    company[i][j].age += 1/12
    # if the agent has reached the age of 68, the agent is retired
    if company[i][j].age >= 68:
        company[i][j] = None

    # REMEMBER: maybe we should retain information about the previous agent to hire someone similar to? eg. instead of setting the value in the dictionary to None we could set it to gender of the old agent? maybe also include age?
        # but we would have to change all the places where it says if company.. == None



########## SIMULATION ##########

# NOTES
# check savepath in beginning of simulation
# figure out what data to include

def run_abm(months: int, save_path: str, company_titles: list, titles_n: list, weights: dict, plot_each_tick = False):
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
                    update_agents(company, i, j)

                if company[i][j] == None:
                    promote_agent(company, i, j, ind_i, weight=weights)

            ### WE NEED TO FIND ANOTHER WAY FOR AGENTS TO QUIT OR GET FIRED ###    
            # fire some random agents
            fire_agent(company, i=i)
            


        # plotting and appending data to data frame                           
        if plot_each_tick:
            plot_gender(company, tick = month)

        f = {'gender': 'female', 'tick': month, 'Department Head': count_gender(company)[0][0],'Leader': count_gender(company)[0][1], 'Senior': count_gender(company)[0][2], 'Junior': count_gender(company)[0][3]}
        m = {'gender': 'male', 'tick': month, 'Department Head': count_gender(company)[1][0],'Leader': count_gender(company)[1][1], 'Senior': count_gender(company)[1][2], 'Junior': count_gender(company)[1][3]}
        
        # create pandas dataframe from dictionaries f and m
        new_data = pd.DataFrame.from_dict([f, m])
        data = data.append(new_data, ignore_index=False, verify_integrity=False, sort=False)

        print('tick {} done'.format(month))

    # plotting the gender development over time
    plot_gender_development(company, months=months, data = data)
    # saving the data to a csv-file
    data.to_csv(save_path)


                