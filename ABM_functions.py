import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helper_functions import *
from scipy.stats import truncnorm

########## PREPERATION ##########
# defining agents as abstact class
class Agent:
    # init-method, the constructor method for agents
    # maybe we dont need the position and index parameters? these will be given in the populate_company function
    def __init__(self, position, index, gender, age, seniority, fire, seniority_position, id):
        self.position = position
        self.index = index
        self.gender = gender
        self.age = age
        self.seniority = seniority
        self.fire = fire[0] # could potentially be dependent on the sex and the seniority and the position?
        self.seniority_position = seniority_position
        self.parental_leave = None # DO SOMETHING HERE both seniority should be paused when on parental leave
        self.id = id


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

def get_truncated_normal(mean=0, sd=1, low=20, upp=68):
    x = truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    return x.rvs(1)[0]

# function for creating the first agents and putting them in the company
def populate_company(company: dict, weights: dict):
    '''
    Creates the first agents and puts them in the company dictionary.

    Parameters
    ----------
    company : dictionary, a dictionary with the job titles as keys and empty lists as values
    weights: dictionary, a dictionary containing information on how to generate the attributes of the agents in each jobtitle
    '''
    id = 1
    for i in company.keys():
        for j in range(0, len(company[i])):
            # id
            id += 1
            company[i][j] = Agent(position = i, index = j, gender = random.choices(['male', 'female'], weights = weights[i]['weights'], k = 1), age = get_truncated_normal(mean = weights[i]['age'][0], sd = weights[i]['age'][1]), seniority = random.gauss(weights[i]['seniority'][0], weights[i]['seniority'][1]), fire = random.choices([True, False], weights = weights[i]['fire']), seniority_position = random.gauss(weights[i]['seniority_position'][0], weights[i]['seniority_position'][1]), id = id)
    return id


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

def mean_seniority(company):
    '''
    Calculates the mean seniority of the company
    '''
    means = []
    for i in (company.keys()):
        seniority_list = [i.seniority for i in company[i] if i is not None]
        seniority_pos_list = [i.seniority_position for i in company[i] if i is not None]

        zipped_list = list(zip(seniority_list, seniority_pos_list))
        zipped_list = [x+y for (x, y) in zipped_list]
        means.append(np.mean(zipped_list))
    
    return means


########## PLOTTING ##########
# function for plotting the gender distribution in the company
def plot_gender(company, tick):
    '''
    Plots the current gender distribution in each job-title of the company

    Parameters
    ---------- 
    company : dictionary, a dictionary with the job titles as keys and lists of agents as values
    tick : 
    '''
    female, male = count_gender(company)
    
    labels = company.keys()
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(9,6))
    rects1 = ax.bar(x - width/2, female, width, label = 'Female', color = 'skyblue')
    rects2 = ax.bar(x + width/2, male, width, label = 'Male', color = 'steelblue')

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


def promote_agent(company:dict, i, j, ind_i, weight:dict, bias: list, threshold: list, diversity_bias_scaler: float, id, intervention, mean_senior):

    '''
    Promotes an agent if the empty position is not at the lowest level of the company. It the empty position is at the lowest level of the company, a new agent is generated and hired. 

    Parameters
    ----------
    company : dictionary, 
    i : the level at which a position is available
    j : the index at which a position is available at level i
    ind_i : the index of the level at which a position is available
    weight : dictionary, a dictionary containing the information used to generate agents (used to generate new agents at he lowest level)
    bias : list,
    threshold : list, the threshold for diversity
    '''
    gender_distribution = count_gender(company)
    diversity_bias = diversity_check(gender_distribution, ind_i, threshold, diversity_bias_scaler)
    choose_agent_promotion(company, i, j, ind_i, weight, bias, diversity_bias, id, intervention, mean_senior)

def choose_agent_promotion(company, i, j, ind_i, weight, bias, diversity_bias, id, intervention, mean_senior):
    '''
    Chooses an agent to promote

    Parameters
    ----------
    company : dictionary,
    i : the level at which a position is available
    j : the index at which a position is available at level i
    ind_i : the index of the level at which a position is available
    weight : dictionary, a dictionary containing the information used to generate agents (used to generate new agents at he lowest level)
    bias : a list of biases calculated from the gender distribution at each level OR the diversity bias, added if the number of women is below threshold
    '''
    # if level is the lowest, a new agent is created
    if i == list(company.keys())[-1]:
        company[i][j] = Agent(position = i, index = j, gender = random.choices(['male', 'female'], weights = weight[i]['weights'], k = 1), age = random.gauss(weight[i]['age'][0], weight[i]['age'][0]), seniority = random.gauss(weight[i]['seniority'][0], weight[i]['seniority'][1]), fire = weight[i]['fire'], seniority_position = 0, id = id) # SHOULD WE PUT IN A RANDOM ONE MAYBE THEY HAD A SIMILAR POSITION
  
    elif i != list(company.keys())[-1]:
        weights = []
        
        # determining weight from the attributes of agents on the level lower that the empty spot
        # the weight is determined by the age and the seniority of the agent and normalized by the number of agents on the level lower that the empty spot
        index = get_nth_key(company, (ind_i+1))
        for k in range(0, len(company[index])):
            if company[index][k] is not None:
                if company[index][k].gender[0] == 'male' and intervention != 'blinding':
                    add_bias = bias[ind_i] + diversity_bias # adding bias of the position that needs to be filled
                else:
                    add_bias = 0
                weights.append((company[index][k].seniority_position + company[index][k].seniority)/(mean_senior[ind_i+1]) + add_bias)

            else:
                weights.append(None)
            

        
        promotion_candidates_index = np.argsort(np.where(~np.isin(weights, [None]), weights, -1e10))[-weight[i]['candidates']:]
        weights = np.array(weights)[promotion_candidates_index]

        if intervention == 'blinding':
            # adding bias to the weights after finding candidates for the position
            for ind_we, k in enumerate(promotion_candidates_index):
                if company[index][k].gender == 'male':
                    weights[ind_we] = weights[ind_we] + bias[ind_i] + diversity_bias 

        #normalizing weights
        weights = normalize_weights(weights)

        # an agent from the promotion candidates is promoted using the random choice function, with weights calculated - CHECK THAT THIS WORKS AS INTENDED
        agent = company[index][random.choices(promotion_candidates_index, weights = weights)[0]]

        # setting the seniority in the position as 0
        agent.seniority_position = 0
        # removing the agent from the lower level of the company
        company[agent.position][agent.index] = None
        agent.position = i
        # promoting the agent
        company[i][j] = agent
        # changing the index of the agent
        company[i][j].index = j


def fire_agent(company, i, j):
    '''
    If the fire attribute of an agent is True, the agent is fired.
    '''
    if company[i][j] is not None:
        if company[i][j].fire[0] == True:
            # ADD CONDitionS EG NOT ALLOWED TO FIRE AGENTS ON PARENTAL LEAVE
            company[i][j] = None


def update_agents(company, i, j, weight:dict, months_pl, intervention):
    '''
    Updates the agents each tick (e.g. adding a month to seniority and age). This function also keeps track of parental leave. 

    Parameters
    ----------
    company : dictionary, 
    i : the job-title of the agent
    j : the index of the agent at the job-title
    weight : dictionary, a dictionary containing the information used to generate agents
    months_pl : int, the number of months of parental leave
    '''
    if company[i][j].parental_leave == None:
        # adding a month to the seniority of all agents not on parental leave
        company[i][j].seniority += 1/12
        # adding a month to the position seniority of all agents not on parental leave
        company[i][j].seniority_position += 1/12
    
    # adding a month to the age of all agents
    company[i][j].age += 1/12
    if intervention == None:
        if company[i][j].gender[0] == 'female':
            update_parental_leave(company, i, j, months_pl)
    if intervention == 'shared_parental':
        update_parental_leave(company, i, j, months_pl)
    
    # fire some agents ADD CONDITIONS e.g., not allowed to fire agents on parental leave
    company[i][j].fire = random.choices([True, False], weights = weight[i]['fire'], k = 1)
    
    # if the agent has reached the age of 68, the agent is retired
    if company[i][j].age >= 68:
        company[i][j] = None


def update_parental_leave(company: dict, i, j, months_pl):
    '''

    Parameters
    ----------
    company : dictionary
    i : the job-title of the agent
    j : the index of the agent at the job-title
    '''

    if company[i][j].parental_leave != None:
        if company[i][j].parental_leave == months_pl:
            company[i][j].parental_leave = None # when the agent has been on parental leave for the time specified, the parental leave is ended
        else:
            company[i][j].parental_leave += 1 # adding one to the parental leave

    elif company[i][j].parental_leave == None:
        parental_leave(company, i, j)

def parental_leave(company, i, j):
    '''
    A function that determines whether an agent should begin parental leave or not

    Parameters
    ----------
    agent : 
    '''
    if company[i][j].age >= 20 and company[i][j].age <= 24:
            leave = random.choices([1, None], weights = [0.072/(12*5), 0.9988])
            company[i][j].parental_leave = leave[0]
    elif company[i][j].age >= 25 and company[i][j].age <= 29:
            leave = random.choices([1, None], weights = [0.0981/(12*5), 0.998365])
            company[i][j].parental_leave = leave[0]
    elif company[i][j].age >= 30 and company[i][j].age <= 34:
            leave = random.choices([1, None], weights = [0.1005/(12*5), 0.998325])
            company[i][j].parental_leave = leave[0]
    elif company[i][j].age >= 35 and company[i][j].age <= 39:
            leave = random.choices([1, None], weights = [0.0524/(12*5), 0.9988])
            company[i][j].parental_leave = leave[0]
    elif company[i][j].age >= 40 and company[i][j].age <= 45:
            leave = random.choices([1, None], weights = [0.0126/(12*5), 0.9997])
            company[i][j].parental_leave = leave[0]
    else:
        company[i][j].parental_leave = None
    



def mean_age(company: dict, i):
    '''
    company : dictionary,
    i : the job title to calculate the mean age of
    '''
    ages_m = []
    ages_f = []
    for k in range(0, len(company[i])):
        if company[i][k] is not None:
            if company[i][k].gender[0] == 'male':
                ages_m.append(company[i][k].age)
            else:
                ages_f.append(company[i][k].age)

    return np.mean(ages_m), np.mean(ages_f)


def diversity_check(gender_distribution, ind_i, threshold, diversity_bias_scaler):
    '''
    Checks if the gender distribution is skewed above a certain threshold. Returns True if the distribution is above threshold (e.g., if the threshold is 30% and there is more than 30% women)

    Parameters
    ----------
    gender_distribution : the output from the count_gender function
    i : the company level at which the diversity is checked
    threshold : the threshold for 
    '''
    percent_women = (gender_distribution[0][ind_i]/(gender_distribution[0][ind_i] + gender_distribution[1][ind_i]))

    if percent_women > threshold:
        diversity_bias = 0

    else:
        diversity_bias = (percent_women - threshold) * diversity_bias_scaler

    return diversity_bias


def get_bias(company, scale = 1):
    '''
    Determines the bias for the promotion of agents. The bias is determined by the gender distribution of the level at which there is an empty position. 
    parameters
    ----------
    company : dictionary
    scale : float, the scale of the bias
    '''
    gender_distribution = count_gender(company)
    csuite_bias = bias(gender_distribution[1][0], gender_distribution[0][0]) * scale
    svp_bias = bias(gender_distribution[1][1], gender_distribution[0][1]) * scale
    vp_bias = bias(gender_distribution[1][2], gender_distribution[0][2]) * scale
    senior_manager_bias = bias(gender_distribution[1][3], gender_distribution[0][3]) * scale
    manager_bias = bias(gender_distribution[1][4], gender_distribution[0][4]) * scale
    entry_level_bias = bias(gender_distribution[1][5], gender_distribution[0][5]) * scale

    return csuite_bias, svp_bias, vp_bias, senior_manager_bias, manager_bias, entry_level_bias



########## SIMULATION ##########

def run_abm(months: int, save_path: str, company_titles: list, titles_n: list, weights: dict, bias_scaler: float = 1.0, plot_each_tick = False, months_pl: float = 9, threshold: float = 0.3, diversity_bias_scaler: float = 1.0, intervention = None, company = None):
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
    bias_scaler : float, higher number increases the influence of the bias of the gender distribution at the level at which a position is empty
    month_pl : int, the number of months women are on parental leave after giving birth
    threshold : list, the threshold for the share of women at a given level (if below threshold a positive bias is added towards women, i.e., increasing their probability of promotion)
    intervention : the type of intervention to apply
    '''
    # creating empty dataframe for the results
    data = create_dataframe(['tick', 'gender'], company_titles)
    adata = pd.DataFrame()
    
    # create company
    if company == None:
        company = create_company(company_titles, titles_n)
    
    # populate company using weights, and saving the last ID given to an agent
    id = populate_company(company, weights)

    # plot initial
    if plot_each_tick:
        plot_gender(company, tick=-1)

    # iterating though the months
    for month in range(months):
        bias = list(get_bias(company, scale = bias_scaler))
        mean_senior = mean_seniority(company)
        # iterating through all agents
        for ind_i, i in enumerate(company.keys()):
            for j in range(0, len(company[i])):
                id += 1
                if company[i][j] is not None:
                    update_agents(company, i, j, weights, months_pl, intervention)
                    fire_agent(company, i, j)

                if company[i][j] == None:
                    promote_agent(company, i, j, ind_i, weight=weights, bias = bias, threshold = threshold[ind_i], diversity_bias_scaler = diversity_bias_scaler, id = id, intervention = intervention, mean_senior = mean_senior)
           
                dat = {'id': company[i][j].id, 'gender': company[i][j].gender[0], 'age': company[i][j].age, 'seniority': company[i][j].seniority, 'seniority_pos': company[i][j].seniority_position, 'parental_leave': company[i][j].parental_leave, 'position': company[i][j].position, 'tick': month}
                adata = adata.append(dat, ignore_index=True, verify_integrity=False, sort=False)
            
        # plotting and appending data to data frame                           
        if plot_each_tick:
            plot_gender(company, tick = month)
        
        counts = count_gender(company)
        f = {'gender': 'female', 'tick': month, 'Level 1': counts[0][0],'Level 2': counts[0][1], 'Level 3': counts[0][2], 'Level 4': counts[0][3], 'Level 5': counts[0][4], 'Level 6': counts[0][5]}
        m = {'gender': 'male', 'tick': month, 'Level 1': counts[1][0],'Level 2': counts[1][1], 'Level 3': counts[1][2], 'Level 4': counts[1][3], 'Level 5': counts[1][4], 'Level 6': counts[1][5] }

        # create pandas dataframe from dictionaries f and m
        new_data = pd.DataFrame.from_dict([f, m])
        data = data.append(new_data, ignore_index=False, verify_integrity=False, sort=False)

        print('tick {} done'.format(month))

    # plotting the gender development over time
    plot_gender_development(company, months = months, data = data)
    # saving the data to a csv-file
    adata.to_csv(save_path)

    # saving the company as is at the end of the simulation
    if intervention == None:
        save_dict(company)


                