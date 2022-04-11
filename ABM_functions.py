import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyparsing import col


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
def populate_company(company: dict):
    '''
    Creates the first agents and puts them in the company dictionary.

    Parameters
    ----------
    company : dictionary, a dictionary with the job titles as keys and empty lists as values
    '''
    for i in company.keys():
        for j in range(0, len(company[i])):
            if i == 'Department Head':
                weights = [0.8, 0.2] # more likely to be male when department head
                age = random.gauss(50, 8)
                senority = random.gauss(10, 3)
            elif i == 'Leader':
                weights = [0.7, 0.3] # more likely to be male when leader
                age = random.gauss(40, 6)
                senority = random.gauss(5, 3)
            elif i == 'Senior':
                weights = [0.6, 0.4] # more likely to be male when senior
                age = random.gauss(35, 6)
                senority = random.gauss(4, 1)
            elif i == 'Junior': 
                weights = [0.5, 0.5] # equally likely to be male and female
                age = random.gauss(30, 6)
                senority = random.gauss(3, 1)

            company[i][j] = Agent(position = i, index = j, gender = random.choices(['male', 'female'], weights = weights, k = 1), age = age, senority = senority)


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


# function for plotting the gender distribution in the company
def plot_gender(company, tick):
    '''
    Plots the current gender distribution in each job-title of the company

    Parameters
    ---------- 
    company : dictionary, a dictionary with the job titles as keys and empty lists as values
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




########## SIMULATION ##########

# NOTES
# check savepath in beginning of simulation
# figure out what data to include


def run_abm(months: int, save_path: str):
    '''
    Runs the ABM simulation

    Parameters
    ----------
    months : int, the number of months to simulate
    save_path : str, the path to the csv file
    '''
    # creating empty dataframe for the results
    data = pd.DataFrame(columns = ['tick', 'gender', 'department_head', 'leader', 'senior', 'junior'])

    # create company
    company = create_company(['Department Head', 'Leader', 'Senior', 'Junior'], [10, 20, 40, 100])
    # populate company
    populate_company(company)
    # plot initial 
    plot_gender(company, tick=-1)

    for month in range(months):
        # iterating through all agents
        for i in company.keys():
            for j in range(0, len(company[i])):
                if company[i][j] is not None:
                    # adding a month to the senority of all agents
                    company[i][j].senority += 1/12
                    # adding a month to the age of all agents
                    company[i][j].age += 1/12
                    # if the agent has reached the age of 68, he should be fired
                    if company[i][j].age >= 68:
                        company[i][j] = None

                # if there is an empty position in the company, a agent should be promoted or created
                if company[i][j] == None:
                    # if level is the lowest, a new agent is created
                    if i == list(company.keys())[-1]:
                        company[i][j] = Agent(position = i, index = j, gender = random.choices(['male', 'female'], weights = [0.5, 0.5], k = 1), age = random.gauss(30, 6), senority = random.gauss(3, 1))

                    # choosing another agent from the lower level of the company to promote (e.g. junior to senior)
                    else:
                        # choosing a random agent from the lower level of the company 
                        # REMEMBER: SHOULD BE CHANGED TO CHOOSE THE BEST AGENT, possibly by age or senority or adding some kind of discrimination
                        agent = None
                        while agent == None: # REMEMBER: This runs infinetly if all agents are None in the given category... FIX THIS
                            agent = random.choice(company[list(company.keys())[2+1]])


                        # promoting the agent
                        company[i][j] = agent
                        # removing the agent from the lower level of the company
                        company[agent.position][agent.index] = None
                        # changing the position of the agent
                        company[i][j].postition = i
                        # changing the index of the agent
                        company[i][j].index = j
                
                # fire one random department head if there are more than 8 department heads
                if i == 'Department Head' and len(company[i]) > 8:
                    company[i][random.randint(0, len(company[i])-1)] = None

                # fire 1 random leader if there are more than 20 leaders
                if i == 'Leader' and len(company[i]) > 20:
                    company[i][random.randint(0, len(company[i])-1)] = None



        # plotting and appending data to data frame                           
        plot_gender(company, tick = month)

        f = {'gender': 'female', 'tick': month, 'department_head': count_gender(company)[0][0],'leader': count_gender(company)[0][1], 'senior': count_gender(company)[0][2], 'junior': count_gender(company)[0][3]}
        m = {'gender': 'male', 'tick': month, 'department_head': count_gender(company)[1][0],'leader': count_gender(company)[1][1], 'senior': count_gender(company)[1][2], 'junior': count_gender(company)[1][3] }
        
        # create pandas dataframe from dictionaries f and m
        new_data = pd.DataFrame.from_dict([f, m])
        
        #new_data_f = pd.DataFrame.from_dict(f)
        #new_data_m = pd.DataFrame.from_dict(d)
        data = data.append(new_data, ignore_index=False, verify_integrity=False, sort=False)
    
    # saving the data to a csv-file
    data.to_csv(save_path)


                