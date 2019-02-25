# Decision making with Matrices
# This is a pretty simple assingment.  You will do something you do everyday, but today it will be with matrix manipulations.
# The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.  Then you should decided if you should split into two groups so eveyone is happier.
# Displicte the simplictiy of the process you will need to make decisions regarding how to process the data.
# This process was thoughly investigated in the operation research community.  This approah can prove helpful on any number of decsion making problems that are currently not leveraging machine learning.


import random
import names
import numpy as np
from scipy.stats import rankdata

random.seed(3)

print("You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.")
print('')
# the results of a survey about lunch preferences are captured in the following dictionary on a scale from 1 to 5 with each feature defined as:

# travelDistance: how far are they willing to travel
#       10: would rather stay in the building // 1: willing to drive 30+ minutes

# cost: how much are they willing to pay
#       10: a Cookout tray (NC thing) // 1: Michellin star tasting menu

# instagrammable: how important is the chance of a cool instagram pic
#       10: someone who has never had a social media account // 1:  Bon Appetit social media coordinator

# busy: a measure of a person's aversion to crowds or lines
#       10: a hermit // 1: Diddy in the club

# vegetarian: how important are vegetarian options
#       10: Ron Swanson // 1:'Vegan' tattooed on their chest

# institution: measure of this persons affinity to local institutions
#       10: millennial just looking for a juice bar // 1: tenured professor who get's greeted with a tea



def createPeople(dictName, numObs, min_travel = 1, max_travel = 10, min_cost = 1, max_cost = 10, min_gram = 1, max_gram = 10,
                 min_busy = 1, max_busy = 10, min_vege = 1, max_vege = 10, min_institute = 1, max_institute = 10):
    '''
        Create random 'survey' results for fake people
            dictName : dictionary you want to output
            numObs : how many fake survey obs you want
            min_* : the bottom of the range you want to populate (int)
            max_*: the top of the range you to populate that value for (int)
        return
            dictName : python dictionary
    '''
    for i in range(numObs):

        name = names.get_first_name() #generate a random name
        dictName[name] = {'travelDistance': random.randint(min_travel,max_travel),
                          'cost':random.randint(min_cost,max_cost),
                          'instagrammable':random.randint(min_gram,max_gram),
                          'busy':random.randint(min_busy,max_busy),
                          'vegetarian':random.randint(min_vege,max_vege),
                          'institution':random.randint(min_institute,max_institute)
                           }
    return dictName

#"create a random 10 people"
people = {}
createPeople(people, 10)
pNames = list(people.keys())  # get the names of the people
print(people)
print('')
print("----------------------------------------------------------------------------------------------------")
print("Transform the user data into a matrix(M_people). Keep track of column and row ids.")
def matrixDict(dictName, names):
    varNames = list(dictName[names[0]].keys())  # get the survey var names
    dtype = dict(names = varNames, formats=(['i4'] * len(varNames))) #structure for array

    M = np.zeros(len(names), dtype = dtype)

    for n in enumerate(names):
        M[n[0]] = tuple(dictName[n[1]].values())

    return M

M_people = matrixDict(people, pNames)
print(M_people)
print(M_people.dtype)
print('')

print("----------------------------------------------------------------------------------------------------")
print("Next you collected data from an internet website. You got the following information.")

def createRestaurants(dictName, names, min_travel = 1, max_travel = 10,min_cost = 1, max_cost = 10, min_gram = 1, max_gram = 10,
                    min_busy = 1, max_busy = 10, min_vege = 1, max_vege = 10, min_institute = 1, max_institute = 10):
    '''
        Create random 'survey' results for fake restaurants
            dictName : dictionary you want to output
            numObs : how many fake survey obs you want
            min_* : the bottom of the range you want to populate (int)
            max_*: the top of the range you to populate that value for (int)
        return
            dictName : python dictionary
    '''
    for name in names:
        dictName[name] = {'travelDistance': random.randint(min_travel,max_travel),
                          'cost':random.randint(min_cost,max_cost),
                          'instagrammable':random.randint(min_gram,max_gram),
                          'busy':random.randint(min_busy,max_busy),
                          'vegetarian':random.randint(min_vege,max_vege),
                          'institution':random.randint(min_institute,max_institute)
                           }
    return dictName

#"create a random 10 restaurants"
restaurants = {}
rNames = ['The Caribbean Flower', 'The Coriander Bites', 'The Indian Lane', 'The Italian Empress',
          'The Juniper Window', 'Chance', 'Bounty', 'Recess', 'Sunset', 'Lemon Grass'] #generated from https://www.fantasynamegenerators.com/restaurant-names.php

createRestaurants(restaurants, rNames)
print(restaurants)
print('')
print("----------------------------------------------------------------------------------------------------")
print("Transform the restaurant data into a matrix(M_restaurants) use the same column index.")
print('')
M_restaurants = matrixDict(restaurants, rNames)

print(M_restaurants)
print(M_restaurants.dtype)
print('')

# print("----------------------------------------------------------------------------------------------------")
# print("The most imporant idea in this project is the idea of a linear combination.")
# print('')
# print("Informally describe what a linear combination is and how it will relate to our restaurant matrix.")
# print('')
# print('Linear combination is the idea that given a matrix of data and corresponding weights of each feature of that matrix we can determine optimal decision/ranking')
# print('')
# print('Alternatively, given a matrix or vector and the scores/rankings we can determine the weights that would produce that output')
# print('')
# print('So for our examples given the characteristics of several restaurants and the weights each person places on those characteristics we can determine an optimal restaurant')
# print('')
# print('Similarly if we are given those restaurant charactertics and the all the peoples rankings of the rest. we can determine everyones preferences')
# print('')

print("----------------------------------------------------------------------------------------------------")
print("Choose a person and compute(using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent.")
# x.view(np.float64).reshape(x.shape + (-1,))
R = M_restaurants.view(int).reshape(len(M_restaurants),-1)
P = M_people.view(int).reshape(len(M_people),-1)

# person0score = np.dot(R, P[0])
#
# restScore = sorted(zip(rNames, map(lambda x: round(x, 4), list(person0score))), reverse=True, key = lambda t: t[1])
# print('')
# print("The top restaurant for %s and score is:" % list(people.keys())[0])
# print(restScore[0])
# print('')
# print("Each score in the resulting vector represents their compatibility with that restaurant based on the similarity between their preferences and the restaurants characteristics")
# print('')
print("----------------------------------------------------------------------------------------------------")
print("Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent?")

M_usr_x_rest = np.matmul(R, P.T)

print(M_usr_x_rest)
print('')
print('Each ijth value represents an individual users score or preference for that type of restaurant with higher scores meaning a greater preference')
print('')

# Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?
rawScoreRest = {}
for n in enumerate(rNames):
    print(n[1])
    print(sum(M_usr_x_rest[:,n[0]]))
    rawScoreRest[n[1]] = sum(M_usr_x_rest[:,n[0]])


# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal resturant choice.
# M_usr_x_rest_rank = np.zeros(len(rNames))
for n in enumerate(rNames):
    print(M_usr_x_rest[:, n[0]])
    print(rankdata(M_usr_x_rest[:, n[0]], method='dense'))


# Why is there a difference between the two?  What problem arrives?  What does represent in the real world?


# How should you preprocess your data to remove this problem.


# Find user profiles that are problematic, explain why?


# Think of two metrics to compute the disatistifaction with the group.


# Should you split in two groups today?


# Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?


# Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix?



