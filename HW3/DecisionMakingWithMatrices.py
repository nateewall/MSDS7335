# Decision making with Matrices
# This is a pretty simple assingment.  You will do something you do everyday, but today it will be with matrix manipulations.
# The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.  Then you should decided if you should split into two groups so eveyone is happier.
# Displicte the simplictiy of the process you will need to make decisions regarding how to process the data.
# This process was thoughly investigated in the operation research community.  This approah can prove helpful on any number of decsion making problems that are currently not leveraging machine learning.


import random
import names
import numpy as np
import os
from scipy.stats import rankdata
import matplotlib.pyplot as plt

random.seed(3)
#create the master project file
projectName = 'RestaurantSelection'

if not os.path.exists(projectName):
    os.makedirs(projectName)

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
for name, value in people.items():
    print(name)
    print(value)

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

for name, value in restaurants.items():
    print(name)
    print(value)

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
#
# print("----------------------------------------------------------------------------------------------------")
# print("Choose a person and compute (using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent.")


R = M_restaurants.view(int).reshape(len(M_restaurants),-1)
P = M_people.view(int).reshape(len(M_people),-1)

# person0score = np.dot(R, P[0])
# restScore = sorted(zip(rNames, map(lambda x: round(x, 4), list(person0score))), reverse=True, key = lambda t: t[1])
#
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

print("----------------------------------------------------------------------------------------------------")
print('Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?')
rawScore = np.sum(M_usr_x_rest, axis = 0)
restRawScore = sorted(zip(rNames, map(lambda x: round(x, 4), list(rawScore))), reverse=True, key = lambda t: t[1])

print('The final sorted restaurant by raw score are shown below.')
for r in restRawScore:
    print(r)

print('')
print('Each value represents the the total compatibility for all the restaurants for the entire group. The goal would be that higher values represent greater overall compatibility.')
print('One issue with scoring this way is if one person is very highly compatible with a restaurant this may drive up the entire overall score')

print("----------------------------------------------------------------------------------------------------")
print('Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal resturant choice.')
tmp = np.empty([len(pNames), len(rNames)], dtype = int)
for n in enumerate(rNames):
    tmp[n[0]] = rankdata(M_usr_x_rest[n[0]], method='dense')

M_usr_x_rest_rank = tmp.reshape(len(M_people),-1)
print(M_usr_x_rest_rank)
print('')

# rankScore = np.sum(M_usr_x_rest_rank, axis = 0)
# print(rankScore)
# rankScoreRest = sorted(zip(rNames, map(lambda x: round(x, 4), list(rankScore))), reverse=True, key = lambda t: t[1])
# print('The final sorted restaurant by ranks scores are shown below, higher values mean more compatible for the group.')
# for r in rankScoreRest:
#     print(r)
# print('')
#
# print("----------------------------------------------------------------------------------------------------")
# print("Why is there a difference between the two?  What problem arrives?  What does represent in the real world?")
# print("")
# print("The most compatible restaurant for the group is different between the raw sum & the sum on the order statistics")
# print("")
# print("This is driven by the fact that certain people have really high raw scores for the restaurant 'Recess' however, that rank value helps reduce the impact of those elevated scores")
# print("")
# print("In the real world this could be related to the people that may have really strong preferences can often times sway the entire group to their preferences.")


# How should you preprocess your data to remove this problem.



# Find user profiles that are problematic, explain why?

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title('Heatmap of Compatibility Scores Person by Restaurant (Raw Score)')  # name hist by variable
plt.imshow(M_usr_x_rest)
# We want to show all ticks
ax.set_xticks(np.arange(len(rNames)))
ax.set_yticks(np.arange(len(pNames)))
#label them with the respective list entries
ax.set_xticklabels(rNames)
ax.set_yticklabels(pNames)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(pNames)):
    for j in range(len(rNames)):
        text = ax.text(j, i, M_usr_x_rest[i, j],
                       ha="center", va="center", color="w")

plt.show()
fileName = str(projectName + '/HeatmapofPersonxRestaurantCompatibilityRawScores.png')  # assumes projectName exists from above
plt.savefig(fileName)
plt.close()


fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title('Heatmap of Compatibility Scores Person by Restaurant (Rank Score)')  # name hist by variable
plt.imshow(M_usr_x_rest_rank)
# We want to show all ticks
ax.set_xticks(np.arange(len(rNames)))
ax.set_yticks(np.arange(len(pNames)))
#label them with the respective list entries
ax.set_xticklabels(rNames)
ax.set_yticklabels(pNames)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(pNames)):
    for j in range(len(rNames)):
        text = ax.text(j, i, M_usr_x_rest_rank[i, j],
                       ha="center", va="center", color="w")

plt.show()
fileName = str(projectName + '/HeatmapofPersonxRestaurantCompatibilityRankScores.png')  # assumes projectName exists from above
plt.savefig(fileName)
plt.close()


# Think of two metrics to compute the disatistifaction with the group.

print("First is the function where each person's actual score is subtracted from the groups averaged scores, squared, summed & averaged")


print("Second is the function where each person's actual ranking is subtracted from the groups average rating, squared, summed & averaged")



print("Should you split in two groups today?")



print("Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?")

print("The best way to do this is convert everyones cost to 1 meaning cost is no longer an issue and determine the new scores")


# Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix?

print("Assuming you have everyone's individual rankings you can determine a similar weighting.")

