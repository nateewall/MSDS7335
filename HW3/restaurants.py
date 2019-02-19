import random
import numpy as np

# the results of a survey about lunch preferences are captured in the following dictionary on a scale from 1 to 5 with each feature defined as:

# travelDistance: how far are they willing to travel
#       0: would rather stay in the building // 5: willing to drive 30+ minutes

# cost: how much are they willing to pay
#       0: a Cookout tray (NC thing) // 5: Michellin star tasting menu

# instagrammable: how important is the chance of a cool instagram pic
#       0: someone who has never had a social media account // 5:  Bon Appetit social media coordinator

# busy: a measure of a person's aversion to crowds or lines
#       0: a hermit // 5: Diddy in the club

# vegetarian: how important are vegetarian options
#       0: Ron Swanson // 5:'Vegan' tattooed on their chest

# institution: measure of this persons affinity to local institutions
#       0: millennial just looking for a juice bar // 5: atenured professor who get's greeted with a tea




def createRestaurants(dictName, names, min_travel = 0, max_travel = 5,min_cost = 0, max_cost = 5, min_gram = 0, max_gram = 5,
                    min_busy = 0, max_busy = 5,min_vege = 0, max_vege = 5, min_institute = 0, max_institute = 5):
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

#create a random 10 restaurants
restaurants = {}
rNames = ['The Caribbean Flower', 'The Coriander Bites', 'The Indian Lane', 'The Italian Empress',
          'The Juniper Window', 'Chance', 'Bounty', 'Recess', 'Sunset', 'Lemon Grass'] #generated from https://www.fantasynamegenerators.com/restaurant-names.php

createRestaurants(restaurants, rNames)

def matrixDict(dictName, names):
    varNames = list(dictName[names[0]].keys())  # get the survey var names
    dtype = dict(names = varNames, formats=(['i4'] * len(varNames))) #structure for array

    M = np.zeros(len(names), dtype = dtype)

    for n in enumerate(names):
        M[n[0]] = tuple(dictName[n[1]].values())

    return M


M_restaurants = matrixDict(restaurants, rNames)

print(M_restaurants)
print(M_restaurants.dtype)
