
import random
import names
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



people = {}

def createPeople(dictName, numPeople, min_travel = 0, max_travel = 5,min_cost = 0, max_cost = 5, min_gram = 0, max_gram = 5,
                    min_busy = 0, max_busy = 5,min_vege = 0, max_vege = 5, min_institute = 0, max_institute = 5):
    for i in range(numPeople):

        name = names.get_first_name()
        dictName[name] = {'travelDistance': random.randint(min_travel,max_travel),
                          'cost':random.randint(min_cost,max_cost),
                          'instagrammable':random.randint(min_gram,max_gram),
                          'busy':random.randint(min_busy,max_busy),
                          'vegetarian':random.randint(min_vege,max_vege),
                          'institution':random.randint(min_institute,max_institute)
                           }
    return dictName

#create some random people
createPeople(people, 10)









