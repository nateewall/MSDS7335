# Decision making with Matrices

# This is a pretty simple assingment.  You will do something you do everyday, but today it will be with matrix manipulations.

# The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.  Then you should decided if you should split into two groups so eveyone is happier.

# Displicte the simplictiy of the process you will need to make decisions regarding how to process the data.

# This process was thoughly investigated in the operation research community.  This approah can prove helpful on any number of decsion making problems that are currently not leveraging machine learning.



# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.

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



def createPeople(dictName, numObs, min_travel = 0, max_travel = 5,min_cost = 0, max_cost = 5, min_gram = 0, max_gram = 5,
                    min_busy = 0, max_busy = 5,min_vege = 0, max_vege = 5, min_institute = 0, max_institute = 5):
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

#create a random 10 people
people = {}
createPeople(people, 10)
pNames = list(people.keys())  # get the names of the people

# Transform the user data into a matrix(M_people). Keep track of column and row ids.
def matrixDict(dictName, names):
    varNames = list(dictName[names[0]].keys())  # get the survey var names
    dtype = dict(names = varNames, formats=(['i4'] * len(varNames))) #structure for array

    M = np.zeros(len(names), dtype = dtype)

    for n in enumerate(names):
        M[n[0]] = tuple(dictName[n[1]].values())

    return M

M_people = matrixDict(people, pNames)
print(M_restaurants)
print(M_restaurants.dtype)

# Next you collected data from an internet website. You got the following information.

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


# Transform the restaurant data into a matrix(M_restaurants) use the same column index.

M_restaurants = matrixDict(restaurants, rNames)

print(M_restaurants)
print(M_restaurants.dtype)

# The most imporant idea in this project is the idea of a linear combination.

# Informally describe what a linear combination is and how it will relate to our restaurant matrix.


# Choose a person and compute(using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent.

# Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent?

# Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?

# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal resturant choice.

# Why is there a difference between the two?  What problem arrives?  What does represent in the real world?

# How should you preprocess your data to remove this problem.

# Find user profiles that are problematic, explain why?

# Think of two metrics to compute the disatistifaction with the group.

# Should you split in two groups today?

# Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?

# Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix?



