import csv #used for initial data exploration
# first lets open the file and test some of the data
# fileName = 'claim.sample.csv'
# with open(fileName,'r') as f:
#     print(f.readline())
#     print(f.readline())



import numpy as np
import os
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics as mt
import matplotlib.pyplot as plt


#create the master project file
projectName = 'JcodePaymentPrediction'

if not os.path.exists(projectName):
    os.makedirs(projectName)

# outfile = str(projectName + '/NWallHW2.txt')
# f = open(outfile, "w+")

fileName = 'claim.sample.csv'

# after looking at the data these types seem reasonable, however, a better test would be ideal
types = ['|S8', '>i4', '>i4', '>i4', '|S14', '|S6', '|S6', '|S6', '|S4', '|S9', '|S7', '|f8',
         '|S5', '|S3', '|S3', '|S3', '|S3', '|S3', 'f8', 'f8', '>i4', '>i4', '>i4',
         '|S3', '|S3', '|S3', '|S4', '|S14', '|S14'] #manually determined dtypes

#read in the data using numpy
data = np.genfromtxt(fileName
                     , names=True
                     , dtype=types
                     , delimiter=',')

# lets look at some of the details about the procedure codes in general
# print(data.dtype.names)
# print(len(data['ProcedureCode'])) #how many obs
# print(len(np.unique(data['ProcedureCode']))) #how many distinct codes for all odds
# unique, counts = np.unique(data['ProcedureCode'], return_counts=True) #get the unique categories & counts
# freq = sorted(dict(zip(unique, counts)).items(), key=lambda x: x[1], reverse=True) #zip to dict and sort
# print(freq)
#
#
# extract just cases where the first characted in procedure code is J and only the first character
def findCode(code,data):
    code = code.encode()
    df = data[np.flatnonzero(np.core.defchararray.find(data['ProcedureCode'],code,end=5) != -1)]
    return df

df = findCode('J', data)

# #-------------------------------------------------------------------------------------------#
# #----------------------------Code used to answer Q1----------------------------------#
# #-------------------------------------------------------------------------------------------#
print("1. J-codes are procedure codes that start with the letter 'J'.")
print()
print('-------------------------------------------------------------------------------------------')
print("A. Find the number of claim lines that have J-codes.")
unique, counts = np.unique(df['ProcedureCode'], return_counts=True) #get the unique categories & counts
print('There are',len(unique),'distinct J-codes representing',sum(counts),'total claim lines')
print('')

print('-------------------------------------------------------------------------------------------')
print("B. How much was paid for J-codes to providers for 'in network' claims?")
networkCode = 'I'
networkCode = networkCode.encode() #encode I fo the join
inNetwork = df[np.flatnonzero(np.core.defchararray.find(df['InOutOfNetwork'],networkCode) != -1)]
print('There are $', round(inNetwork['ProviderPaymentAmount'].sum(),2), 'in Provider Payouts')
print('')

print('-------------------------------------------------------------------------------------------')
print("C. What are the top five J-codes based on the payment to providers?")
providerPayment = {}
for jcode in unique: #loop through the unique procedure codes as numpy had no group by logic
    tmp = df[df['ProcedureCode'] == jcode]
    payment = round(tmp['ProviderPaymentAmount'].sum(),2) # sum all the provider payment amounts by jcode
    providerPayment[jcode] = payment #write to dictionary of jcodes

print('Below are the the top five JCodes by the amount paid to providers:')
providerSumm = sorted(providerPayment.items(), key=lambda x: x[1], reverse=True)[0:4] #sort by payment amounts
for p in providerSumm:
   print(p[0].decode(),"=", p[1]) #pretty print top five

#-------------------------------------------------------------------------------------------#
#----------------------------Code used to answer Q2---------------------------------#
#-------------------------------------------------------------------------------------------#
print('')
print('')
print("2. For the following exercises, determine the number of providers that were paid for at least one J-code.")
    # Use the J-code claims for these providers to complete the following exercises.
paidProviderList = np.unique(df['ProviderID'][df['ProviderPaymentAmount'] > 0]) #get list of providers with payment

#  A. Create a scatter plot that displays the number of unpaid claims
    # (lines where the ‘Provider.Payment.Amount’ field is equal to zero)
    #  for each provider versus the number of paid claims.

print('')
print('-------------------------------------------------------------------------------------------')
print("A. Create a scatter plot that displays the number of unpaid claims")
paid = []
unPaid = []
provider = []
for p in paidProviderList: #loop through the unique procedure codes as numpy had no group by logic
    tmp = df[df['ProviderID'] == p]
    unPaid.append(sum(tmp['ProviderPaymentAmount'] == 0)) # get all the paidClaims
    paid.append(sum(tmp['ProviderPaymentAmount'] > 0)) # get all the paidClaims
    provider.append(p.decode())

print(list(zip(provider, paid, unPaid)))
print('Plot provides a better visual explanation of the data')

# plot the scatter of paid vs unpaid
fig, ax = plt.subplots(figsize=(15, 10))
plt.scatter(paid, unPaid)
for i, p in enumerate(provider):
    ax.annotate(p, (paid[i], unPaid[i])) #get the providerid listed on the plots
ax.set_xlabel('Paid Claim Lines')
ax.set_ylabel('Unpaid Claim Lines')
ax.set_title('Total Paid Claim Lines vs Unpaid Claim Line by Provider')
ax.grid(True) #get some grid lines to make it a little easier to see
# Put a legend to the right of the current axis
plt.show()
print('')

# B. What insights can you suggest from the graph?
print('-------------------------------------------------------------------------------------------')
print('B. What insights can you suggest from the graph?')
print('')
print('There may be a positive linear trend although there are limited observations and there are some outliers.')
print('Provider FA0001387001 & FA1389001 appears to have a higher than expected amount of unpaid claims, although the reason cannot be determined from the available data')
print('Any additional insights from the graph would require understanding of other potential contributing factors')


# C. Based on the graph, is the behavior of any of the providers concerning? Explain.
print('-------------------------------------------------------------------------------------------')
print('C. Based on the graph, is the behavior of any of the providers concerning? Explain.')
print('')
print('Some of the providers with a high rate of unpaid claims are concerning.')
print('Specifically, FA0001387001 & FA1389001 as they appear to fail to pay out on the majority of their claims while also having the highest 2 of the highest volumes of claims.')
print('Some of the providers in the upper right quadrants may show higher number of unpaid claims.')
print('However, FA0001387001 appears to pay out ~1-2% of their total claims.')
print('Although, understanding all the additional confounding factors that may contribute to an unpaid claim is would be important to determine how concerning these volumes of unpaid claims are.')


#-------------------------------------------------------------------------------------------#
#----------------------------Code used to answer Q3---------------------------------#
#-------------------------------------------------------------------------------------------#
print('')
print('')
print("3. Consider all claim lines with a J-code.")

#A. What percentage of J-code claim lines were unpaid?
print('-------------------------------------------------------------------------------------------')
print('A. What percentage of J-code claim lines were unpaid?')
print('Total J-codes =', len(df))
print('Total Unpaid J-Codes =', len(df[df['ProviderPaymentAmount'] == 0]))
print('Percent Un-paid =', (len(df[df['ProviderPaymentAmount'] == 0])/len(df))*100)

print('-------------------------------------------------------------------------------------------')
print('B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.')
print('Before any modeling lets perform very basic EDA to understand what we can use and what we need to do to use it')
print('')

# ----------------------------------# ---------------------------------------------------------#
# ----------------------------Below is some preliminary EDA----------------------------------#
# -------------------------------------------------------------------------------------------#
# find how many obs are missing data where missing is defined as zero for
# this is very specific to this problem set but would be good to make more flexible going forward
def findMissing(data):
    missing = {}  # declare missing values
    for name in data.dtype.names:
        try:
            np.count_nonzero(np.isnan(data[name]))
            zero = np.count_nonzero(data[name] == 0)  # count where == 0 as no numerics a na
            missing[name] = zero
        except:
            null = np.count_nonzero(data[name] == b'" "')  # count empty strings
            missing[name] = null

    fig, ax = plt.subplots(figsize=(15, 10))
    plt.barh(range(len(missing)), list(missing.values()), align='center')
    plt.yticks(range(len(missing)), list(missing.keys()))
    ax.set_xlabel('Number of Observations')
    ax.set_ylabel('Variable Name')
    ax.set_title('Number of Missing  or Zero values by Variable')
    fileName = str(projectName + '/ZeroOrNullCounts.png')  # assumes projectName exists from above
    plt.savefig(fileName)
    plt.close()
    return missing

findMissing(df)


print('Based on a the plot in %s of missing or 0 values there are several potential features we may need to impute' %fileName)
print('To better understand what we may impute with lets see what the most common classes are & numeric distribution')
# set a binary paid vs non-paid
PayStatus = []
for row in df:
    if row['ProviderPaymentAmount'] == 0:
        PayStatus.append(1)
    else:
        PayStatus.append(0)

target = np.array(PayStatus)

dropVars = ['V1', 'ClaimNumber', 'ClaimLineNumber', 'MemberID','DenialReasonCode','ClaimCurrentStatus',
            'ProviderPaymentAmount', 'AgreementID','NetworkID']
names = list(df.dtype.names)
for v in dropVars:
    names.remove(v)
df2 = df[names]

def singleVarSummary(data):
    for name in data.dtype.names: #iterate through each variable
        print(data[name].dtype)
        if data[name].dtype.type is np.string_:
            unique, counts = np.unique(data[name], return_counts=True) #get the unique categories & counts
            freq = sorted(dict(zip(unique, counts)).items(), key=lambda x: x[1], reverse=True) #zip to dict and sort
            if len(unique) < 20 :
                labels, values = [*zip(*freq)]
                print('There are',len(unique),'total classes in', name)
                fig, ax = plt.subplots(figsize=(15, 10))
                plt.barh(range(len(labels)), values, align='center') #plot horizontal bar
                plt.yticks(range(len(labels)), labels)
                ax.set_xlabel('Number of Observations')
                ax.set_ylabel(name)
                ax.set_title('Number of Observations by Class')
                fileName = str(projectName + '/FrequencyPlot' + name + '.png')  # assumes projectName exists from above
                plt.savefig(fileName)
                plt.close()

                # crosstab = pd.crosstab(data[name], np.asarray(PayStatus))  # by categories freqs
                # fig, ax = plt.subplots(figsize=(15, 12))
                # sns.heatmap(crosstab, annot=True, fmt="d", ax=ax, linewidths=.5)
                # fileName = str(projectName + '/CatHeatMap' + name + '.png')  # assumes projectName exists from above
                # plt.savefig(fileName)
                # plt.close()

            else:
                print('There are', len(unique), 'total classes in', name)
                freq20 = freq[0:19]
                labels, values = [*zip(*freq20)]
                fig, ax = plt.subplots(figsize=(15, 10))
                plt.barh(range(len(labels)), values, align='center') #plot horizontal bar
                plt.yticks(range(len(labels)), labels)
                ax.set_xlabel('Number of Observations')
                ax.set_ylabel(name)
                ax.set_title('Number of Observations by Class for')
                fileName = str(projectName + '/FrequencyPlot' + name + '.png')  # assumes projectName exists from above
                plt.savefig(fileName)
                plt.close()

                # crosstab = pd.crosstab(data[name], np.asarray(PayStatus))  # by categories freqs
                # fig, ax = plt.subplots(figsize=(15, 12))
                # sns.heatmap(crosstab, annot=True, fmt="d", ax=ax, linewidths=.5)
                # fileName = str(projectName + '/CatHeatMap' + name + '.png')  # assumes projectName exists from above
                # plt.savefig(fileName)
                # plt.close()

        else:
            print('Below is the mean, median, & standard deviation for', name)
            mu, med, sd = np.mean(data[name]), np.median(data[name]), np.std(data[name])
            print(mu, med, sd)
            hist, bins = np.histogram(data[name], bins=20) #bin the data into 20 bins for plotting
            width = 0.7 * (bins[1] - bins[0]) #set plot width
            center = (bins[:-1] + bins[1:]) / 2 #center bars
            fig, ax = plt.subplots(figsize=(15, 15))
            plt.bar(center, hist, align='center', width=width) #plot
            ax.set_title(name) #name hist by variable
            ax.set_ylabel('Number of Observations')
            fileName = str(projectName + '/Histogram' + name + '.png')  # assumes projectName exists from above
            plt.savefig(fileName)
            plt.close()

singleVarSummary(df2)
print()
print('After the reviewing the plots produced for each variable we will begin by treating missing values as there own class')
print('Additionally, most of the numeric variables are index vars and we will not consider these in our model')
print('We will also weed out some of the categorical values with more than 20 dimensions')
print()

keepVar = ['ProviderID','LineOfBusinessID','RevenueCode','ServiceCode','PriceIndex','InOutOfNetwork',
           'PricingIndex','CapitationIndex','SubscriberPaymentAmount','ClaimType','ClaimSubscriberType']


print('Finally we are ready to start developing our model. We have chosen to train a random forest model with default parameters to begin.')
print('We have chosen this approach because it is more interpretable than some of the other methods. Considering this model reviews the status of insurance claims it is important to understand what features are driving our predictions')
print('')
print('For our modeling approach we will begin by one-hot encoding all our categorical features.')

m = df2[keepVar] #keep what we want
print('Prior to one-hot encoding our matrix shape is:')
print(m.shape)

#there are a a good proportion of obs with null values. The zero values, however we will treat those a uniqueclass
catVar = []
numVar = []
for name in m.dtype.names:  # iterate through each variable
    if m[name].dtype.type is np.string_:
        catVar.append(name)
    else:
        numVar.append(name)


#split the cat and num vars
mCat = np.array(m[catVar].tolist())
mNum = np.array(m[numVar].tolist())


encoder = preprocessing.LabelEncoder()
featureNames = []
for i in range(len(catVar)):
    mCat[:,i] = encoder.fit_transform(mCat[:,i])
    var = catVar[i]
    for c in encoder.classes_:
        feature = var+"_"+(c.decode().strip())
        featureNames.append(feature)

featureNames.append(m[numVar].dtype.names)

oneHotEncode = preprocessing.OneHotEncoder()
oheM = oneHotEncode.fit_transform(mCat).toarray()

mFinal = np.concatenate((oheM,mNum), axis = 1)

print('After one-hot encoding our matrix shape is:')
print(mFinal.shape)

print('We have also chosen to use split our training data & test data using an 80/20 split')
split = np.random.rand(len(mFinal)) < 0.80

xTrain = mFinal[split,:]
yTrain = target[split]
print('The training features set dimensions are:')
print(xTrain.shape)
print('The training target set dimensions are:')
print(yTrain.shape)
print('')

xTest = mFinal[~split,:]
yTest = target[~split]
print('The test features set dimensions are:')
print(xTest.shape)
print('The test target set dimensions are:')
print(yTest.shape)
print('')

print('-------------------------------------------------------------------------------------------')
print("C. How accurate is your model at predicting unpaid claims?")

clf = RandomForestClassifier()
clf.fit(xTrain,yTrain)
yhat = clf.predict(xTest)
print('------------------------------------------')
print('Default Random Forest Results')
print('------------------------------------------')
print('Model Scoring')
print('Accuracy', mt.accuracy_score(yTest, yhat))
print('f1_score:', mt.f1_score(yTest, yhat))
print('Precision:', mt.precision_score(yTest, yhat))
print('Recall:', mt.recall_score(yTest, yhat))
print('Confusion Matrix:')
print(mt.confusion_matrix(yTest, yhat))
print('------------------------------------------')


#zip the feature importance with the varname
print('-------------------------------------------------------------------------------------------')
print('D. What data attributes are predominately influencing the rate of non-payment?')
print("Top 5 Features sorted by their score:")
featureImportance = sorted(zip(featureNames, map(lambda x: round(x, 4), clf.feature_importances_)), reverse=True, key = lambda t: t[1])
print(featureImportance[0:4])

labels, values = [*zip(*featureImportance)]
fig, ax = plt.subplots(figsize=(15, 10))
plt.barh(range(len(labels)), values, align='center') #plot horizontal bar
plt.yticks(range(len(labels)), labels)
ax.set_xlabel('Feature Importance Score')
ax.set_ylabel(labels)
ax.set_title('Feature Importance of Random Forest Model')
fileName = str(projectName + '/FeatureImportance.png')  # assumes projectName exists from above
plt.savefig(fileName)
plt.close()

