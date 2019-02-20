import numpy as np
import matplotlib.pyplot as plt
import os
import timeit
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ----------------------------------------------------------------------------- #
# From here until the next break is where the user defines the data, k folds, models & directory #

# import the necessary classifier objects
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#tmp import statement for preparing sample data
from sklearn import datasets

# below we will load our training data that we will consider for our
digits = datasets.load_digits()
num_instances = len(digits.images) #then get the number of instances
M = np.array(digits.images.reshape((num_instances, -1))) #create your feature matrix for your data
L = np.array(digits.target) #then create your target vector used to train your model

n_folds = 5 #select the k-folds used to train your model

#create the master project file
projectName = 'DigitClassification'

#define the list of algorithms with the corresponding hyperparameter grids you wanna search
clfDicts = [
    {RandomForestClassifier:{'min_samples_split': [3, 4], 'n_estimators': [10, 20]}},
    {KNeighborsClassifier:{'n_neighbors': [5, 10]}}
]

data = (M, L, n_folds) #create data object

# ----------------------------------------------------------------------------- #

def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data # unpack data containter
  kf = KFold(n_splits=n_folds) # Establish the cross validation
  ret = {} # classic explicaiton of results

  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    clf = a_clf(**clf_hyper) # unpack paramters into clf is they exist
    clf.fit(M[train_index], L[train_index]) #fit the

    pred = clf.predict(M[test_index])

    #   lets try and get the class accuracies to see what the model does well
    conf = confusion_matrix(L[test_index], pred, labels = np.unique(L)) # assigned the unique labels as order
                                                                        # if class is not in test still maintained order
    norm_conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]  # calculate the in class accuracy
    classAcc = dict(zip(np.unique(L),np.diag(norm_conf)))               # build a dict of the class & accuracy

    ret[ids]= {'k': ids,                    #store the ids
               'a_clf': a_clf,              #store the classifier type
               'clf_hyper':clf_hyper,       #store the hyper parameters
               'clf': clf,                  #store the classifier
               'train_index': train_index,  #store the training indexes
               'test_index': test_index,    #store the test indexes
               'classAcc': classAcc,        #store the class accuracies
               'score':[ #store off the 4 metrics that I allow to evaluate the model
                   accuracy_score(L[test_index], pred),
                   # for the precision, recall, & fscore the average is weighted by support size
                   precision_recall_fscore_support(L[test_index], pred, average='weighted')[0],
                   precision_recall_fscore_support(L[test_index], pred, average='weighted')[1],
                   precision_recall_fscore_support(L[test_index], pred, average='weighted')[2]
                   ]
               }

  return ret

def kFoldMetrics(result):
    for key, clf in result.items():
        # print(clf['clf'])
        # get the classifier which we scored
        model = str(clf['a_clf'])+str(clf['clf_hyper'])
        modelStr = str(model).replace("<class '", '').replace("'>", '').replace("{", '').replace("}", '')
        modelStr = modelStr.replace(",", '').replace("'", '').replace(":","")
        # pull the 4 classifier scoring metric for each k-fold
        score = result[key]['score']
        if modelStr in kMetrics:
            kMetrics[modelStr].append(score)  # if you already have scoring metrics for the classifier
            #  then apply the k-fold results
        else:
            kMetrics[modelStr] = [score]  # then start a new dict key and being storing k-fold result
    return kMetrics

def plotKFold(kMetrics):
    for mod, met in kMetrics.items():
        metrics = list(zip(*met)) #zip the metrics together for one list for each metric
        f, axarr = plt.subplots(2, 2,figsize=(20, 15)) #set my size as a 2x2 matrix

        f.suptitle(mod, fontsize = 16).set_position([.5, .97]) #set title to the model being evaluated

        axarr[0, 0].plot(metrics[0]) #plot the accuracy by the k-fold
        axarr[0, 0].set_title('Classifier Accuracy by K-fold' , fontsize = 14) #set accuracy title
        axarr[0, 1].plot(metrics[1]) #plot the precision
        axarr[0, 1].set_title('Classifier Precision by K-fold', fontsize = 14)
        axarr[1, 0].plot(metrics[2]) #plot the recall
        axarr[1, 0].set_title('Classifier Recall by K-fold', fontsize = 14)
        axarr[1, 1].plot(metrics[3]) #plot the fscore
        axarr[1, 1].set_title('Classifier F-Score by K-fold', fontsize = 14)
        for ax in axarr.flat: #set the axis labels
            ax.set(xlabel='K-fold', ylabel='Value')
            ax.label_outer() #only show on the outer subplot axis
            ax.set_ylim([0,1]) #show limits between 0,1
        fileName = str(projectName + '/KFoldPerformance/' + mod + '.png') #assumes projectName exists from above
        plt.savefig(fileName)
        plt.close()
        # plt.show()


def classAccuracy(result):
    for key, clf in result.items():
        # print(clf['clf'])
        # get the classifier which we scored
        model = str(clf['a_clf'])+str(clf['clf_hyper'])
        modelStr = str(model).replace("<class '", '').replace("'>", '').replace("{", '').replace("}", '')
        modelStr = modelStr.replace(",", '').replace("'", '').replace(":","")
        # pull the 4 classifier scoring metric for each k-fold
        classAcc = result[key]['classAcc']
        if modelStr in classMetrics:
            classMetrics[modelStr].append(classAcc)  # if you already have scoring metrics for the classifier
            #  then apply the k-fold results
        else:
            classMetrics[modelStr] = [classAcc]  # then start a new dict key and being storing k-fold result

    for mod, met in classMetrics.items():
        fig, axes = plt.subplots(nrows=n_folds, ncols=1, sharex=True, sharey=False,figsize=(12, 12))
        fig.suptitle(mod, fontsize=14).set_position([.5, .97])  # set title to the model being evaluated
        fig.subplots_adjust(hspace=0)
        for i in range(n_folds):
            axes[i].bar(range(len(met[i])), list(met[i].values()), align='center')
            # ax[i].xticks(range(len(met[i])), list(met[i].keys()))
        for ax in axes:
            ax.label_outer()
        plt.xticks(range(len(met[i])), np.unique(L), rotation='vertical')
        fileName = str(projectName + '/ModelClassAccuracies/' + mod + '.png')
        plt.savefig(fileName)
        plt.close()


def modelEval(kMetrics):
    for mod, met in kMetrics.items():
        metrics = list(zip(*met)) # zip lists together by metric type
        performanceDict[mod]= {
                    # get the average across all k-folds for the 4 metrics
                   'avg_accuracy': np.mean(metrics[0]),
                   'avg_precision': np.mean(metrics[1]),
                   'avg_recall': np.mean(metrics[2]),
                   'avg_fscore': np.mean(metrics[3])
                   }
    return performanceDict


def plotModelEval(performanceDict):
    #print the best performing models based on the metrics captured
    print('-----------------------------------------------------------')
    bestAccuracy = max(performanceDict, key=lambda x: performanceDict[x]['avg_accuracy'])
    accuracy = performanceDict.get(bestAccuracy, {}).get('avg_accuracy')
    print('Highest Accuracy Model: %s' % bestAccuracy)
    print(accuracy)
    print('-----------------------------------------------------------')
    bestPrecision = max(performanceDict, key=lambda x: performanceDict[x]['avg_precision'])
    precision = performanceDict.get(bestPrecision, {}).get('avg_precision')
    print('Highest Precision Model: %s' % bestPrecision)
    print(precision)
    print('-----------------------------------------------------------')
    bestRecall = max(performanceDict, key=lambda x: performanceDict[x]['avg_recall'])
    recall = performanceDict.get(bestRecall, {}).get('avg_recall')
    print('Highest Recall Model: %s' % bestRecall)
    print(recall)
    print('-----------------------------------------------------------')
    bestFscore = max(performanceDict, key=lambda x: performanceDict[x]['avg_fscore'])
    fscore = performanceDict.get(bestFscore, {}).get('avg_fscore')
    print('Highest Fscore Model: %s' % bestFscore)
    print(fscore)


    models = list(performanceDict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'fscore']
    metricMatrix = []
    #convert the dict into an array to be used by matplotlib
    for k1, v1 in performanceDict.items():
        for k2, v2 in v1.items():
            metricMatrix.append(v2) #append the values to the metric matrices
    metricMatrix = np.array(metricMatrix).reshape(len(models), len(metrics)).round(decimals=3)

    fig, ax = plt.subplots(figsize=(25, 15))
    im = ax.imshow(metricMatrix)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, cmap="Wistia")
    cbar.ax.set_ylabel('', rotation=-90, va="bottom")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    # We want to show all ticks
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(models)))
    # and label them with the respective list entries
    ax.set_xticklabels(metrics)
    ax.set_yticklabels(models)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(models)):
        for j in range(len(metrics)):
            text = ax.text(j, i, metricMatrix[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Average Performance by Model & Metric")
    plt.subplots_adjust(top=0.919,
                        bottom=0.165,
                        left=0.0,
                        right=0.827,
                        hspace=0.2,
                        wspace=0.2)
    fileName = str(projectName + '/ModelComparison/ModelMetricHeatMap.png')
    plt.savefig(fileName)
    plt.close()


if not os.path.exists(projectName):
    os.makedirs(projectName)

#create the neccesary output directories
outputDir = ['KFoldPerformance','ModelClassAccuracies','ModelComparison']
for d in outputDir:
    dirName = str(projectName+'/'+d)
    if not os.path.exists(dirName):
        os.makedirs(dirName)


#finally we can run everything through a single loop.
kMetrics = {}
performanceDict = {}
classMetrics = {}
for clf in clfDicts:
    for est, hyper in clf.items():
        params = list(ParameterGrid(hyper)) # sklearn ParameterGrid will create all possible hyperparameter dict
                                            # this function has lots of good exception handling already built in
        print('-----------------------------------------------------------')
        print('Fitting Model: %s' % est)
        print('-----------------------------------------------------------')
        for p in params:
            print('Hyperparameters: %s' %p )
            start = timeit.default_timer()

            result = run(est, data, p)

            stop = timeit.default_timer()

            print('Training Time: ', stop - start)
            print('')
            kMetrics = kFoldMetrics(result)
            plotKFold(kMetrics)
            classAccuracy(result)

performanceDict = modelEval(kMetrics)
print('-----------------------------------------------------------')
print('For details please see the Graphic shown in the %s directory' %projectName)
print('Below is a summary of the best model per metric')
plotModelEval(performanceDict)

