from pyexpat import features
from tkinter import Y
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import numpy as np
import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import get_file
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import seaborn as sns
import datetime as dt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
#from sklearn.externals import joblib

df = pd.read_csv('Train_data2.csv', header=None,skiprows=1)
#print(df)

df.columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'xAttack'
]

df[0:19289]

print("Read {} rows.".format(len(df)))
print('='*40)
print('The number of data points are:', df.shape[0])
print('='*40)
print('The number of features are:', df.shape[1])
print('='*40)
output = df['xAttack'].values
output2 = df['service'].values
output3 = df['flag'].values
labels = set(output)
labels2 = set(output2)
labels3 = set(output3)
print('The different type of output labels are:', labels)
print('='*125)
print('The number of different output labels are:', len(labels))
print('The different type of services are:', labels2)
print('='*125)
print('The different type of flags are:', labels3)
print('='*125)

# Data Cleaning

# Checking for NULL values
print('Null values in dataset are',len(df[df.isnull().any(1)]))
print('='*40)

# Checkng for DUPLICATE values
df.drop_duplicates(keep='first', inplace = True)

# For now, just drop NA's (rows with missing values)
df.dropna(inplace=True,axis=1)


print("Read {} rows.".format(len(df)))

# Exploratory data analysis
plt.figure(figsize=(15,7))
class_distribution = df['xAttack'].value_counts()
class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()


sorted_yi = np.argsort(-class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', class_distribution.index[i],':', class_distribution.values[i],
          '(', np.round((class_distribution.values[i]/df.shape[0]*100), 3), '%)')

#make a boxplot for univariate analysis
plt.figure(figsize=(20,16))
sns.set(style="whitegrid")
ax = sns.violinplot(x="xAttack", y="duration", data=df, fliersize=None)
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)
plt.show()


X_train, X_test, Y_train, Y_test = train_test_split(df.drop('xAttack', axis=1), df['xAttack'], stratify=df['xAttack'], test_size=0.25)
print('Train data')
print(X_train.shape)
print(Y_train.shape)
print('='*20)
print('Test data')
print(X_test.shape)
print(Y_test.shape)


#Process of features with One-hot Encoding
#1. Protocol_type
#Protocol types are: ['udp', 'tcp', 'icmp']

protocol = list(X_train['protocol_type'].values)
protocol = list(set(protocol))
print('Protocol types are:', protocol)
one_hot = CountVectorizer(vocabulary=protocol, binary=True)
train_protocol = one_hot.fit_transform(X_train['protocol_type'].values)
test_protocol = one_hot.transform(X_test['protocol_type'].values)
print(train_protocol[826].toarray())
print(train_protocol.shape)



#Standardizing the features
#Function that performs the standardisation on the features
def feature_scaling(X_train, X_test, feature_name):

    scaler = StandardScaler()
    scaler1 = scaler.fit_transform(X_train[feature_name].values.reshape(-1, 1))
    scaler2 = scaler.transform(X_test[feature_name].values.reshape(-1, 1))

    return scaler1, scaler2
#0.0. service
service1, service2 = feature_scaling(X_train, X_test, 'service')
print(service1[1])

#0. Flag
flag1, flag2 = feature_scaling(X_train, X_test, 'flag')
print(flag1[1])

#1. Duration
duration1, duration2 = feature_scaling(X_train, X_test, 'duration')
print(duration1[1])

#2. src_bytes
src_bytes1, src_bytes2 = feature_scaling(X_train, X_test, 'src_bytes')
print(src_bytes1[1])

#3. dst_bytes
dst_bytes1, dst_bytes2 = feature_scaling(X_train, X_test, 'dst_bytes')
print(dst_bytes1[1])

#4.land
land1, land2 = feature_scaling(X_train, X_test, 'land')
print(land1[1])

#5.wrong_fragment
wrong_fragment1, wrong_fragment2 = feature_scaling(X_train, X_test, 'wrong_fragment')
print(wrong_fragment1[1])

#6.urgent
urgent1, urgent2 = feature_scaling(X_train, X_test, 'urgent')
print(urgent1[1])

#7.hot
hot1, hot2 = feature_scaling(X_train, X_test, 'hot')
print(hot1[1])

#8.num_failed_logins
num_failed_logins1, num_failed_logins2 = feature_scaling(X_train, X_test, 'num_failed_logins')
print(num_failed_logins1[1])

#9.logged_in
logged_in1, logged_in2 = feature_scaling(X_train, X_test, 'logged_in')
print(logged_in1[1])

#10.num_compromised
num_compromised1, num_compromised2 = feature_scaling(X_train, X_test, 'num_compromised')
print(num_compromised1[1])

#11.root_shell
root_shell1, root_shell2 = feature_scaling(X_train, X_test, 'root_shell')
print(root_shell1[1])

#12.su_attempted
su_attempted1, su_attempted2 = feature_scaling(X_train, X_test, 'su_attempted')
print(su_attempted1[1])

#13.num_root
num_root1, num_root2 = feature_scaling(X_train, X_test, 'num_root')
print(num_root1[1])

#14.num_file_creations
num_file_creations1, num_file_creations2 = feature_scaling(X_train, X_test, 'num_file_creations')
print(num_file_creations1[1])

#15.num_shells
num_shells1, num_shells2 = feature_scaling(X_train, X_test, 'num_shells')
print(num_shells1[1])

#16.num_access_files
num_access_files1, num_access_files2 = feature_scaling(X_train, X_test, 'num_access_files')
print(num_access_files1[1])
#----------------------
#17. num_outbound_cmds
num_outbound_cmds1, num_outbound_cmds2 = feature_scaling(X_train, X_test, 'num_outbound_cmds')
print(num_outbound_cmds1[1])

#18. is_host_login
is_host_login1, is_host_login2 = feature_scaling(X_train, X_test, 'is_host_login')
print(is_host_login1[1])

#19. is_guest_login
is_guest_login1, is_guest_login2 = feature_scaling(X_train, X_test, 'is_guest_login')
print(is_guest_login1[1])

#20. count
count1, count2 = feature_scaling(X_train, X_test, 'count')
print(count1[1])

#21.srv_count
srv_count1, srv_count2 = feature_scaling(X_train, X_test, 'srv_count')
print(srv_count1[1])

#22. serror_rate
serror_rate1, serror_rate2 = feature_scaling(X_train, X_test, 'serror_rate')
print(serror_rate1[1])

#23.srv_serror_rate
srv_serror_rate1, srv_serror_rate2 = feature_scaling(X_train, X_test, 'srv_serror_rate')
print(srv_serror_rate1[1])

#24.rerror_rate
rerror_rate1, rerror_rate2 = feature_scaling(X_train, X_test, 'rerror_rate')
print(rerror_rate1[1])

#25.srv_rerror_rate
srv_rerror_rate1, srv_rerror_rate2 = feature_scaling(X_train, X_test, 'srv_rerror_rate')
print(srv_rerror_rate1[1])

#26.same_srv_rate
same_srv_rate1, same_srv_rate2 = feature_scaling(X_train, X_test, 'same_srv_rate')
print(same_srv_rate1[1])

#27.diff_srv_rate
diff_srv_rate1, diff_srv_rate2 = feature_scaling(X_train, X_test, 'diff_srv_rate')
print(diff_srv_rate1[1])

#28.srv_diff_host_rate
srv_diff_host_rate1, srv_diff_host_rate2 = feature_scaling(X_train, X_test, 'srv_diff_host_rate')
print(srv_diff_host_rate1[1])

#29. dst_host_count
dst_host_count1, dst_host_count2 = feature_scaling(X_train, X_test, 'dst_host_count')
print(dst_host_count1[1])

#30.dst_host_srv_count
dst_host_srv_count1, dst_host_srv_count2 = feature_scaling(X_train, X_test, 'dst_host_srv_count')
print(dst_host_srv_count1[1])

#31. dst_host_same_srv_rate
dst_host_same_srv_rate1, dst_host_same_srv_rate2 = feature_scaling(X_train, X_test, 'dst_host_same_srv_rate')
print(dst_host_same_srv_rate1[1])

#32.dst_host_diff_srv_rate
dst_host_diff_srv_rate1, dst_host_diff_srv_rate2 = feature_scaling(X_train, X_test, 'dst_host_diff_srv_rate')
print(dst_host_diff_srv_rate1[1])

#33. dst_host_same_src_port_rate
dst_host_same_src_port_rate1, dst_host_same_src_port_rate2 = feature_scaling(X_train, X_test, 'dst_host_same_src_port_rate')
print(dst_host_same_src_port_rate1[1])

#34. dst_host_srv_diff_host_rate
dst_host_srv_diff_host_rate1, dst_host_srv_diff_host_rate2 = feature_scaling(X_train, X_test, 'dst_host_srv_diff_host_rate')
print(dst_host_srv_diff_host_rate1[1])

#35. dst_host_serror_rate
dst_host_serror_rate1, dst_host_serror_rate2 = feature_scaling(X_train, X_test, 'dst_host_serror_rate')
print(dst_host_serror_rate1[1])

#36.dst_host_srv_serror_rate
dst_host_srv_serror_rate1, dst_host_srv_serror_rate2 = feature_scaling(X_train, X_test, 'dst_host_srv_serror_rate')
print(dst_host_srv_serror_rate1[1])

#37.dst_host_rerror_rate
dst_host_rerror_rate1, dst_host_rerror_rate2 = feature_scaling(X_train, X_test, 'dst_host_rerror_rate')
print(dst_host_rerror_rate1[1])

#38.dst_host_srv_rerror_rate
dst_host_srv_rerror_rate1, dst_host_srv_rerror_rate2 = feature_scaling(X_train, X_test, 'dst_host_srv_rerror_rate')
print(dst_host_srv_rerror_rate1[1])


X_train_1 = hstack((duration1, train_protocol, service1, flag1, src_bytes1, dst_bytes1, land1, wrong_fragment1, urgent1, hot1, num_failed_logins1, logged_in1, num_compromised1, root_shell1, su_attempted1, num_root1, num_file_creations1, num_shells1, num_access_files1, num_outbound_cmds1, is_host_login1, is_guest_login1, count1, srv_count1, serror_rate1, srv_serror_rate1, rerror_rate1, srv_rerror_rate1, same_srv_rate1, diff_srv_rate1, srv_diff_host_rate1, dst_host_count1, dst_host_srv_count1, dst_host_same_srv_rate1, dst_host_diff_srv_rate1, dst_host_same_src_port_rate1, dst_host_srv_diff_host_rate1, dst_host_serror_rate1, dst_host_srv_serror_rate1, dst_host_rerror_rate1, dst_host_srv_rerror_rate1))
X_train_1.shape
# #--------------------------------------------------------------------
X_test_1 = hstack((duration2, test_protocol, service2, flag2, src_bytes2, dst_bytes2, land2, wrong_fragment2, urgent2, hot2, num_failed_logins2, logged_in2, num_compromised2, root_shell2, su_attempted2, num_root2, num_file_creations2, num_shells2, num_access_files2, num_outbound_cmds2, is_host_login2, is_guest_login2, count2, srv_count2, serror_rate2, srv_serror_rate2, rerror_rate2, srv_rerror_rate2, same_srv_rate2, diff_srv_rate2, srv_diff_host_rate2, dst_host_count2, dst_host_srv_count2, dst_host_same_srv_rate2, dst_host_diff_srv_rate2, dst_host_same_src_port_rate2, dst_host_srv_diff_host_rate2, dst_host_serror_rate2, dst_host_srv_serror_rate2, dst_host_rerror_rate2, dst_host_srv_rerror_rate2))
X_test_1.shape



#This function computes the confusion matrix using Predicted and Actual values and plots a confusion matrix heatmap
def confusion_matrix_func(Y_test, y_test_pred):
    C = confusion_matrix(Y_test, y_test_pred)
    cm_df = pd.DataFrame(C)
    labels = ['dos', 'normal', 'probe', 'r2l', 'u2r']
    plt.figure(figsize=(20, 15))
    sns.set(font_scale=1.4)
    sns.heatmap(cm_df, annot=True, annot_kws={"size": 12}, fmt='g', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')

    plt.show()

#Fits the model on train data and predict the performance on train and test data.
def model(model_name, X_train, Y_train, X_test, Y_test):

    print('Fitting the model and prediction on train data:')
    start = dt.datetime.now()
    model_name.fit(X_train, Y_train)
    y_tr_pred = model_name.predict(X_train)
    print('Completed')
    print('Time taken:', dt.datetime.now() - start)
    print('=' * 50)

    results_tr = dict()
    y_tr_pred = model_name.predict(X_train)
    results_tr['precision'] = precision_score(Y_train, y_tr_pred, average='weighted')
    #results_tr['roc_auc'] = roc_auc_score(Y_train, y_tr_pred, average='weighted')
    results_tr['recall'] = recall_score(Y_train, y_tr_pred, average='weighted')
    results_tr['f1_score'] = f1_score(Y_train, y_tr_pred, average='weighted')
    results_tr['accuracy'] = accuracy_score(Y_train, y_tr_pred)

    results_test = dict()
    print('Prediction on test data:')
    start = dt.datetime.now()
    y_test_pred = model_name.predict(X_test)
    print('Completed')
    print('Time taken:', dt.datetime.now() - start)
    print('=' * 50)

    print('Performance metrics:')
    print('=' * 50)
    print('Confusion Matrix is:')
    confusion_matrix_func(Y_test, y_test_pred)
    print('=' * 50)
    results_test['precision'] = precision_score(Y_test, y_test_pred, average='weighted')
    print('Precision score is:')
    print(precision_score(Y_test, y_test_pred, average='weighted'))
    print('=' * 50)
    results_test['recall'] = recall_score(Y_test, y_test_pred, average='weighted')
    print('Recall score is:')
    print(recall_score(Y_test, y_test_pred, average='weighted'))
    print('=' * 50)
    results_test['f1_score'] = f1_score(Y_test, y_test_pred, average='weighted')
    print('F1-score is:')
    print(f1_score(Y_test, y_test_pred, average='weighted'))
    print('=' * 50)
    results_test['accuracy'] = accuracy_score(Y_test, y_test_pred)
    print('Accuracy-score is:')
    print(accuracy_score(Y_test, y_test_pred))
    print('=' * 50)
    results_test['model'] = model

    return results_tr, results_test

#This function prints all the grid search attributes
def print_grid_search_attributes(model):

    print('---------------------------')
    print('|      Best Estimator     |')
    print('---------------------------')
    print('\n\t{}\n'.format(model.best_estimator_))


    # parameters that gave best results while performing grid search
    print('---------------------------')
    print('|     Best parameters     |')
    print('---------------------------')
    print('\tParameters of best estimator : \n\n\t{}\n'.format(model.best_params_))
    #  number of cross validation splits
    print('----------------------------------')
    print('|   No of CrossValidation sets   |')
    print('----------------------------------')
    print('\n\tTotal number of cross validation sets: {}\n'.format(model.n_splits_))
    # Average cross validated score of the best estimator, from the Grid Search
    print('---------------------------')
    print('|        Best Score       |')
    print('---------------------------')
    print('\n\tAverage Cross Validate scores of best estimator : \n\n\t{}\n'.format(model.best_score_))


#This function computes the TPR and FPR scores using the actual and predicetd values.
def tpr_fpr_func(Y_tr, Y_pred):

    results = dict()
    Y_tr = Y_tr.to_list()
    tp = 0;
    fp = 0;
    positives = 0;
    negatives = 0;
    length = len(Y_tr)
    for i in range(len(Y_tr)):
        if Y_tr[i] == 'normal.':
            positives += 1
        else:
            negatives += 1

    for i in range(len(Y_pred)):
        if Y_tr[i] == 'normal.' and Y_pred[i] == 'normal.':
            tp += 1
        elif Y_tr[i] != 'normal.' and Y_pred[i] == 'normal.':
            fp += 1

    tpr = tp / positives
    fpr = fp / negatives

    results['tp'] = tp;
    results['tpr'] = tpr;
    results['fp'] = fp;
    results['fpr'] = fpr

    return results


#Creation of the Gaussian Naive Bayes
hyperparameter = {'var_smoothing':[10**x for x in range(-9,3)]}
nb = GaussianNB()
nb_grid = GridSearchCV(nb, param_grid=hyperparameter, cv=5, verbose=1, n_jobs=-1)
nb_grid_results = model(nb_grid, X_train_1.toarray(), Y_train, X_test_1.toarray(), Y_test)
print('NAIVE BAYES CLASSIFIER RESULTS:')
print_grid_search_attributes(nb_grid)
print('='*50)



#Creation of the Decision tree
hyperparameter = {'max_depth':[5, 10, 20, 50], 'min_samples_split':[5, 10, 100]}
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(criterion='gini', splitter='best',class_weight='balanced')
decision_tree_grid = GridSearchCV(decision_tree, param_grid=hyperparameter, cv=3, verbose=1, n_jobs=-1)
decision_tree_grid_results = model(decision_tree_grid, X_train_1.toarray(), Y_train, X_test_1.toarray(), Y_test)
print('DECISION TREE')
print_grid_search_attributes(decision_tree_grid)

from sklearn.svm import SVC


#Creation of the svm
model = SVC()

model.fit(X_train_1.toarray(), Y_train)

#Make predictions with the model

predictions = model.predict(X_test_1.toarray())

#Measure the performance of our model

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

print(classification_report(Y_test, predictions))
confusion_matrix_func(Y_test, predictions)
print(confusion_matrix(Y_test, predictions))