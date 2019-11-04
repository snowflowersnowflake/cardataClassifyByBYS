import pandas as pd



#make some bugs

data = pd.read_csv('cardata.csv')
print(data.head())
data.to_csv('dataHEAD.csv' ,index=False)
ffff=open('dataNotNull.txt', 'w+')
print(data.all().notnull(), file=ffff)

X_data = data.drop('class',axis = 1)
y_data = data['class']


print(X_data.shape)
print(y_data.shape)

X_data = data.drop('class',axis = 1)
y_data = data['class']

# make onehot
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = False)

X_data = vec.fit_transform(X_data.to_dict(orient = 'records'))
vec.get_feature_names()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_data = le.fit_transform(y_data)
classes = le.classes_
print(classes)



# GaussianNB
from sklearn.naive_bayes import GaussianNB
gaussianNB=GaussianNB()
gaussianNB.fit(X_data,y_data)


dataset_predict_y=gaussianNB.predict(X_data)
correct_predicts=(dataset_predict_y==y_data).sum()
accuracy=100*correct_predicts/y_data.shape[0]
print('GaussianNB, correct prediction num: {}, accuracy: {:.2f}%'
      .format(correct_predicts,accuracy))



'''
#split!!!!
X_train, X_test, y_train, y_test = train_test_split(X_data,y_data,test_size=0.1,random_state=12)
train=pd.concat([X_train,y_train], axis=1)
print(train.shape)
#print(type(train))
test=pd.concat([X_test,y_test], axis=1)

#>>!replace
train=train.replace({'class':'unacc'},0)
train=train.replace({'class':'acc'},1)
test=test.replace({'class':'unacc'},0)
test=test.replace({'class':'acc'},1)
print(train.head())

# makecls==

classlabels_unacc = train[train['class'] == 0]
print(classlabels_unacc.head())
print(classlabels_unacc.shape)

classlabels_acc = train[train['class'] == 1]




totals = {'u': len(classlabels_unacc),
          'a': len(classlabels_acc)}

frequency_list_u = defaultdict(int)
for classlabel in classlabels_unacc['class']:
    for char in classlabel:
        frequency_list_u[char] += 1. / totals['u']


frequency_list_a = defaultdict(int)
for classlabel in classlabels_acc['class']:
    for char in classlabel:
        frequency_list_a[char] += 1. / totals['a']



#fuckthis

def LaplaceSmooth(char, frequency_list, total, alpha=1.0):
    count = frequency_list[char] * total
    distinct_chars = len(frequency_list)
    freq_smooth = (count + alpha ) / (total + distinct_chars * alpha)
    return freq_smooth

def GetLogProb(char, frequency_list, total):
    freq_smooth = LaplaceSmooth(char, frequency_list, total)
    return math.log(freq_smooth) - math.log(1 - freq_smooth)


def ComputeLogProb(classlabel, bases, totals, frequency_list_a, frequency_list_u):
    logprob_a = bases['a']
    logprob_u = bases['u']
    for char in classlabel:
        logprob_a += GetLogProb(char, frequency_list_a, totals['a'])
        logprob_u += GetLogProb(char, frequency_list_u, totals['u'])
    return {'acc': logprob_a, 'unacc': logprob_u}

def GetGender(LogProbs):
    return LogProbs['acc'] > LogProbs['unacc']

#combat
base_u = math.log(1 - train['class'].mean())
base_u += sum([math.log(1 - frequency_list_u[char]) for char in frequency_list_u])

base_a = math.log(train['class'].mean())
base_a += sum([math.log(1 - frequency_list_a[char]) for char in frequency_list_a])

bases = {'u': base_u, 'a': base_a}

result = []
for classlabel in test['classlabel']:
    LogProbs = ComputeLogProb(classlabel, bases, totals, frequency_list_a, frequency_list_u)
    gender = GetGender(LogProbs)
    result.append(int(gender))

result.to_csv('my_NB_prediction.csv', index=False)

'''
