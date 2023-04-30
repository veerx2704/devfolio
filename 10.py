'''
import seaborn as sns
import numpy as np
sns.set(style="ticks", color_codes=True)
dfiris = sns.load_dataset("iris")
sns.pairplot(dfiris, hue="species", palette="husl")

labels = np.asarray(dfiris.species)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(labels)
labels = le.transform(labels)

df_selected1 = dfiris.drop(['sepal_width', "species"], axis=1)
df_features = df_selected1.to_dict(orient='records')

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
features = vec.fit_transform(df_features).toarray()

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
features, labels, test_size=0.20, random_state=0)

from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(features_train, labels_train)
svm_predictions = svm_model_linear.predict(features_train)
accuracy = svm_model_linear.score(features_train, labels_train)
print(accuracy)
print(dfiris)
print(features)

import sklearn
import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()

x=iris.data
y=iris.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression

clf=LogisticRegression(random_state=42)
clf.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
y_pred=clf.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)


from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, x, y, cv=5)
print("Cross-validation scores:", scores)



'''

    
'''
from selenium import webdriver
driver=webdriver.Chrome()
driver.get('www.google.com')
element=driver.find_element_by_id('python')
element.click()
element.send_keys('python')
test=element.text
driver.quit()

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
driver = webdriver.Chrome()
driver.get("https://www.google.com")
search_box = driver.find_elements("q")
search_box.send_keys("Python")
search_box.submit()
driver.implicitly_wait(10)
results = driver.find_elements_by_css_selector('div.g')
for result in results:
    title = result.find_element_by_css_selector('h3').text
    link = result.find_element_by_css_selector('a').get_attribute('href')
    print(title)
    print(link)
driver.quit()

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x=np.array([5,15,25,35,45,55]).reshape((-1,1))
y=np.array([5,20,14,32,22,38])
model=LinearRegression()
model.fit(x,y)
y_pred=model.predict(x)
plt.scatter(x,y)
plt.plot(x,y_pred,color='red')
plt.show()
'''

import requests
from bs4 import BeautifulSoup




url = "https://www.twitter.com"
response = requests.get(url)
cp=[]
keywords=[]
usedlink=[url]
if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")

    links = soup.find_all("a")
    
    for link in links:
        print(link.text.strip(), link.get("href"))
        cp.append([link.text.strip(),link.get("href")])
else:
    print("Failed to fetch the website data.")
for i in cp:
    if ('/' or '//' or 'http' or 'www') in i[1]:
        if i[1] not in usedlink:
            url=i[1]
            response=requests.get(url)
            if response.status_code==200:
                soup=BeautifulSoup(response.content,'html.parser')
                links=soup.find_all("a")
                for link in links:
                    cp.append([link.text.strip(),link.get("href")])
            else:
                print("Failed to fetch data")
        usedlink.append(i[1])
for i in cp:
    print(i[1])
print('\n',usedlink)
    
