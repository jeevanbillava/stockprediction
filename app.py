#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv("50_Startups.csv")


# In[3]:


data=data.drop('State',axis=1)


# In[4]:


data.head()


# In[5]:


x=data.iloc[:,0:-1]
y=data.iloc[:,-1]


# In[6]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)


# In[7]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain,ytrain)


# In[8]:


ypred=model.predict(xtest)


# In[9]:


from sklearn.metrics import r2_score
r2_score(ytest,ypred)


# In[ ]:


from flask import Flask,render_template,request
app=Flask(__name__)
@app.route("/")
def nitin():
    return render_template("es.html")
@app.route("/detail",methods=['POST','GET'])
def jeevan():
    if(request.method=='POST'):
        a=float(request.form['v1'])
        b=float(request.form['v2'])
        c=float(request.form['v3'])
        result=model.predict([[a,b,c]])
        return render_template("es.html",jvn=result)
if __name__=="__main__":
    app.run()


# In[ ]:





# In[ ]:




