#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI
import uvicorn
import pickle

app = FastAPI(debug = True)

@app.get('/')
def home( ):
    return {'text': 'Customer Personality Analysis'}
    

    
@app.get('/predect') 
def predect(Education: str, Marital_Status: str, Incomes: str, Kidhome: str, Teenhome: str, Purchases: str, Expense: str, Recency: str, Campaign: str, Complain: str, Response: str, NumDealsPurchases: str, NumWebPurchases: str, NumCatalogPurchases: str, NumWebVisitsMonth: str):   
    
    model = pickle.load(open('C:/Users/hp/DS_Project2/classifier.pkl','rb'))
    makepredection = model.predict([[Education, Marital_Status, NumDealsPurchases, NumWebPurchases, NumWebVisitsMonth, NumCatalogPurchases, Incomes, Kidhome, Teenhome, Purchases, Expense, Recency, Campaign, Complain, Response]])   
 
    
    output = round(makepredection[0],2)
    return {'Common cluster is {}'.format(output)}

if __name__ == '__main__':
    uvicorn.run(app)


# In[ ]:




