import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data1=pd.read_csv("loan_data_train.csv")
data2=pd.read_csv("loan_data_test.csv")
columns1=data1.columns
columns2=data2.columns
type(columns1)
np1=np.array(columns1)
np2=np.array(columns2)
col1=list(np1)
col2=list(np2)
#Removing the column which is not present in both the datasets.
for i in col1:
    if i not in col2:
        print(i)

data1.drop("Interest.Rate",axis=1,inplace=True) #Removed "Interest.Rate"
data=pd.concat([data1,data2],ignore_index=True)
data.dtypes
#modifying "Amount.Requested" column
count=0
for i in range(2500):
    if data.loc[i,"Amount.Requested"]==data.loc[i,"Amount.Funded.By.Investors"]:
        count=count+1
#count obtained is around 72%
for i in range(2500):
    if data.loc[i,"Amount.Requested"]==".":
        data.loc[i,"Amount.Requested"]=data.loc[i,"Amount.Funded.By.Investors"]
        
data["Amount.Requested"]=data["Amount.Requested"].astype(np.float64)
#Modifying "Amount.Funded.By.Investors" column
for i in range(2500):
    if data.loc[i,"Amount.Funded.By.Investors"]==".":
        data.loc[i,"Amount.Funded.By.Investors"]=data.loc[i,"Amount.Requested"]
data["Amount.Funded.By.Investors"]=data["Amount.Funded.By.Investors"].astype(np.float64)   

#separating "FICO.Range" in the upper and lower element. 
l=[]
for i in range(2500):
    data.loc[i,"FICO.Range"]=str(data.loc[i,"FICO.Range"])
    l.append(data.loc[i,"FICO.Range"])
l[1]
l[1].split("-")[1]
l_lower=[]
l_upper=[]
for i in range(len(l)):
    l[i].split("-")
    l_lower.append(l[i].split("-")[0])
    l_upper.append(l[i].split("-")[1])
data["lower_FICO"]=l_lower
data["upper_FICO"]=l_upper


#handling missing values in "Loan.Length" column
data["Loan.Length"]=data["Loan.Length"].astype("category")
data["Loan.Length"].value_counts()
#At this point we encounter with a problem
#total counts obtained from various categories is 2499,which is one less than total obs.

for i in data["Loan.Length"]:
    if i!="60 months":
        if i!="36 months":
            print(i)
            
for i in range(2500):
    if data.loc[i,"Loan.Length"]=="." or data.loc[i,"Loan.Length"]==np.nan:
        data.loc[i,"Loan.Length"]="36 months"
        
#the missing value "." is cured but "nan" is remaining
# will use sklearn to tackle this situation.

from sklearn.impute import SimpleImputer
si=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
si=si.fit(data[["Loan.Length"]])
data[["Loan.Length"]]=si.transform(data[["Loan.Length"]])

# "Revolving.CREDIT.Balance"
for i in range(len(data["Revolving.CREDIT.Balance"])):
    if data.loc[i,"Revolving.CREDIT.Balance"]==".":
        data.loc[i,"Revolving.CREDIT.Balance"]=0
data["Revolving.CREDIT.Balance"]=data["Revolving.CREDIT.Balance"].astype(np.float64)
data.dtypes

#Employment.Length
data["Employment.Length"].value_counts()

si=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
si=si.fit(data[["Employment.Length"]])
data[["Employment.Length"]]=si.transform(data[["Employment.Length"]])

for i in range(2500):
    if data.loc[i,"Employment.Length"]=="." or data.loc[i,"Employment.Length"]==np.nan:
        data.loc[i,"Employment.Length"]="10+ years"

plt.scatter(data["Amount.Requested"],data["Employment.Length"],c=data["Loan.Purpose"])
data["Loan.Purpose"]=data["Loan.Purpose"].astype("category")
data["Loan.Purpose"].value_counts().plot(kind="bar")

#Debt.To.Income.Ratio
# remove the percentage sign

si=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
si=si.fit(data[["Debt.To.Income.Ratio"]])
data[["Debt.To.Income.Ratio"]]=si.transform(data[["Debt.To.Income.Ratio"]])

u=[]
v=list(data["Debt.To.Income.Ratio"])
for i in range(2500):
    u.append(v[i][:-1])
data["Debt.To.Income.Ratio"]=u
data.dtypes
pd.to_numeric(data["Debt.To.Income.Ratio"])
data["Debt.To.Income.Ratio"]=data["Debt.To.Income.Ratio"].astype(np.float64)
plt.scatter(data["Amount.Requested"],data["Debt.To.Income.Ratio"])
data["lower_FICO"]=data["lower_FICO"].astype(np.float64)
data["upper_FICO"]=data["upper_FICO"].astype(np.float64)
corr=data.corr()

