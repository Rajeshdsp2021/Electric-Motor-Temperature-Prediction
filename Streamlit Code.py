%%writefile app.py
 
import pickle
import streamlit as st
from numpy.core.numeric import True_
from sklearn import metrics
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import mean_absolute_error

 
# loading the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(ambient,coolant,U_D,U_Q,Motor_Speed,Torque,I_D,I_Q,PM,Stator_Yoke,Stator_Tooth,Stator_Winding,Profile_ID):

    ambient = st.ambient_input('Enter a value')
    st.write('The current value is ', ambient)

    coolant = st.coolant_input('Enter a value')
    st.write('The current value is ', coolant)

    u_d = st.u_d_input('Enter a value')
    st.write('The current value is ', u_d)

    u_q = st.u_q_input('Enter a value')
    st.write('The current value is ', u_q)

    motor_speed = st.motor_speed_input('Enter a value')
    st.write('The current value is ', motor_speed)

    torque = st.torque_input('Enter a value')
    st.write('The current value is ', torque)
 
    i_d = st.i_d_input('Enter a value')
    st.write('The current value is ', i_d)
 
    i_q = st.i_q_input('Enter a value')
    st.write('The current value is ', i_q)

    pm = st.pm_input('Enter a value')
    st.write('The current value is ', pm)

    stator_yoke = st.stator_yoke_input('Enter a value')
    st.write('The current value is ', stator_yoke)

    stator_tooth = st.stator_tooth_input('Enter a value')
    st.write('The current value is ', stator_tooth)

    stator_winding = st.stator_windig_input('Enter a value')
    st.write('The current value is ', stator_winding)

    profile_id = st.profile_id_input('Enter a value')
    st.write('The current value is ', profile_id)

    # split the dataset
    X_train, X_val, Y_train, Y_val = train_test_split(
        	X,Y,test_size = 0.2, random_state=12)

   # X_train, X_val, Y_train, Y_val = train_test_split

    # default axis range
Y_max_train = max([max(Y_train), max(Y_pred_train)])
y_max_val = max([max(Y_val), max(Y_pred_val)])
y_max = int(max([y_max_train, y_max_val])) 

# interactive axis range
left_column, right_column = st.beta_columns(2)
x_min = left_column.number_input('x_min:',value=0,step=1)
x_max = right_column.number_input('x_max:',value=y_max,step=1)
left_column, right_column = st.beta_columns(2)
y_min = left_column.number_input('y_min:',value=0,step=1)
y_max = right_column.number_input('y_max:',value=y_max,step=1)


fig = plt.figure(figsize=(3, 3))
if show_train == 'Yes':
	plt.scatter(Y_train, Y_pred_train,lw=0.1,color="r",label="training data")
if show_val == 'Yes':
	plt.scatter(Y_val, Y_pred_val,lw=0.1,color="b",label="validation data")
plt.xlabel("PRICES",fontsize=8)
plt.ylabel("PRICES of prediction",fontsize=8)
plt.xlim(int(x_min), int(x_max)+5)
plt.ylim(int(y_min), int(y_max)+5)
plt.legend(fontsize=6)
plt.tick_params(labelsize=6)
st.pyplot(fig)

@st.cache(persist=True)
prediction = LinearRegression()
prediction.fit(X_train, Y_train)

#validation
Y_pred_train = prediction.predict(X_train)
Y_pred_val = prediction.predict(X_val)



#metrics
R2 = r2_score(Y_val, Y_pred_val)
st.write(f'R2 score: {R2:.2f}')



# default axis range
y_max_train = max([max(Y_train), max(Y_pred_train)])
y_max_val = max([max(Y_val), max(Y_pred_val)])
y_max = int(max([y_max_train, y_max_val])) 

# interactive axis range
left_column, right_column = st.beta_columns(2)
x_min = left_column.number_input('x_min:',value=0,step=1)
x_max = right_column.number_input('x_max:',value=y_max,step=1)
left_column, right_column = st.beta_columns(2)
y_min = left_column.number_input('y_min:',value=0,step=1)
y_max = right_column.number_input('y_max:',value=y_max,step=1)


fig = plt.figure(figsize=(3, 3))
if show_train == 'Yes':
	plt.scatter(Y_train, Y_pred_train,lw=0.1,color="r",label="training data")
if show_val == 'Yes':
	plt.scatter(Y_val, Y_pred_val,lw=0.1,color="b",label="validation data")
plt.xlabel("PRICES",fontsize=8)
plt.ylabel("PRICES of prediction",fontsize=8)
plt.xlim(int(x_min), int(x_max)+5)
plt.ylim(int(y_min), int(y_max)+5)
plt.legend(fontsize=6)
plt.tick_params(labelsize=6)
st.pyplot(fig)

