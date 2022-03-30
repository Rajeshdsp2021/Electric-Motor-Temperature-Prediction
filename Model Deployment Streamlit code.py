from pyexpat import features
import pandas as pd
import numpy as np
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from PIL import Image

st.title("PMS Motor")

if st.sidebar.checkbox("About"):
#    img = Image.open("1.png","2.png","3.png")
    st.image("1.png")
    st.image("2.png")
    st.image("3.png")
    st.image("4.png")
    
if st.sidebar.checkbox("Prediction"):
    
    st.sidebar.header("User input parameters")



    def user_input_features():
#        ambient = st.sidebar.slider('ambient',-8.573954 , 2.967117 )
        ambient = st.sidebar.number_input(label="ambient")
#        st.bar_chart(ambient)
        
#        coolant = st.sidebar.slider('coolant',-1.429349,  2.649032 )   
        coolant = st.sidebar.number_input(label="coolant")       
         
#        u_d     = st.sidebar.slider('Voltage d-component',    -1.655373 , 2.274734 )   
        u_d = st.sidebar.number_input(label="Voltage d-component")        
        
#        u_q     = st.sidebar.slider('Voltage q-component',    -1.861463, 1.793498 )  
        u_q = st.sidebar.number_input(label="Voltage q-component")        
        
#        motor_speed = st.sidebar.slider('motor_speed',  -1.371529, 2.024164 )
        motor_speed = st.sidebar.number_input(label="motor_speed")        
        
        # torque = st.sidebar.slider('torque', -3.345953 , 3.016971 )    
        torque = st.sidebar.number_input(label="torque")        
        
        
#        i_d = st.sidebar.slider('Current d-component',-3.245874 , 1.060937 )
        i_d = st.sidebar.number_input(label="Current d-component")       
        
#        i_q = st.sidebar.slider('Current q-component',-3.341639 , 2.914185 )   
        i_q = st.sidebar.number_input(label="Current q-component")       
        
        # stator_yoke = st.sidebar.slider('stator_yoke', -1.834688, 2.449158  )
        stator_yoke = st.sidebar.number_input(label="stator_yoke")        
        
#        stator_tooth = st.sidebar.slider('stator_tooth', -2.066143, 2.326668 )
        stator_tooth = st.sidebar.number_input(label="stator_tooth")        
        
        # stator_winding = st.sidebar.slider('stator_winding', -2.019973 ,  2.653781)
        stator_winding = st.sidebar.number_input(label="stator_winding")
    
        data = {	'ambient' : ambient,
			'coolant':coolant,
			'Voltage d-component':u_d,
      		'Voltage q-component':u_q,
			'motor_speed':motor_speed,
		    'torque':torque,
			'Current d-component':i_d,
			'Current q-component':i_q,
			'stator_yoke':stator_yoke,
			'stator_tooth':stator_tooth,               
			'stator_winding':stator_winding
            }
    
    
        features = pd.DataFrame(data ,index = [0])
        return features


    df = user_input_features()
    st.subheader("User input Parameters")
    st.table(df, )

    chart_data = pd.DataFrame(df.values[0],)
    st.bar_chart(chart_data)


    with open('model_rf.pkl','rb') as f:
        mp_rf = pickle.load(f)
        
    with open('model_xgb.pkl','rb') as g:
        mp_xgb = pickle.load(g)
        
    status = st.radio(("Select model"),("RF","XGB","None"))
    if st.button("Predict"):
        if status == "RF":
            pred_rf = mp_rf.predict(df)
            st.markdown("### value of Permanent Magnet surface temperature")
            st.markdown(pred_rf)
        
        elif status == "XGB":
            pred_xgb = mp_xgb.predict(df)
            st.markdown("### value of Permanent Magnet surface temperature")
            st.markdown(pred_xgb)
            
#            st.info("Using XGB")
        
        else:
            st.warning("Pic some Model")



