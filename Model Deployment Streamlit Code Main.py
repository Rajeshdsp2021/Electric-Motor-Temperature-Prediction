import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit PMSM</h1>
    </div>
    """
   
    st.title('This is a title')
    st.sidebar.title('This is a title')
    global df
    uploaded_file = st.sidebar.file_uploader(label="Upload your Excel file", type = ['csv','xlsx'])
    if uploaded_file is not None:
       df = pd.read_csv("E:\P-95 Excelr\\temperature1_data.csv")
       st.write("Data Set")
       st.dataframe(df)
   


    if st.sidebar.checkbox("display data",False):
       st.subheader("Show Temperature dataset")
       st.write(df)


    
    
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True)

    st.sidebar.subheader("Choose classifier")
    Model = st.sidebar.selectbox("Model", ("Linear Regression","KNN","XG Boosting", "Ada Boosting","Random Forest","Decision Tree"))

    
    st.sidebar.subheader("Metrics")
    Metrics = st.sidebar.selectbox("Metrics", ("R2","RMSE","MAE"))

    st.sidebar.subheader("Plot the result")

   
    st.sidebar.left_column, st.sidebar.right_column = st.sidebar.columns(2)
    st.sidebar.show_train = st.sidebar.left_column.radio(
	         			'Show the training dataset:', 
             				('Yes','No')
		        		)
    st.sidebar.show_val = st.sidebar.right_column.radio(
        				'Show the validation dataset:', 
	        			('Yes','No')
		        		)

    
      
    # following lines create boxes in which user can enter data required to make prediction 
    Ambient = st.number_input("ambient")
    Coolant = st.number_input("coolant")
    U_D = st.number_input("u_d")
    U_Q = st.number_input("u_q")
    Motor_Speed = st.number_input("motor_speed")
    Torque = st.number_input("torque")
    I_D = st.number_input("i_d")
    I_Q = st.number_input("i_q")
    PM = st.number_input("PM")
    Stator_Yoke = st.number_input("stator_yoke")
    Stator_Tooth = st.number_input("stator_tooth")
    Stator_Winding = st.number_input("stator_winding")
    Profile_ID = st.number_input("profile_id")



    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Ambient,Coolant,U_D,U_Q,Motor_Speed,Torque,I_D,I_Q,PM,Stator_Yoke,Stator_Tooth,Stator_Winding,Profile_ID) 
        st.success('Your Temp is {}'.format(result))
        #print(LoanAmount)

     #prediction = classifier.predict( 
      #  [[ambient,coolant,U_D,U_Q,Motor_Speed,Torque,I_D,I_Q,PM,Stator_Yoke,Stator_Tooth,Stator_Winding,Profile_ID]])
   
     
if __name__=='__main__': 
     main()
