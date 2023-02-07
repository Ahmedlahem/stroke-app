import joblib
import lahm_preprocess
from flask import Flask , render_template , request
import numpy as np

app = Flask(__name__)

scaler = joblib.load('Models/scaler.h5')
model = joblib.load('Models/rf_clf.h5')

Gender=['Female','Male','Other']
Ever_Married=['No','Yes']
Work_Type=['Govt_job','Never_worked','Private','Self-employed','children']
Residence_Type=['Rural','Urban']
Smoking_Status=['Unknown','formerly smoked','never smoked','smokes']
@app.route('/')
def index() :
    return render_template('index.html')
@app.route('/predict' , methods = ['GET'])
def get_prediction():
        inp_data=[int(request.args.get('Age'))
        ,int(request.args.get('Hypertension'))
        ,int(request.args.get('Heart_Disease'))
        ,float(request.args.get('Avg_Glucose_Level'))
        ,float(request.args.get('BMI'))
        ]
        gender_dummies=[0 for i in range(3)]
        marry_dummies=[0 for i in range(2)]
        work_dummies=[0 for i in range(5)]
        residence_dummies=[0 for i in range(2)]
        smoking_dummies=[0 for i in range(4)]
        
        try :
             gender_dummies[Gender.index(request.args.get('Gender'))] = 1
        except :
             pass     
        
        try :
             marry_dummies[Ever_Married.index(request.args.get('Ever_Married'))] = 1
        except :
             pass  
        
        try :
             work_dummies[Work_Type.index(request.args.get('Work_Type'))] = 1
        except :
             pass        
        
        try :
             residence_dummies[Residence_Type.index(request.args.get('Residence_Type'))] = 1
        except :
             pass     
        
        try :
             smoking_dummies[Smoking_Status.index(request.args.get('Smoking_Status'))] = 1
        except :
             pass     
        
        inp_data += gender_dummies
        inp_data += marry_dummies
        inp_data += work_dummies
        inp_data += residence_dummies
        inp_data += smoking_dummies

        #inp_data=[int(n) for n in inp_data]
        stroke=model.predict(scaler.transform([inp_data]))[0]
        if stroke==0:
            stroke= 'No stroke' 
             
        else:
            stroke= 'Stroke'


        return render_template('prediction.html', stroke = str(stroke))

   

if __name__ == '__main__':
    app.run(debug = True)