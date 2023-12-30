
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask , request, render_template


app = Flask(__name__)

model = load_model("EyeModel.h5",compile=False)
                 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (128,128)) 
        x = image.img_to_array(img)
        print(x)
        x = np.expand_dims(x,axis =0)
        print(x)
        pred =np.argmax(model.predict(x))
        print("prediction",pred)
        index = {0:'Cataract',1:'Diabetic Retinopathy',2:'Glaucoma',3:'Normal'}
        text = "The classified Disease is : " + str(index[pred])
    return text

if __name__ == '__main__':
    app.run(debug = False, threaded = False)
