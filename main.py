from fastapi import FastAPI,Body
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import time
from moviepy.editor import *
from datetime import datetime    
import pytz    
tz_NY = pytz.timezone('Asia/Kolkata')  

import smtplib
from uvicorn import run
import os

app = FastAPI()

origins = ["*",'https://3000-lovepreetk8-tadfrontend-wj2jqh7y8nz.ws-us97.gitpod.io']
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

from firebase_admin import credentials, initialize_app, storage
cred_file_path = "/etc/secrets/tadbml-firebase-adminsdk-a54ib-b3bca638e0.json"
cred = credentials.Certificate(cred_file_path)
initialize_app(cred)

model_path = "./tf_lite_model.tflite"
interpreter = tf.lite.Interpreter(model_path)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.resize_tensor_input(input_details[0]['index'], (1, 250, 250,3))
interpreter.resize_tensor_input(output_details[0]['index'], (1, 2))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_frame(img):
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img_batch)
    interpreter.invoke()
    tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
    prediction=(tflite_model_predictions > 0.5).astype("int32")
    if(prediction[0][0]==0):
        print(tflite_model_predictions)
        return("Accident Detected")
    else:
        return("No Accident")
    
def base64_to_image(base64_string):

    """
    The base64_to_image function accepts a base64 encoded string and returns an image.
    The function extracts the base64 binary data from the input string, decodes it, converts 
    the bytes to numpy array, and then decodes the numpy array as an image using OpenCV.    
    :param base64_string: Pass the base64 encoded image string to the function
    :return: An image
    """

    base64_data = base64_string.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


bucket = storage.bucket('tadbml.appspot.com')


@app.get("/")
async def root():
    return {"message": "Welcome to the Food Vision API!"}

@app.post("/r_accident_detection/")
async def rad(data=Body()):
    image=data['image']
    image=base_64_image(image)
    resized_frame=tf.keras.preprocessing.image.smart_resize(image, (250, 250), interpolation='bilinear')
    result=predict_frame(resized_frame)
    return {
        "result":result
        }

@app.post("/accident_detection/")
async def get_net_image_prediction(data=Body()):
    footage_video=data['footage_video'] 
    email_id=data['email_id']
    nlocation=data['location']
    glocation=data['glocation']
    if(nlocation!=""):
        nlocation = "\nNear by location: "+nlocation
    if(glocation!=""):
        glocation = "\nLocation on Google maps: "+glocation
    if footage_video == "":
        return {"message": "No video provided"}
    if email_id == "":
        return {"message": "No email id provided"}
    
    c=1
    receivers=[email_id]
    s = smtplib.SMTP("smtp.gmail.com", 587)
    s.ehlo()
    s.starttls()
    s.ehlo()

    s.login("ttmklg1by0@gmail.com", "bitnhdlleftszngg")

    sender = "ttmklg1by0@gmail.com"

    image=[]
    label=[]
    result=[]
    cap = cv2.VideoCapture(footage_video)
    fps = cap.get(cv2. CAP_PROP_FPS)
    filename = time.strftime("%Y%m%d%H%M%S",time.localtime())+".mp4"
    clip = VideoFileClip(footage_video)
    start=time.time()
    limit = 0
    acc_time = ""
    id = ""
    ids=[]
    acc_times=[]
    while True:
        grabbed, frame = cap.read()
        c=c+1
        key_time=5
        if c%fps==0 and grabbed:
            c=0
       # print(c)
            resized_frame=tf.keras.preprocessing.image.smart_resize(frame, (250, 250), interpolation='bilinear')
            key_time = key_time + time.time()-start
            label.append(predict_frame(resized_frame))
          #  image.append(frame)
            
            
        end=time.time()-start+5
        if end>clip.duration-2:
           break
    #print(label)
        if label.count("Accident Detected")>limit:
            limit = limit+1
            clip1=clip.subclip(max(key_time-7,0),min(key_time,clip.duration))
            clip1.write_videofile(filename)
            blob = bucket.blob(('accident/'+filename))
            blob.upload_from_filename(filename)
            blob.make_public()
            video_link=blob.public_url
            result.append(video_link)
            print("your file url", video_link)
            datetime_NY = datetime.now(tz_NY)  
            acc_time = datetime_NY.strftime("%Y-%m-%d %H:%M")
            id = time.strftime("%Y%m%d%H%M%S",time.localtime())
            acc_times.append(acc_time)
            ids.append(id)
            subject = "Accident Detected at India time:"+ acc_time 
            text ="Accident Details \nMode: Recorded Video\nDate & Time: "+datetime_NY.strftime("%Y-%m-%d %H:%M:%S.%f")+"\nClip link: " +video_link +nlocation+glocation+"\nClip Id: "+id
            message = """From: %s\nTo: %s\nSubject: %s\n\n%s
            """ % (sender, ", ".join(receivers), subject, text)
            s.sendmail(sender, receivers, message)
            filename='./'+filename
            os.remove(filename)
            filename= id+".mp4"
            fps=fps*5
            while c<fps:
                grabbed, frame = cap.read()
                c=c+1
            fps=fps/5
            c=0
    cap.release()
    s.quit()
    
    return {
        "Result":result,
        "Id":ids,
        "acc_time":acc_times
    }

if __name__ == "__main__":
     port = int(os.environ.get('PORT', 5000))
     run(app, host="0.0.0.0", port=port)
