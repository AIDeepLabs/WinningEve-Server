# %%
import torch
import numpy as np
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from flask import request, abort
from collections import defaultdict
import bcrypt,datetime
import secrets
import os

# %%
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

# %%
model.eval()

# %%
from PIL import Image
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# %%
image=Image.open('images/dog.jpeg')
image

# %%
# preprocess(torch.Tensor(np.array(image)))
# A quick test
# preprocess(torch.Tensor())
# trans = transforms.ToPILImage()
item=preprocess(image).unsqueeze(0)
# preprocess(image).shape
output=model(item)
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)


# %%
# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

# %% [markdown]
# ## prediction 

# %%
import cv2
def predict(image):
    prob=defaultdict(float)
    with torch.no_grad():
        output=model(preprocess(image).unsqueeze(0))
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        prob[categories[top5_catid[i]]]=top5_prob[i].item()
    return prob


def predict_video(video):
    cap=cv2.VideoCapture(video)
    count=0
    fps=int(cap.get(cv2.CAP_PROP_FPS))
    predictions=[]
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            count+=1
            if count%int(fps*app.config["PER_FRAME"])==0:
                colourCorrected=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prob=predict(Image.fromarray(colourCorrected))
                predictions.append({"frame":count, "probabilities":prob})
        else:
            cap.release()
            break
    return predictions

# %% [markdown]
# ## Attach MongoDB

# %%
from pymongo import MongoClient
# pprint library is used to make the output look more pretty
from pprint import pprint
# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient("mongodb://root:root@mongodb:27017")
db=client.winning_eve
client.server_info()


# %% [markdown]
# ## Server flask start

# %% [markdown]
# ### Helper functions

# %%
# Check if API is valid
def is_api_valid(api_key,db):
    print(api_key,db)
    return db.users.find_one({"apiKey":api_key})

# %%
def get_hashed_password(plain_text_password):
    # Hash a password for the first time
    #   (Using bcrypt, the salt is saved into the hash itself)
    return bcrypt.hashpw(plain_text_password, bcrypt.gensalt())

def check_password(plain_text_password, hashed_password):
    # Check hashed password. Using bcrypt, the salt is saved into the hash itself
    return bcrypt.checkpw(plain_text_password, hashed_password)

def get_duration_and_fps(filename):
    video = cv2.VideoCapture(filename)

    # duration = video.get(cv2.CAP_PROP_POS_MSEC)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps

    return {"duration":duration, "frame_count":frame_count}

# %%
# %% [markdown]
# ### Main server code

# %%
from bson.objectid import ObjectId
import hashlib
import matplotlib.pyplot as plt
from flask_cors import CORS, cross_origin
app = Flask(__name__)
#CORS(app)
app.config.update(
    BUFFER= 262144,
    TMP_DIR= '/tmp',
    RANDOM_BYTES= 32,
    MAX_VIDEO_LENGTH= 60*3,
    PER_FRAME= 0.5,
    FREE_IP_LIMIT={'images':10000,'videos':10},
    MEMBER_LIMIT={'images':10000,'videos':10}
)


@app.route('/detectObjectsInImage',methods=['POST'])
def detectObjectsInImage():

    plt.imshow(Image.open(request.files['image']))
    plt.show()
    # check if image in request
    if not 'image' in request.files:
        return jsonify({"status":"failure","error":"No image found in request"})
    
    serverSettings=db.settings.find_one()
    ipFreeQuota=app.config['FREE_IP_LIMIT']
    memberQuota=app.config['MEMBER_LIMIT']
    
    # Check if user is valid and has credit
    user=None
    if 'apiKey' in request.headers:
        user=is_api_valid(request.headers['apiKey'],db)
        if user!=None: 
            if 'imagesProcessed' in user and user['imagesProcessed']>memberQuota['images']:
                return jsonify({"status":"failure","error":"You have exceeded your quota"})
            else:
                user['imagesProcessed']+=1
                db.users.update_one({"_id":user['_id']},{'$set':{"imagesProcessed":user['imagesProcessed']}})
        else:
            return jsonify({"status":"failure","error":"Invalid API key"})

    # Check ip quota
    else:
        
        request_ip=request.headers['X-Forwarded-For']
        
        ip=db.ips.find_one({"ip":request_ip})
        if ip!=None:
            if ip['imagesProcessed']>ipFreeQuota['images']:
                return jsonify({"status":"failure","error":"You have exceeded your quota"})
            else:
                ip['imagesProcessed']+=1
                db.ips.update_one({"_id":ip['_id']},{'$set':{"imagesProcessed":ip['imagesProcessed']}})
        else:
            db.ips.insert_one({"ip":request_ip,"imagesProcessed":1,"videosProcessed":0})

    # if apiKey!=None ]:



    # Get the image from post request
    image=Image.open(request.files['image'])
    
    # Check if we have predicted the image before using a hash
    m=hashlib.sha512()
    m.update(image.tobytes())
    image_hash=m.hexdigest()

    if db.predictions.find_one({'hash':image_hash,'type':'image'}):
        # In the future check if the prediction is still valid by looking at the time
        previous_prediction=db.predictions.find_one({'hash':image_hash})
        # Return the prediction
        print(previous_prediction["prediction"])
        return jsonify(previous_prediction["prediction"])
    else:
        # Otherwise predict the image using GPU power
        prediction=predict(image)
        # Save the prediction in the database
        db.predictions.insert_one({'type':'image','hash':image_hash,'prediction':prediction,'dateCreated':datetime.datetime.utcnow()})
        # Return the prediction
        return prediction

@app.route('/createUser',methods=["POST"])
def createUser():
    results=db.users.find_one({"username":request.form['username']})
    emailCheck=db.users.find_one({"email":request.form['email']})
    if emailCheck!=None:
        return jsonify({"status":"failure","error":"Email already exists"})
    generated_api=secrets.token_urlsafe(app.config["RANDOM_BYTES"])
    if results==None:
        db.users.insert_one({
            "username":request.form['username'],
            "password":get_hashed_password(request.form['password'].encode('utf-8')),
            "apiKey": generated_api,
            "email":request.form['email'],
            "dateCreated":datetime.datetime.utcnow(),
            "imagesProcessed":0,
            "videosProcessed":0
            })
        return jsonify({"status":"success","apiKey":generated_api})
    else:
        return jsonify({"status":"failure","error":"Username already exists"})

@app.route('/deleteUser',methods=["POST"])
def deleteUser():
    done=db.users.delete_one({"_id":ObjectId(request.form['id'])})
    if done.acknowledged:
        return jsonify({"status":"success"})
    else:
        return jsonify({"status":"failure"})

@app.route('/getQuota',methods=["POST"])
def getQuota():
    serverSettings=db.settings.find_one()
    ipFreeQuota=app.config['FREE_IP_LIMIT']
    memberQuota=app.config['MEMBER_LIMIT']
    if 'apiKey' in request.headers:
        user=is_api_valid(request.headers['apiKey'],db)
        if user!=None: 
            return jsonify({"status":"success","image-quota":memberQuota["images"]-user['imagesProcessed'],"video-quota":memberQuota["videos"]-user['videosProcessed']})
        else:
            return jsonify({"status":"failure","error":"Invalid API key"})
    else:
        request_ip=request.headers['X-Forwarded-For']
        ip=db.ips.find_one({"ip":request_ip})
        if ip!=None:
            return jsonify({"status":"success","image-quota":ipFreeQuota["images"]-ip['imagesProcessed'],"video-quota":ipFreeQuota["videos"]-ip['videosProcessed']})
        else:
            db.ips.insert_one({"ip":request_ip,"imagesProcessed":0,"videosProcessed":0})
            return jsonify({"status":"success","image-quota":ipFreeQuota["images"],"video-quota":ipFreeQuota["videos"]})

@app.route('/detectObjectsInVideo',methods=["POST"])
def detectObjectsInVideo():
    serverSettings=db.settings.find_one()
    ipFreeQuota=app.config['FREE_IP_LIMIT']
    memberQuota=app.config['MEMBER_LIMIT']
    if not 'video' in request.files:
        return jsonify({"status":"failure","error":"No video found in request"})

    

    # Check if user is valid and has credit
    user=None
    if 'apiKey' in request.headers:
        user=is_api_valid(request.headers['apiKey'],db)
        if user!=None: 
            if 'videosProcessed' in user and user['videosProcessed']>memberQuota['videos']:
                return jsonify({"status":"failure","error":"You have exceeded your quota"})
            else:
                user['videosProcessed']+=1
                db.users.update_one({"_id":user['_id']},{'$set':{"videosProcessed":user['videosProcessed']}})
        else:
            return jsonify({"status":"failure","error":"Invalid API key"})

    # Check ip quota
    else:
        
        request_ip=request.headers['X-Forwarded-For']
        
        ip=db.ips.find_one({"ip":request_ip})

        if ip!=None:
            if ip['videosProcessed']>ipFreeQuota['videos']:
                return jsonify({"status":"failure","error":"You have exceeded your quota"})
            else:
                ip['videosProcessed']+=1
                db.ips.update_one({"_id":ip['_id']},{'$set':{"videosProcessed":ip['videosProcessed']}})
        else:
            db.ips.insert_one({"ip":request_ip,"imagesProcessed":0,"videosProcessed":1})

    
    # Save video for prediction then delete the video
    video_tmp_name=secrets.token_urlsafe(app.config["RANDOM_BYTES"])
    # Get the extention
    # extention=request.files['video'].filename.split('.')[-1]
    # Location of the video
    video_location=os.path.join(app.config["TMP_DIR"],video_tmp_name)
    # Save the video
    request.files['video'].save(video_location)
    # Check if the video is larger than the max size
    video_length=get_duration_and_fps(video_location)
    print(video_length)
    if video_length['duration']>app.config["MAX_VIDEO_LENGTH"]:
        return jsonify({"status":"failure","error":f"Video is too long, it should be smaller than {app.config['MAX_VIDEO_LENGTH']} seconds"})
    # Create a hash of the video
    # This is too slow so using sha512
    # video_hash=VideoHash(path=video_location).hash_hex
    m=hashlib.sha512()
    with open(video_location, 'rb') as f:
        while True:
            data = f.read(app.config["BUFFER"])
            if not data:
                break
            m.update(data)
    video_hash=m.hexdigest()


    # Check if we have predicted the video before
    previous_prediction=db.predictions.find_one({'hash':video_hash,'type':'video'})
    if previous_prediction!=None: 
        # If we have predicted the video before return the prediction
        return jsonify({"status":"success","prediction":previous_prediction["prediction"]})
    # Otherwise predict the video
    # Predict whats inside the video
    prediction=predict_video(video_location)
    # Delete the video
    os.remove(video_location)
    # Save the prediction in the database
    db.predictions.insert_one({'type':'video','hash':video_hash,'prediction':prediction,'dateCreated':datetime.datetime.utcnow()})
    # Return the prediction
    return jsonify({"status":"success","prediction":prediction})
    
app.run(host='0.0.0.0', port=9000)

# %%
# !pip install -U flask-cors


