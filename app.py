from flask import Flask, render_template ,request, send_from_directory, Response
from flask import jsonify
from flask_bootstrap import Bootstrap
import json
import re
import os
import scipy.misc
import warnings
import sys
import compare_image
import time
import detect_face
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import numpy
from prometheus_flask_exporter import PrometheusMetrics



app = Flask(__name__)
metrics = PrometheusMetrics(app)
if metrics:
    print("metrics!!!")
Bootstrap(app)

TOO_BIG_SIZE = 300
UPLOAD_FOLDER = "static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENTIONS = set(['png', 'jpg', 'jpeg'])
app.secret_key = 'secret'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENTIONS


@app.route("/", methods=['GET', 'POST'])
def home_page():
    if request.method == "POST":
        print("request.files['target'].filename: ", request.files['target'].filename)
        if request.files['target'].filename != '' and request.files['face'].filename != '':
            target = request.files['target']
            face =  request.files['face']
            
            try:
                target_image = Image.open(target.stream)
            except:
                return render_template('web.html', msg='Cannot read anh vo')

            try:
                face_image = Image.open(face.stream)
            except:
                return render_template('web.html', msg='Cannot read anh chong')

            # Scale down image if it's giant so things run a little faster
            print("target image size: ", target_image.size)
            if target_image.size[0] > TOO_BIG_SIZE:
                print("target image is too big, converting")
                target_new_width = TOO_BIG_SIZE
                target_new_height = int(round((TOO_BIG_SIZE / target_image.size[0]) * target_image.size[1], 0))
                target_image = target_image.resize ((target_new_width, target_new_height))

            target_image.save(os.path.join(app.config['UPLOAD_FOLDER'], target.filename))
            print("done saving target image", target.filename)
            target_image_src=os.path.join(app.config['UPLOAD_FOLDER'], target.filename)


            # Scale down image if it's giant so things run a little faster
            print("face image size: ", face_image.size)
            if face_image.size[0] > TOO_BIG_SIZE:
                print("face image is too big, converting")
                face_new_width = TOO_BIG_SIZE
                face_new_height = int(round((TOO_BIG_SIZE / face_image.size[0]) * face_image.size[1], 0))
                face_image = face_image.resize ((face_new_width, face_new_height))

            face_image.save(os.path.join(app.config['UPLOAD_FOLDER'], face.filename))
            print("done saving face image", face.filename)
            face_image_src=os.path.join(app.config['UPLOAD_FOLDER'], face.filename)


            if target_image != face_image:
                start = time.time()
                similarity, perfect_similarity, status = compare_image.main(target_image_src,face_image_src)
                end=time.time()
                time_taken = round(end-start,1)
            else:
                similarity = 100
                perfect_similarity = 100
                status = "ok"
                time_taken = "1"

            target_filename=secure_filename(target.filename)
            face_filename=secure_filename(face.filename)
            return render_template("web.html", target_image=target.filename, face_image=face.filename, time_taken=time_taken, similarity=similarity, perfect_similarity=perfect_similarity, status=status)

            
        
        else:
            return render_template('web.html', msg='Not enough pics')
    else:
        return render_template('web.html')



@app.route('/api/v1/compare_faces', methods=['POST'])
def compare_faces():
    target = request.files['target']
    faces =  request.files.getlist("faces")
    target_filename=secure_filename(target.filename)
    response=[]
    for face in faces:
        start = time.time()
        distance,result = compare_image.main(target,face)
        end=time.time()
        json_contect={
                'result':str(result),
                'distance':round(distance,2),
                'time_taken':round(end-start,3),
                'target':target_filename,
                'face':secure_filename(face.filename)
            }
        response.append(json_contect)
    python2json = json.dumps(response)
    return app.response_class(python2json, content_type='application/json') 



@app.route('/api/v1/detect_faces', methods=['POST'])
def detect_faces():
    faces =  request.files.getlist("faces")
    # target_filename=secure_filename(target.filename)
    response=[]
    for face in faces:
        start = time.time()
        _,result = detect_face.get_coordinates(face)
        end=time.time()
        json_contect={
                'coordinates':result,
                'time_taken':round(end-start,3),
                'image_name':secure_filename(face.filename)
            }
        response.append(json_contect)
    python2json = json.dumps(response)
    return app.response_class(python2json, content_type='application/json') 

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=8000)
