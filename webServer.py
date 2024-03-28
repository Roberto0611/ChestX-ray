from flask import Flask,render_template,request
from werkzeug.utils import secure_filename
import os

app = Flask("webServer")

app.config["UPLOAD_FOLDER"] = 'static/uploads'

ALLOWED_EXTENSIONS = set(['png',"jpg","jpeg"])

def allowed_file(file):
    file = file.split('.') # Esto lo convertira en una lista el indice 0 es el nombre y el 1 es el formato
    if file[1] in ALLOWED_EXTENSIONS:
        return True
    else:
        return False

@app.route('/upload',methods=["POST"])
def upload():
    file = request.files.get('uploadFile')
    print(file,file.filename)
    filename = secure_filename(file.filename)
    print(filename)
    if file and allowed_file(filename):
        print('permitido')
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        return render_template('upload_file.html',mensaje="Archivo subido exitosamente :)")
    return render_template('upload_file.html',mensaje="no se subio el archivo :(")

    return 's'
@app.route("/")
def hello_world():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port='5000')
