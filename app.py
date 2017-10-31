import os
import subprocess

from flask import Flask, request, render_template, make_response, Response
from werkzeug.utils import secure_filename
from flask import jsonify

from PIL import Image, ImageFilter

from matching import matches
import database
from orientation import fix_orientation
from authenticate import basicauth

from prediction import pred

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'pdf',
                      'png',
                      'jpg',
                      'JPG',
                      'jpeg',
                      'gif',
                      'tif',
                      'tiff'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # else, return 413


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def cleanup(*args):
    for i in args:
        os.remove(i)


@app.route("/", methods=['GET'])
def listpublications():
    """
    Lists the available target publications on database.
    :return: template
    """
    db = database.Database()
    result = db.availableidentifiers()
    return render_template("list.html", list=result)


@app.route('/book/<int:isbn>', methods=['GET'])
def main(isbn):
    """
    Main application logic. See each IF for info.

    :type isbn int
    :param isbn Publication identifier
    """
    if request.method == 'GET':
        # Show an upload form for the target publication at the required page number.

        db = database.Database()
        result = db.querydocument(isbn)
        #return jsonify(isbn=isbn,similarity=result['contents'])
        if not result:
            return make_response(render_template("error.html"), 404)
        return render_template("single.html", book=result)


@app.route("/new/", methods=["GET", "POST"], defaults={"isbn": None})
@app.route("/new/<int:isbn>", methods=["POST", "GET"])
@basicauth(username="user", password="pass")  # credentials should be fetched somewhere else...
def create_resource(isbn):
    """
    An alterative route to insert target contents.
    The html form accepts a picture of the page and a page number
    """
    if request.method == "GET":
        if isbn is not None:
            return """<!doctype html>
                        <title>Upload new File</title>
                        <h1>Upload new File</h1>
                        <form action="/new/" method=post enctype=multipart/form-data>
                        <input type=hidden name={isbn}>
                        <input type=text name=page value=PAGE>
                        <p><input type=file name=file accept="image/*" capture="camera">
                        <input type=submit value=Upload>
                        </form>""".format(isbn=isbn)
        else:
            return """<!doctype html>
                        <title>Upload new File</title>
                        <h1>Upload new File</h1>
                        <form action="" method=post enctype=multipart/form-data>
                        <input type=text name=isbn value=ISBN>
                        <input type=text name=page value=PAGE>
                        <p><input type=file name=file accept="image/*" capture="camera">
                        <input type=submit value=Upload>
                        </form>"""

    if request.method == "POST":
        # Resource creation.
        # Proper REST dialects have a PUT for this, but HTML forms only POST

        if request.form["isbn"] and request.files['file'] and request.form["page"]:
            # the same logic as POST /book/isbn
            # should be moved to a function

            sent_file = request.files["file"]
            isbn = request.form["isbn"]
            page = request.form["page"]
            filename = secure_filename(os.tempnam("src_")+sent_file.filename)
            sent_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # fix_orientation(os.path.join(app.config['UPLOAD_FOLDER'], filename), save_over=True)

            # img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)).convert('LA')
            # img = img.filter(ImageFilter.DETAIL)
            # img = img.filter(ImageFilter.SHARPEN)
            # img_name = os.tempnam("uploads/", "img_")+".png"
            # img.save(img_name)
            # temp = os.tempnam("uploads/", "tess_")
            # command = ["tesseract", img_name, temp, "-l ita"]

            # try:
            #     ocr = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # except subprocess.CalledProcessError:
            #     return make_response(render_template("error.html"), 500)

            # ocr.wait()

            # with open(temp+".txt") as g:
            #     source = g.readlines()

            source = str("%.2f") %pred(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #source = u"precision: 0.9"
            db = database.Database()
            rValue = db.insertreferencepage(isbn, page, source)
            #cleanup(img_name, temp+".txt", os.path.join(app.config['UPLOAD_FOLDER'], filename))
            if rValue:
                return Response("Inserted book number:{isbn}, \
                                for page {page}, \
                                with contents {source}".format(isbn=isbn, page=page, source=source), 201)
            else:
                return make_response(render_template("error.html"), 500)
        else:
            return make_response(render_template("error.html"), 403)
    if request.method == "DELETE":
        # RESTful APIs should pair a delete with each put/post
        # Not yet implemented, since I can't bother to write a XHR to make the DELETE request from a html page
        return Response("Not supported", 501)
