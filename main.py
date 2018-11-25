"""
Project:      Equirectangular Camera Mapping
Name:         main.py
Date:         November 25th, 2018.
Author:       Thomas Bellucci
Description:  This file contains a Flask application which allows the user to
              construct 3D geometry from equirectangular / mercator projected
              images obtained by a 360 camera.
"""

from flask import Flask, render_template


app = Flask(__name__)


@app.route('/')
def page():
    return render_template('application.html')


if __name__ == '__main__':
    app.run()