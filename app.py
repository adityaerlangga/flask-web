from flask import Flask, render_template, request
from keras.models import load_model
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = load_model("Input/covid.h5")

model.make_predict_function()

def predict_label(img_path):
	img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
	img = tf.keras.preprocessing.image.img_to_array(img)/255
	img = img.reshape(-1, 150, 150, 3)
	p = model.predict(img)
	prediction = float(p)
	if prediction > 0.5:
		result = 'Covid'
	else:
		result = 'Normal'
	return result


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "Input/" + img.filename
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(port=3000, debug = True)