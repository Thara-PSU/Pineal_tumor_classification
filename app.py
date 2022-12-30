from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import streamlit as st
#import keras


def main():
    from PIL import Image
    image_ban = Image.open('images/image2.png')
    st.image(image_ban, use_column_width=False)

if __name__ == '__main__':
        main()
st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('./model.hdf5')
	return model


def predict_class(image, model):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [180, 180])

	image = np.expand_dims(image, axis = 0)

	prediction = model.predict(image)

	return prediction


model = load_model()
st.title('Classification between Germinoma and Non-germinoma using Deep learning')

file = st.file_uploader("Upload an image of an MRI", type=["jpg", "png"])

if st.button("Push for classification", key='classify'):
	if file is None:
		st.text('Waiting for upload....')

	else:
		slot = st.empty()
		slot.text('Running inference....')

		test_image = Image.open(file)

		st.image(test_image, caption="Input Image", width = 400)
		#prediction = predict_class(model, my_image)
		pred = predict_class(np.asarray(test_image), model)

		class_names = ['Germinoma', 'Non_germinoma']

		result = class_names[np.argmax(pred)]

		output = 'The image is a ' + result

		slot.text('Done')

		st.success(output)