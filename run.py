from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tkinter import *
from PIL import Image, ImageOps
import tkinter as tk
import numpy as np

model = load_model('digits.h5')
print('Model Loaded Successfully.')

line_id = None
line_points = []
line_options = {'width': 5, 'fill': 'black'}

def draw_line(event):
    global line_id
    line_points.extend((event.x, event.y))
    if line_id is not None:
        canvas.delete(line_id)
    line_id = canvas.create_line(line_points, **line_options)

def set_start(event):
    line_points.extend((event.x, event.y))

def end_line(event=None):
    global line_id
    line_points.clear()
    line_id = None

def preprocess_image(img_path):
    img = image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
    img = image.img_to_array(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def save_and_predict(canvas, fileName):
    # Save the canvas content as an EPS file
    canvas.postscript(file=fileName + '.eps')
    # Open the EPS file and convert it to PNG
    img = Image.open(fileName + '.eps')
    img.save(fileName + '.png', 'png')

    # Preprocess the image
    img = preprocess_image(fileName + '.png')

    # Make a prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    
    # Update the label with the prediction
    result_label.config(text=f'Predicted class: {predicted_class[0]}')

def clear_canvas():
    canvas.delete('all')
    result_label.config(text='')

root = tk.Tk()

canvas = tk.Canvas(root, width=400, height=400, bg='white')
canvas.pack()

canvas.bind('<Button-1>', set_start)
canvas.bind('<B1-Motion>', draw_line)
canvas.bind('<ButtonRelease-1>', end_line)

submit_btn = Button(root, text='Submit', width=5, height=3, bd='10', command=lambda: save_and_predict(canvas, 'drawing'))
submit_btn.place(x=150, y=300)

clear_btn = Button(root, text='Clear', width=5, height=3, bd='10', command=clear_canvas)
clear_btn.place(x=230, y=300)

result_label = Label(root, text='', font=('Helvetica', 12))
result_label.place(x=150, y=350)

root.mainloop()
