import os
import tkinter as tk
from tkinter import filedialog as fd, NW, RAISED
from tkinter import ttk
import numpy as np
from PIL import ImageTk, Image

# create the root window
from makePrediction import predict_binary_classification, predict_isup

root = tk.Tk()
root.title('Detect prostate cancer')
root.resizable(False, False)
root.geometry('900x600')
root.configure(bg='sky blue')
selected_path = ""


def put_image(img):
    render = ImageTk.PhotoImage(img)
    i = tk.Label(image=render, width=512, height=512)
    i.image = render
    i.place(x=110, y=10)


def select_file():
    filetypes = (
        ('PNG', '*.png'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir=os.path.join(os.getcwd(), ".."),
        filetypes=filetypes)

    img = Image.open(filename)
    global selected_path
    selected_path = filename
    put_image(img)
    result_button["state"] = "enable"


def get_result():
    print(selected_path)
    result = predict_binary_classification(selected_path)
    if result[0][0] < 0.5:
        label.config(text='NEGATIVE', fg='black', font=("Arial", 25))
    else:
        label.config(text='POSITIVE', fg='red', font=("Arial", 25))

    isup = predict_isup(selected_path)
    idx = np.where(isup == np.amax(isup))
    # label1.config(text=idx[0])


s = ttk.Style()
s.configure('first.TFrame',background='midnight blue')

# open button
open_button = ttk.Button(
    root,
    text='Open file',
    command=select_file
)
# open_button.configure(style='first.TFrame')
open_button.place(x=10, y=230)

# result button
result_button = ttk.Button(
    root,
    text='Get result',
    command=get_result
)
result_button["state"] = "disabled"
# result_button.configure(style='first.TFrame')
result_button.place(x=10, y=260)

label = tk.Label(root, textvariable='')
label.place(x=700, y=245)
label1 = tk.Label(root, textvariable='')
# label1.place(x=700, y=295)
root.mainloop()
