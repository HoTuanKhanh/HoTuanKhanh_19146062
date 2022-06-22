import cv2
import numpy as np 
import time
from tkinter import *
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image,ImageTk
from cProfile import label
import os
from pickle import FROZENSET
from tkinter import filedialog
import tkinter as tk
from matplotlib.image import thumbnail
import matplotlib.pyplot as plt

mango_model = load_model('best_model_mango.h5')
mango_labels=['Khong_khuyet_tat','Thoi_trai','Than_thu']
mango = ""

#Khoi tao giao dien gui 
tk=Tk() 
tk.title("Recognize Defects Mango") 
tk.geometry("1000x600+0+0") 
tk.resizable(0,0) 
tk.configure(background="white") 
content1="Recognize Defects Mango"
lb01=Label(tk,fg="green",bg="white",font="Times 18",text=content1) 
lb01.pack()
lb01.place(x=290,y=10) 

#Hien thi ten khung hinh
lb04=Label(tk,text="Recognize",font="Times 12",fg="blue",bg="white")
lb04.pack()
lb04.place(x=180,y=40)

##### Hien thi so luong loai### 
lb13=Label(tk,fg="green",bg="white",font="Times 18",text="Mango:   ") 
lb13.pack()
lb13.place(x=40,y=400) 

#Khoi tao camera 
video = cv2.VideoCapture(0) 

def close_window():
    tk.destroy()

def ConvertImage(convert_img):
    image = convert_img[:,80:(80+480)]
    image = cv2.resize(image, dsize =(150,150))
    image = np.expand_dims(image, axis=0)
    return image

def Regconition(reg_img):
    #Shape Face
    mango_predict = mango_model.predict(reg_img)
    mango_label= mango_labels[np.argmax(mango_predict)]
    return mango_label

def showimage():
    global fln
    fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image File", filetypes=(("JPG File",".jpg"),("PNG File",".png"), ("ALL Files",".")))
    img = Image.open(fln)
    img.thumbnail((400,400))
    img = ImageTk.PhotoImage(img)
    lbl3.configure(image= img)
    lbl3.image = img

def recognize():
    global lbl1
    global lbl2
    img_path= fln
    img=plt.imread(img_path)
    print ('Input image shape:', img.shape)
    img=cv2.resize(img, (150,150))
    print ('the resized image has shape:', img.shape)
    plt.axis('off')
    plt.imshow(img)
   
    img=np.expand_dims(img, axis=0)
    print ('image shape after expanding dimensions:',img.shape)
    pred=mango_model.predict(img)
    print ('the shape of prediction:', pred.shape)
    index=np.argmax(pred[0])
    klass=mango_labels[index]
    probability=pred[0][index]*100-1.2
    print(f'the image recognized is: {klass} with a probability of {probability:6.2f} %')
    lbl1 = Label(tk,text = f"Nhận diện là: {klass}" , fg= "red", font=("Arial", 20))
    lbl1.pack(pady= 20)
    lbl1.place(x=590,y=400)
    lbl2 = Label(tk,text = f"Độ chính xác: {probability:6.2f} %" , fg= "blue", font=("Arial", 20))
    lbl2.pack(pady= 20)
    lbl2.place(x=590,y=450)
    return

def clear():
    lbl1.after(1000, lbl1.destroy())
    lbl2.after(1000, lbl2.destroy())
    return

lbl3 = Label(tk)
lbl3.pack()
lbl3.place(x=590, y=70)

btn1 = Button(tk,text = "Select Image", command= showimage)
btn1.pack()
btn1.place(x=590,y=550)

btn2 = Button(tk,text = "Recognize", command= recognize)
btn2.pack()
btn2.place(x=690,y=550)

btn3 = Button(tk,text = "Clear", command= clear )
btn3.pack()
btn3.place(x=790,y=550)


while video.isOpened():
    ret, image_mango = video.read() 
    cv2.imwrite('image_mango.jpg',image_mango) 
    imagelg=Image.open('image_mango.jpg')
    imagelg=imagelg.resize((400,300),Image.ANTIALIAS) 
    imagelg=ImageTk.PhotoImage(imagelg) 
    lb05=Label(image=imagelg)
    lb05.image=imagelg 
    lb05.pack()  
    lb05.place(x=30,y=70)
    tk.update()
    image = ConvertImage(image_mango)
    mango = Regconition(image)   
    lb23=Label(tk,fg="green",bg="white",font="Times 18",text = mango) 
    lb23.pack()
    lb23.place(x=150,y=400)
    mango = ""
    
    if cv2.waitKey(1) == ord('q'):
        break
video.release() 
cv2.destroyAllWindows() 

tk.mainloop()