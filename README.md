**IMAGE PROCESSING**<br>
**1. Develop a program to display grayscale image using read and write operation.**<br>
pip install opencv-python<br>
import cv2<br>
img=cv2.imread('flower5.jpg',0)<br>
cv2.imshow('image',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940850/174041726-df4b96be-11b2-4f5a-b4b9-aee932a7be26.png)<br>
<br>
<br>
**2. Develop a program to display the image using matplotlib.**<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('plant4.jpg')<br>
plt.imshow(img)<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940850/174042162-a75f11ae-0ea3-4885-ae30-ff9720232530.png)<br>
<br>
<br>
**3. develop a program to perform linear transformation. Rotation**<br>
import cv2<br>
from PIL import Image<br>
img=Image.open("plant4.jpg")<br>
img=img.rotate(180)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940850/174042633-31ed2a88-33e8-4f3e-9a90-0a1a721187b8.png)<br>
<br>
<br>
**4. Develop a program to convert colour string to RGB color values.**<br>
from PIL import ImageColor<br>
img1=ImageColor.getrgb("Yellow")<br>
print(img1)<br>
img2=ImageColor.getrgb("red")<br>
print(img2)<br>
**OUTPUT**<br>
(255, 255, 0)<br>
(255, 0, 0)<br>
<br>
**5. Write a program to create Image using programs.**<br>
from PIL import Image <br>
img=Image.new('RGB',(200,400),(255,255,0))<br>
img.show()<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940850/174046289-402f6aa6-9029-4efc-97d7-09f1cfa62f2c.png)<br>
<br>
**6. Develop a program to visualize the image using various color space.**<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('butterfly3.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
plt.show()<br>
<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.imshow(img)<br>
plt.show()<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940850/174046727-aa05f644-2482-4671-925f-b6f5ed75d095.png)<br>
![image](https://user-images.githubusercontent.com/97940850/174046770-58ad4a03-36f0-47ed-bb0b-2aa87b9b6147.png)<br>
![image](https://user-images.githubusercontent.com/97940850/174046823-471827fe-7b38-4614-879c-16660c9c7990.png)<br>
<br>
**7. Write a program to display the image attributes.**<br>
from PIL import Image<br>
image=Image.open('plant4.jpg')<br>
print("FileName: ",image.filename)<br>
print("Format: ",image.format)<br>
print("Mode: ",image.mode)<br>
print("Size: ",image.size)<br>
print("Width: ",image.width)<br>
print("Height: ",image.height)<br>
image.close();<br>
**OUTPUT**<br>
FileName:  plant4.jpg<br>
Format:  JPEG<br>
Mode:  RGB<br>
Size:  (480, 720)<br>
Width:  480<br>
Height:  720<br>
<br>
<br>
**8. Resize the original image.**<br>
import cv2<br>
img=cv2.imread('flower5.jpg')<br>
print('origial image length width',img.shape)<br>
cv2.imshow('original image',img)<br>
cv2.waitKey(0)<br>
#to show the resized image<br>
imgresize=cv2.resize(img,(150,160))<br>
cv2.imshow('Resized image',imgresize)<br>
print('Resized image length, width',imgresize.shape)<br>
cv2.waitKey(0)<br>
**OUTPUT**<br>
origial image length width (640, 960, 3)<br>
Resized image lenght width (160, 150, 3)<br>
<br>
**9. Convert the original image to gray scale and then to binary.**<br>
import cv2<br>
#read the image file<br>
img=cv2.imread('butterfly3.jpg')<br>
cv2.imshow("RGB",img)<br>
cv2.waitKey(0)<br>
<br>
#Grayscale<br>
<br>
img=cv2.imread('butterfly3.jpg',0)<br>
cv2.imshow("Gray",img)<br>
cv2.waitKey(0)<br>
<br>
#Binary image<br>
<br>
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)<br>
cv2.imshow("Binary",bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940850/174048120-9fc7d698-0466-459c-a2bf-38816501be12.png)<br>
![image](https://user-images.githubusercontent.com/97940850/174048200-092e4aca-f297-492e-8af8-010b96b689dc.png)<br>
![image](https://user-images.githubusercontent.com/97940850/174048342-35f42e3d-bccf-4d8b-8b8f-44b19dd6182a.png)<br>

**10.Develop a program to read image using URL.**



