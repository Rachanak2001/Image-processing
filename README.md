**IMAGE PROCESSING**
**1. Develop a program to display grayscale image using read and write operation.**
pip install opencv-python
import cv2
img=cv2.imread('flower5.jpg',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
**OUTPUT**
![image](https://user-images.githubusercontent.com/97940850/174041726-df4b96be-11b2-4f5a-b4b9-aee932a7be26.png)


**2. Develop a program to display the image using matplotlib.**
import matplotlib.image as mping
import matplotlib.pyplot as plt
img=mping.imread('plant4.jpg')
plt.imshow(img)
**OUTPUT**
![image](https://user-images.githubusercontent.com/97940850/174042162-a75f11ae-0ea3-4885-ae30-ff9720232530.png)


**3. develop a program to perform linear transformation. Rotation**
import cv2
from PIL import Image
img=Image.open("plant4.jpg")
img=img.rotate(180)
img.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
**OUTPUT**
![image](https://user-images.githubusercontent.com/97940850/174042633-31ed2a88-33e8-4f3e-9a90-0a1a721187b8.png)


**4. Develop a program to convert colour string to RGB color values. **
from PIL import ImageColor
img1=ImageColor.getrgb("Yellow")
print(img1)
img2=ImageColor.getrgb("red")
print(img2)
**OUTPUT**
(255, 255, 0)
(255, 0, 0)

**5. Write a program to create Image using programs.**
from PIL import Image 
img=Image.new('RGB',(200,400),(255,255,0))
img.show()
**OUTPUT**
![image](https://user-images.githubusercontent.com/97940850/174046289-402f6aa6-9029-4efc-97d7-09f1cfa62f2c.png)

**6. Develop a program to visualize the image using various color space.**
import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('butterfly3.jpg')
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
plt.imshow(img)
plt.show()
**OUTPUT**
![image](https://user-images.githubusercontent.com/97940850/174046727-aa05f644-2482-4671-925f-b6f5ed75d095.png)
![image](https://user-images.githubusercontent.com/97940850/174046770-58ad4a03-36f0-47ed-bb0b-2aa87b9b6147.png)
![image](https://user-images.githubusercontent.com/97940850/174046823-471827fe-7b38-4614-879c-16660c9c7990.png)

**7. Write a program to display the image attributes.**
from PIL import Image
image=Image.open('plant4.jpg')
print("FileName: ",image.filename)
print("Format: ",image.format)
print("Mode: ",image.mode)
print("Size: ",image.size)
print("Width: ",image.width)
print("Height: ",image.height)
image.close();
**OUTPUT**
FileName:  plant4.jpg
Format:  JPEG
Mode:  RGB
Size:  (480, 720)
Width:  480
Height:  720


**8. Resize the original image.**
import cv2
img=cv2.imread('flower5.jpg')
print('origial image length width',img.shape)
cv2.imshow('original image',img)
cv2.waitKey(0)
#to show the resized image
imgresize=cv2.resize(img,(150,160))
cv2.imshow('Resized image',imgresize)
print('Resized image lenght width',imgresize.shape)
cv2.waitKey(0)
**OUTPUT**
origial image length width (640, 960, 3)
Resized image lenght width (160, 150, 3)

**9. Convert the original image to gray scale and then to binary.**
import cv2
#read the image file
img=cv2.imread('butterfly3.jpg')
cv2.imshow("RGB",img)
cv2.waitKey(0)

#Grayscale

img=cv2.imread('butterfly3.jpg',0)
cv2.imshow("Gray",img)
cv2.waitKey(0)

#Binary image

ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow("Binary",bw_img)
cv2.waitKey(0)
cv2.destroyaAlWindows()
**OUTPUT**
![image](https://user-images.githubusercontent.com/97940850/174048120-9fc7d698-0466-459c-a2bf-38816501be12.png)
![image](https://user-images.githubusercontent.com/97940850/174048200-092e4aca-f297-492e-8af8-010b96b689dc.png)
![image](https://user-images.githubusercontent.com/97940850/174048342-35f42e3d-bccf-4d8b-8b8f-44b19dd6182a.png)


