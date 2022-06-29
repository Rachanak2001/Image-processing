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
<br>
**10.Develop a program to read image using URL.**<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://www.thoughtco.com/thmb/mik7Z00SAYN786BQbieXWOzZmc8=/2121x1414/filters:fill(auto,1)/lotus-flower-828457262-5c6334b646e0fb0001dcd75a.jpg'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>
<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940850/175267175-16d4e9e5-a412-49b8-8d6b-9e39b29064b4.png)<br>
<br>
<br>
**11.Write a program to mask and blur the image.**<br>
import cv2<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('fish2.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97940850/175267604-fade1834-70c7-480d-97d4-0aefac998ffd.png)<br>
<br>
hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
light_orange=(1,190,200)<br>
dark_orange=(18,255,255)<br>
mask=cv2.inRange(img,light_orange,dark_orange)<br>
result=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97940850/175267763-ce4b43ce-48a4-47ce-9bf3-b1ca3105b8fe.png)<br>
<br>
light_white=(0,0,200)<br>
dark_white=(145,60,225)<br>
mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img,mask=mask_white)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask_white,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97940850/175268219-32d7eb9a-3ef4-4a8d-9510-fadc180d07f7.png)<br>
<br>
final_mask=mask+mask_white<br>
final_result=cv2.bitwise_and(img,img,mask=final_mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(final_mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(final_result)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97940850/175268463-d53cb81d-d3ea-457e-b94c-2baa0050dcd3.png)<br>
<br>
blur=cv2.GaussianBlur(final_result,(7,7),0)<br>
plt.imshow(blur)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97940850/175268590-97ab250f-c39b-4f74-8ad1-dc4fc6b1b807.png)<br>
<br>
**12. Write a program to perform arithmatic operations on image.**<br>
import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
#Reading image files<br>
img1=cv2.imread('plant1.jpg')<br>
img2=cv2.imread('plant3.jpg')<br>
<br>
#Applying Numpy addition on image<br>
fimg1=img1+img2<br>
plt.imshow(fimg1)<br>
plt.show()<br>
<br>
#Saving the output image<br>
cv2.imwrite('output.jpg',fimg1)<br>
fimg2=img1-img2<br>
plt.imshow(fimg2)<br>
plt.show()<br>
<br>
#Saving the output image<br>
cv2.imwrite('output.jpg',fimg2)<br>
fimg3=img1*img2<br>
plt.imshow(fimg3)<br>
plt.show()<br>
<br>
#Saving the output image<br>
cv2.imwrite('output.jpg',fimg3)<br>
fimg4=img1/img2<br>
plt.imshow(fimg4)<br><br>
plt.show()<br>
<br>
#Saving the output image<br>
cv2.imwrite('output.jpg',fimg4)<br>
<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940850/175286295-9e8579b6-738c-4c1e-8e5d-6b98fd174224.png)
![image](https://user-images.githubusercontent.com/97940850/175286432-76d45d3a-774a-4d96-b359-e5fee841a548.png)
![image](https://user-images.githubusercontent.com/97940850/175286459-cec413b9-fedb-4f16-a583-b8626be69de0.png)
![image](https://user-images.githubusercontent.com/97940850/175286479-424eda13-e076-4aef-ab87-08f821a654a3.png)

<br>
**13.Develop the program to change the image to different color spaces.**<br>
import cv2 <br>
img=cv2.imread("flower5.jpg")<br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br>
cv2.imshow("GRAY image",gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)<br>
cv2.imshow("HLS image",hls)<br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br><br>
<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940850/175287410-60d53f53-0581-4ac6-a9a4-eab46aa7dff9.png)<br>
![image](https://user-images.githubusercontent.com/97940850/175287466-cef916c4-4116-48f1-b256-902942fead78.png)<br>
![image](https://user-images.githubusercontent.com/97940850/175287520-5dd9e8fe-db58-468d-ab86-ca919f4661c7.png)<br>
![image](https://user-images.githubusercontent.com/97940850/175287565-4e4287da-a5c9-4ea3-a5f9-b5373677319b.png)<br>
![image](https://user-images.githubusercontent.com/97940850/175287632-a1a7edf4-6c84-4550-aef8-030de594bbe0.png)<br>

<br>
**14.Program to create an image using 2D array.**<br>
import cv2 as c<br>
import numpy as np<br>
from PIL import Image<br>
array=np.zeros([100,200,3],dtype=np.uint8)<br>
array[:,:100]=[255,130,0]<br>
array[:,100:]=[0,0,255]<br>
img=Image.fromarray(array)<br>
img.save('flower5.jpg')<br>
img.show()<br>
c.waitKey(0)<br>
<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940850/175271426-a7f9f364-377a-4390-96df-e286e4bed715.png)<br>

**15. Bitwise Operation**
import cv2
import matplotlib.pyplot as plt
image1=cv2.imread('pic1.jpg',1)
image2=cv2.imread('pic1.jpg')
ax=plt.subplots(figsize=(15,10))
bitwiseAnd=cv2.bitwise_and(image1,image2)
bitwiseOr=cv2.bitwise_or(image1,image2)
bitwiseXor=cv2.bitwise_xor(image1,image2)
bitwiseNot_img1=cv2.bitwise_not(image1)
bitwiseNot_img2=cv2.bitwise_not(image2)
plt.subplot(151)
plt.imshow(bitwiseAnd)
plt.subplot(152)
plt.imshow(bitwiseOr)
plt.subplot(153)
plt.imshow(bitwiseXor)
plt.subplot(154)
plt.imshow(bitwiseNot_img1)
plt.subplot(155)
plt.imshow(bitwiseNot_img2)
cv2.waitKey(0)

**OUTPUT**
![image](https://user-images.githubusercontent.com/97940850/176423043-c43a0c49-5f42-4eb6-bd60-1542e0c1ff6a.png)

**16.Blurring an Image**
#importing libraries
import cv2
import numpy as np

image=cv2.imread('pic6.jpg')

cv2.imshow('Original Image',image)
cv2.waitKey(0)

#Gaussian blur
Gaussian=cv2.GaussianBlur(image,(7,7),0)
cv2.imshow('Gaussian Blurring',Gaussian)
cv2.waitKey(0)

#Medium Blur
median=cv2.medianBlur(image,5)
cv2.imshow('Median Blurring',median)
cv2.waitKey(0)

#Bilateral Blur
bilateral=cv2.bilateralFilter(image,9,75,75)
cv2.imshow('Bilateral Blurring',bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()

**OUTPUT**

**18. Morphological Operation**
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image,ImageEnhance
img=cv2.imread('pic1.jpg',0)
ax=plt.subplots(figsize=(20,10))
kernel=np.ones((9,9),np.uint8)
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
erosion=cv2.erode(img,kernel,iterations=1)
dilation=cv2.dilate(img,kernel,iterations=1)
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
plt.subplot(151)
plt.imshow(opening)
plt.subplot(152)
plt.imshow(closing)
plt.subplot(153)
plt.imshow(erosion)
plt.subplot(154)
plt.imshow(dilation)
plt.subplot(155)
plt.imshow(gradient)
cv2.waitKey(0)

**OUTPUT**
![image](https://user-images.githubusercontent.com/97940850/176424620-edb73f30-1828-4b0c-95a6-3ca15b3461a2.png)
