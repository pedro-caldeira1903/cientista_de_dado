!pip install opencv-python
!sudo apt install tesseract-ocr
!pip install pytesseract
!mkdir tessdata
!wget -O ./tessdata/por.traineddata https://github.com/tesseract-ocr/tessdata/blob/main/por.traineddata?raw=true
import cv2, pytesseract, re
from google.colab.patches import cv2_imshow
img=cv2.imread('#placa de carro')
cv2_imshow(img)
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(img)
#Esse tipo de filtro só serve para alguns tipos de imagens
a, lim_otsu=cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2_imshow(lim_otsu)
print(a)
erosao=cv2.erode(lim_otsu2, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
cv2_imshow(erosao)
borda2=cv2.Canny(img2, 100, 200)
cv2_imshow(borda2)
contornos2, hierarquia=cv2.findContours(borda2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contornos2=sorted(contornos2, key = cv2.contourArea, reverse = True)[:10]
for contorno2 in contornos2:
  epsilon2=0.02 * cv2.arcLength(contorno2, True)
  aproximacao2=cv2.approxPolyDP(contorno2, epsilon2, True)
  if cv2.isContourConvex(aproximacao2) and len(aproximacao2) == 4:
    localizacao2=aproximacao2
    break
localizacao2
x, y, w, h=cv2.boundingRect(localizacao2)
placa2=img2[y:y+h, x:x+w]
cv2_imshow(placa2)
valor2, lim_otsu2=cv2.threshold(placa2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
erosao2=cv2.erode(lim_otsu2, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
cv2_imshow(erosao2)
config_tesseract2='--tessdata-dir tessdata --psm 6'
texto2=pytesseract.image_to_string(lim_otsu2, lang='por', config=config_tesseract2)
print(texto2)
print(re.search('\w{3}\d{1}\w{1}\d{2}', texto2).group(0))
