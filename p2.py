print("Practica 2")

import cv2 
import numpy as np
import matplotlib.pyplot as plt
import math

def formas_geometricas(img, lados):
    
    isRectangle = False #Debo volver esta variable a True cuando detecte efectivamente un rectangulo

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Blur=cv2.GaussianBlur(gray,(5,5),1) #apply blur to roi
    Canny=cv2.Canny(Blur,15,150) #apply canny to roi    

    #Find my contours
    contours =cv2.findContours(Canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]

    cntrRect = []
    for i in contours:
        epsilon = 0.05*cv2.arcLength(i,True)
        approx = cv2.approxPolyDP(i,epsilon,True)
    
        if len(approx) == lados:
            cv2.drawContours(img,cntrRect,-1,(0,255,0),2)
            cv2.imshow('IMAGEN con formas geometricas: {} lados'.format(lados),img)
            cntrRect.append(approx)

    # Mostramos el contorno en la imagen original.
    
    plt.title("IMAGEN con formas geometricas: {} lados".format(lados))
    plt.imshow(img)
    plt.show()
    return img

cv2.destroyAllWindows()

def filtrado_HSV_1(image, u_bajo, u_alto):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    plt.title("IMAGEN Convertida a HSV: ")
    plt.imshow(img_hsv)
    plt.show()
    
    # Elegimos el umbral de azul en HSV
    umbral_bajo = (0,0,u_bajo)
    umbral_alto = (0,0,u_alto)
    
    # hacemos la mask y filtramos en la original
    mask = cv2.inRange(img_hsv, umbral_bajo, umbral_alto)
    res = cv2.bitwise_and(image, image, mask=mask)# imprimimos los resultados
    
    #plt.imshow(mask, cmap='gray')
    #plt.show()
    plt.title("IMAGEN solo con color azul:")
    plt.imshow(res)
    plt.show()
    return res
    
def filtrado_HSV_2(image, u_bajo1, u_alto1, u_bajo2, u_alto2):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    plt.title("IMAGEN Convertida a HSV: ")
    plt.imshow(img_hsv)
    plt.show()
    
    # Elegimos el umbral de azul en HSV
    umbral_bajo1 = (0,0,u_bajo1)
    umbral_alto1 = (0,0,u_alto1)

    umbral_bajo2 = (0,0,u_bajo2)
    umbral_alto2 = (0,0,u_alto2)
    
    # hacemos la mask y filtramos en la original
    mask1 = cv2.inRange(img_hsv, umbral_bajo1, umbral_alto1)
    mask2 = cv2.inRange(img_hsv, umbral_bajo2, umbral_alto2)
    mask3 = cv2.inRange(img_hsv, 0, 200)

    res1 = cv2.bitwise_and(image, image, mask=mask1)# imprimimos los resultados
    res2 = cv2.bitwise_and(image, image, mask=mask2)
    res3 = cv2.bitwise_or(res1, res2)
    #plt.imshow(mask, cmap='gray')
    #plt.show()
    plt.title("IMAGEN solo con color azul:")
    plt.imshow(res3)
    plt.show()
    return res3

def detector_bordes(img, min, max):
    #Paso la imagen a escala de grises
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Filtro el ruido con filtro gausiano
    gauss = cv2.GaussianBlur(gris,(5,5), 0) #sigma es la anchura de la campana de Gauss(la desviacion) si lo ponemos a 0 se calcula automaticamente
    #Detección de bordes de Canny
    canny = cv2.Canny(gauss, min, max)
    #cv2.imshow("canny", canny)
    #Buscamos los contornos
    (contornos, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Mostramos numero de objetos por consola:
    print("He encontrado {} objetos".format(len(contornos)))
    cv2.drawContours(img, contornos, -1, (0,0,255), 2)
    
    return img

def detector_circulos(img, n_circulos, maxRadio):
    
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))
    
    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,param2 = 30, minRadius = 0, maxRadius = maxRadio)
   
    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
    
        n = 0
        for pt in detected_circles[0, :]:
            if(n < n_circulos):
                a, b, r = pt[0], pt[1], pt[2]
                # Draw the circumference of the circle.
                cv2.circle(img, (a, b), r, (0, 255, 0), 2)
                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
                cv2.imshow("Detected Circle", img)
                cv2.waitKey(10000)
                n = n+1
    
    return img


def showImage (image, title):
    plt.title(title)
    plt.imshow(image)
    plt.show()

def main():

    image1 = cv2.imread('00001.ppm',0)    #Leo la imagen en la escala de grises   (Tam: 675 X 921)
    image2 = cv2.imread('00159.ppm',0)
    image3 = cv2.imread('00214.ppm',0)
    image4 = cv2.imread('00040.ppm',0)
    image5 = cv2.imread('00115.ppm',0)
    image6 = cv2.imread('00023.ppm',0)
    image7 = cv2.imread('00171.ppm',0)
    image8 = cv2.imread('00177.ppm',0)
    image9 = cv2.imread('00219.ppm',0)
    image10 = cv2.imread('00235.ppm',0)

    image_color_1 = cv2.imread('00001.ppm',cv2.IMREAD_COLOR)
    image_color_2 = cv2.imread('00159.ppm',cv2.IMREAD_COLOR)
    image_color_3 = cv2.imread('00214.ppm',cv2.IMREAD_COLOR)
    image_color_4 = cv2.imread('00040.ppm',cv2.IMREAD_COLOR)
    image_color_5 = cv2.imread('00115.ppm',cv2.IMREAD_COLOR)
    image_color_6 = cv2.imread('00023.ppm',cv2.IMREAD_COLOR)
    image_color_7 = cv2.imread('00171.ppm',cv2.IMREAD_COLOR)
    image_color_8 = cv2.imread('00177.ppm',cv2.IMREAD_COLOR)
    image_color_9 = cv2.imread('00219.ppm',cv2.IMREAD_COLOR)
    image_color_10 = cv2.imread('00235.ppm',cv2.IMREAD_COLOR)

    #showImage(image_color_1,"IMAGEN ORIGINAL 1:")
    #showImage(image_color_2,"IMAGEN ORIGINAL 2:")
    #showImage(image_color_3,"IMAGEN ORIGINAL 3:")
    #showImage(image_color_4,"IMAGEN ORIGINAL 4:")
    #showImage(image_color_5,"IMAGEN ORIGINAL 5:")
    #showImage(image_color_6,"IMAGEN ORIGINAL 6:")
    #showImage(image_color_7,"IMAGEN ORIGINAL 7:")
    #showImage(image_color_8,"IMAGEN ORIGINAL 8:")
    #showImage(image_color_9,"IMAGEN ORIGINAL 9:")
    #showImage(image_color_10,"IMAGEN ORIGINAL 10:")

    #FILTRO LAS IMAGEN POR COLOR 
    #hsv1 = filtrado_HSV_1(image1, 35, 60)       #FUNCIONA Filtra un color en un umbral (umbral_alto, umbral_bajo)

    #DETECTO LOS BORDES para tener los objetos de la imagen (señales) 
    #d1 = detector_bordes(hsv1, 20, 185)
    #showImage(d2,"Bordes Canny con filtrado hsv")
   
    #DEETCTO LOS CIRULOS
    #circulos = detector_circulos(d2, 2, 40)   #Detectar circulos de imagen 
    #showImage(circulos, "Imagen con circulos")
    
    #formas_geometricas(circulos, 3)

#################### EJEMPLOS ###################################################################

    #Ejemplo 1: detección circulos sin más filtros 
    #circulos = detector_circulos(image_color_1, 2, 30)
    #showImage(circulos, "señales redondas")

    #Ejemplo 2: Detección de 2 señales (circulos AZUL), señal INDICACION (cuadrada AZUL) y PELIGRO(triangulo rojo)
    #hsv1 = filtrado_HSV_1(image1, 35, 60)  
    #d1 = detector_bordes(hsv1, 20, 185)
    #showImage(d1,"Bordes Canny con filtrado hsv")
    #circulos = detector_circulos(d1, 1, 30)
    #showImage(circulos, "Imagen con señales redondas azules")
    #g = formas_geometricas(circulos, 4)    #Las señales de Stop las detecta con 4 lados no 8

    #Ejemplo 3: señal OBLIGACION (redonda azul)
    #hsv3 = filtrado_HSV_1(image3, 80, 89)
    #d3 = detector_bordes(hsv3, 0, 190)     #con min = 50 y max = 311 detecta las dos señales y solo 3 objetos
    #showImage(d3,"Bordes Canny")
    #circulos = detector_circulos(d3, 1, 40)   #Detectar circulos de imagen 1 sin filtrar
    #showImage(circulos, "Imagen con señales redondas azules")
    

    #Ejemplo 4: 4 señales OBLIGACION (redonda AZUL) Detecto 248 objetos entre ellos las señales pero no detecta los circulos 
    #hsv2 = filtrado_HSV_2(image2, 35, 45, 70, 120)  #Filtro entre dos rangos de colores (min, max) para detectar todas las señales
    #d2 = detector_bordes(hsv2, 10, 305)
    #showImage(d2,"Bordes Canny con filtrado hsv")   
    
    #Ejemplo 5:  mismo que el ejemplo 4 pero aumento el rango de canny, detectando menos señales pero ahora me detecta el con circulos
    #hsv2 = filtrado_HSV_2(image2, 35, 45, 70, 120)
    #d2 = detector_bordes(hsv2, 10, 360)
    #circulos = detector_circulos(d2, 2, 30)

    #Ejemplo 6 : Detecto señal OBLIGACION (redonda azul) y PELIGRO (triangulo rojo)
    #hsv4 = filtrado_HSV_1(image4, 85, 160)
    #d4 = detector_bordes(hsv4, 5, 455)  #con 475 detecta las dos señales y 45 objetos pero ya no detecta la señal redonda 
    #showImage(d4,"Bordes Canny con filtrado hsv")
    #circulos = detector_circulos(d4, 1, 40)
    #showImage(circulos, "Imagen con circulos")
    #g = formas_geometricas(circulos, 3)

    #Ejemplo 7: Deteccion señal PROHIBICION (STOP)
    #hsv5 = filtrado_HSV_1(image5, 70, 80)
    #d5 = detector_bordes(hsv5, 10, 280)
    #showImage(d5,"Bordes Canny con filtrado hsv")
    #g = formas_geometricas(d5, 4)   #Las señales de Stop las detecta con 4 lados no con 8

    #Ejemplo 8: Deteccion 2 señales PROHIBICION (redonda roja) y PELIGRO(triangular roja)
    #hsv6 = filtrado_HSV_1(image6, 110, 155)
    #d6 = detector_bordes(hsv6, 20, 510)
    #showImage(d6,"Bordes Canny con filtrado hsv")
    #circulos = detector_circulos(d6, 1, 40)   
    #g = formas_geometricas(circulos, 3)

    #Ejemplo 9 : Detección 2 señales PROHIBICION (redondas rojas)
    #hsv7 = filtrado_HSV_1(image7, 20, 40)
    #d7 = detector_bordes(hsv7, 20, 100)
    #showImage(d7,"Bordes Canny con filtrado hsv")
    #circulos = detector_circulos(d7, 2, 40)
    #showImage(circulos, "Imagen con circulos")

    #Ejemplo 10: Detección señales PELIGRO (triangular roja) y OBLIGACION(redonda azul) 
    #hsv9 = filtrado_HSV_1(image9, 40, 65)
    #d9 = detector_bordes(hsv9, 20, 160)
    #showImage(d9,"Bordes Canny con filtrado hsv")
    #circulos = detector_circulos(d9, 1, 40)
    #showImage(circulos, "Imagen con circulos")
    #g = formas_geometricas(d9, 3)

    #Ejemplo 11: Señales INDICACION (cuadradas azules) 
    #hsv10 = filtrado_HSV_1(image10, 60, 110)
    #d10 = detector_bordes(hsv10, 20, 325)
    #showImage(d10,"Bordes Canny con filtrado hsv")
    #formas_geometricas(d10, 4)

    ###COSAS QUE MEJORAR PARA SUBIR LA NOTA: (También se pude mejorar la P1) 
    #1.- filtro HSV, para filtrar el rojo filtrar por H, S y V no solo por H (con el azul lo mismo)
    #2.- Eliminar errores de la detección de triagulos (detecta más triangulos a parte de las señales), (los triangulos que quiero detectar son esquilatero)
    #3.- Automatizar el programa para que se ejecuten siempre todos los filtros sobre cada imagen con los mismos parametros y de una solución aceptable
    #4.- Detectar las señales, mostrar de alguna forma que la señal que la señal azul circular es de obligacion (por ejemplo)
if __name__ == '__main__':
    main()