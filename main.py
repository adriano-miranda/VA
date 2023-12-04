import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
import statistics

def equalizeIntensity(inImage,nBin):
    if nBin is None:
        nBin=256
    hist,bins=get_histogram(inImage,nBin)
    ecu=hist.cumsum()
    ecu_m=np.ma.masked_equal(ecu,0)
    k=255/(ecu_m.max()-ecu_m.min())
    Hc=ecu_m-ecu.min()
    ecu_m=Hc*k
    ecu = np.ma.filled(ecu_m,0).astype('uint8')
    outImage=ecu[inImage]
    return outImage


def adjustIntensity(inImage,inRange,outRange):
    if inRange is None:
        imin=np.min(inImage)
        imax=np.max(inImage)
    else:
        imin=inRange[0]
        imax=inRange[1]

    if outRange is None:
        omax=1
        omin=0
    else:
        omin = outRange[0]
        omax = outRange[1]

    outImage=omin+(((omax-omin)*(inImage-imin))/(imax-imin))
    return outImage

def get_histogram(inImage,nBin):
    histogram,nBin=np.histogram(inImage.flatten(),nBin,[0,nBin])

    return histogram,nBin

def show_histogram(hist,nBin):
    plt.bar(nBin[0:-1], hist)
    plt.show()

def save_image(doc):
    imagen = cv2.imread(doc, 0)
    #image=np.array(imagen)
    #cv2.imwrite('lena.png', imagen)
    return imagen

def show_image(imagen):
    cv2.imshow('Imagen grises', imagen)
    cv2.waitKey(0)

def filterImage(inImage,kernel):
    ancho, largo = inImage.shape
    anchok, largok=kernel.shape
    x = anchok // 2
    y=largok//2
    x1=x*2
    y1=y*2
    x2=x
    y2=y
    anchoz=ancho+x1
    largoz=largo+y1
    zeroImage = np.zeros([anchoz,largoz])
    finalImage=np.zeros([ancho,largo])
    a=np.zeros(kernel.shape)

    for x in range (x2,anchoz-x2):
        for y in range (y2,largoz-y2):
            zeroImage[x,y]=inImage[x-x2,y-y2]

    for x in range (x2,anchoz-x2):
        for y in range (y2,largoz-y2):
            for i in range(anchok):
                for j in range(largok):
                    a[i][j]=zeroImage[x-x2+i][y-y2+j]
            finalImage[x-x2, y-y2] = np.sum(a* kernel)


   
    return finalImage;

def gaussKernel1D(sigma):
    N=2*(3*sigma)+1
    x0=N//2+1
    x=N//2
    j=x
    k=x*2
    kernel=np.zeros([N])
    for i in range(-x,x+1):
        kernel[i+x]=(1/(math.sqrt(2*math.pi)*sigma))*math.e**((-1*(i**2))/2*(sigma**2))
    print(kernel.sum())
    kernel=kernel/kernel.sum()
    return kernel


def gaussianfilter(inImage,sigma):
    N=2*(3*sigma)+1
    x0=N//2+1
    ancho, largo = inImage.shape
    x=N//2
    k=x*2
    zeroImage = np.zeros([ancho+k,largo+k])
    a=kernel=np.zeros([N,N])
    finalImage=np.zeros([ancho,largo])
    for i in range(-x,x+1):
        for j in range(-x,x+1):
            kernel[i+x][j+x]=(1/(2*math.pi*(sigma**2)))*math.e**(-1*(((i**2)+(j**2))/(2*(sigma**2))))
    print(kernel.sum())
    kernel=kernel/kernel.sum()
    print(kernel)
    finalImage=filterImage(inImage,kernel)


    
    return finalImage


def medianFilter (inImage, filterSize):
    ancho, largo = inImage.shape
    x = filterSize // 2
    x1=x*2
    x2=x
    anchoz=ancho+x1
    oneImage = np.ones([anchoz,anchoz])
    oneImage=oneImage*-1
    outImage=np.zeros([ancho,largo])
    a=[]
    for x in range (x2,anchoz-x2):
        for y in range (x2,anchoz-x2):
            oneImage[x,y]=inImage[x-x2,y-x2]

    for x in range (x2,anchoz-x2):
        for y in range (x2,anchoz-x2):
            a=[]
            for i in range(filterSize):
                for j in range(filterSize):
                    if(oneImage[x-x2+i][y-x2+j]>=0):
                        a.append(oneImage[x-x2+i][y-x2+j])
            outImage[x-x2, y-x2] = statistics.median(sorted(a))

    return outImage


def highBoost (inImage, A, method, param):
    highImage=inImage*(A-1)
    if(method=="gaussian"):
        filterImage=gaussianfilter(inImage,param)
    else: 
        if (method=="median"):
            filterImage=medianFilter(inImage,param)
        else:
            print("Metodo no válido")
            return highImage
    outImage=highImage+filterImage
    return outImage

def erode(inImage,SE,center=[]):
    ancho, largo = inImage.shape
    anchok=SE.shape[0]
    largok=SE.shape[1]
 
    if center==None:
        centerx=anchok//2
        centery=largok//2
    else:
        centerx=center[0]
        centery=center[1]
    x = anchok // 2
    y=largok//2
    x1=x*2
    y1=y*2
    x2=x
    y2=y
    anchoz=ancho+x1
    largoz=largo+y1
    zeroImage = np.zeros([anchoz,largoz])
    auxfinalImage=np.zeros([anchoz,largoz])
    finalImage=np.zeros([ancho,largo])
    a=np.zeros(SE.shape)

    for x in range (x2,anchoz-x2):
        for y in range (y2,largoz-y2):
            zeroImage[x,y]=inImage[x-x2,y-y2]

    for x in range (anchoz-1):
        for y in range (largoz-1):
            for i in range(anchok):
                for j in range(largok):
                    a[i][j]=zeroImage[x-x2+i][y-y2+j]
            control=True;
            for n in range(anchok):
                for m in range(largok):
                    if(a[n][m]<1 & SE[n][m]>=1):
                        control=False;   
            
            if control==False:
                auxfinalImage[x+centerx-x2][y+centery-y2]=0
            else:
                auxfinalImage[x+centerx-x2][y+centery-y2]=1
    for x in range (x2,anchoz-x2):
        for y in range (y2,largoz-y2):
            finalImage[x-x2,y-y2]=auxfinalImage[x,y]
    return finalImage;

def dilate(inImage,SE,center=[]):
    ancho, largo = inImage.shape
    anchok=SE.shape[0]
    largok=SE.shape[1]
 
    if center==None:
        centerx=anchok//2
        centery=largok//2
    else:
        centerx=center[0]
        centery=center[1]
    x = anchok // 2
    y=largok//2
    x1=x*2
    y1=y*2
    x2=x
    y2=y
    anchoz=ancho+x1
    largoz=largo+y1
    zeroImage = np.zeros([anchoz,largoz])
    auxfinalImage=np.zeros([anchoz,largoz])
    finalImage=np.zeros([ancho,largo])
    a=np.zeros(SE.shape)

    for x in range (x2,anchoz-x2):
        for y in range (y2,largoz-y2):
            zeroImage[x,y]=inImage[x-x2,y-y2]

    for x in range (anchoz-1):
        for y in range (largoz-1):
            for i in range(anchok):
                for j in range(largok):
                    a[i][j]=zeroImage[x-x2+i][y-y2+j]
            control=True;
            for n in range(anchok):
                for m in range(largok):
                    if(a[centerx][centery]>=1 & SE[n][m]>=1):
                        auxfinalImage[x+-x2+n][y-y2+m]=1   
    for x in range (x2,anchoz-x2):
        for y in range (y2,largoz-y2):
            finalImage[x-x2,y-y2]=auxfinalImage[x,y]
    return finalImage;

 

def opening(inImage,SE,center=[]):
    auxImage=erode(inImage,SE,center)
    finalImage=dilate(auxImage,SE,center)
    return finalImage;

def closing(inImage,SE,center=[]):
    auxImage=dilate(inImage,SE,center)
    finalImage=erode(auxImage,SE,center)
    return finalImage;

def fill (inImage, seeds, SE, center):
    ancho, largo = inImage.shape
    auxImage=np.zeros([ancho,largo])
    auxImage+=inImage
    filled=False
    if SE is None:
        SE=np.array([[0,1,0],[1,1,1],[0,1,0]])
    anchoe,largoe=SE.shape
    centerx=anchoe//2
    centery=largoe//2
    anchos,largos=seeds.shape
    for k in range(len(seeds)):
            auxImage[seeds[k][0]][seeds[k][1]]=1
    while (filled==False):
        auxImage1=auxImage
        auxImage=auxImage-inImage
        auxImage=dilate(auxImage,SE,center)
        auxImage+=inImage
        for x in range(ancho):
            for y in range(largo):
                if(auxImage[x][y]>1):
                    auxImage[x][y]=1
        if(np.array_equal(auxImage1,auxImage)):
            filled=True

                
                

        

    outImage=auxImage;
    return outImage


def gradientImage(inImage,operator):
    Gx=np.array([[0,0]])
    Gy=np.array([[0,0]])
    if (operator=="Roberts"):
        Gx=np.array([[-1,0],[0,1]])
        Gy=np.array([[1,0],[0,-1]])
    elif (operator=="CentralDiff"):
        Gx=np.array([[-1,0,1]])
        Gy=np.transpose(Gx)
    elif (operator=="Prewitt"):
        Gx=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        Gy=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    elif (operator=="Sobel"):
        Gx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        Gy=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    else:
            print("Metodo no válido")
            return [0,0]
    result=[filterImage(inImage,Gx),filterImage(inImage,Gy)]

    return result

def magnitud(Gx,Gy):
    mag=np.sqrt(Gx*Gx+Gy*Gy)
    return mag
    
def edgeCanny (inImage, sigma, tlow, thigh):
    ang=0.0
    gImage=gaussianfilter(inImage,sigma)
    Gx,Gy=gradientImage(gImage,"Sobel")
    modImage=np.sqrt(Gx*Gx+Gy*Gy)
    anchox, largox = Gx.shape
    anchoy, largoy = Gy.shape
    zeroimage=np.zeros([anchox,largox])
    q=0.0
    r=0.0
    q1=0.0
    r1=0.0

    for x in range(anchox):
        for y in range(largox):
            ang=math.degrees(math.atan2(Gx[x][y],Gy[x][y]))
            mg=magnitud(Gx[x][y],Gy[x][y])
            try:
                if (ang<0):
                    ang+=180
                if (ang<22.5 or 157.5<=ang<=180):
                    q=x
                    q1=x
                    r=y-1
                    r1=y+1
                elif (22.5<=ang<67.5):
                    q=x+1
                    q1=x-1
                    r=y-1
                    r1=y+1
                elif (67.5<=ang<112.5):
                    q=x+1
                    q1=x-1
                    r=y
                    r1=y
                elif (112.5<=ang<15.5):
                    q=x+1
                    r=y+1
                    q1=x-1
                    r1=y-1

                if(magnitud(Gx[q][r],Gy[q][r])>mg or magnitud(Gx[q1][r1],Gy[q1][r1])>mg):
                    zeroimage[x][y]=0
                else:
                    zeroimage[x][y]=mg
            except IndexError as e:
                pass
    for row in range(anchox):
        for col in range(largox):
            if (zeroimage[row][col]<tlow):
                zeroimage[row][col]=0
            elif (zeroimage[row][col]>thigh):
                zeroimage[row][col]=1
    filled=False
    while (filled==False):
        auxImage1=zeroimage
        try:
            for row in range(anchox):
                for col in range(largox):
                    if(zeroimage[row+1][col+1]>=1 or zeroimage[row-1][col-1]>=1 or zeroimage[row+1][col]>=1 or 
                        zeroimage[row-1][col]>=1 or zeroimage[row][col+1]>=1 or zeroimage[row][col-1]>=1 or 
                        zeroimage[row-1][col+1]>=1 or zeroimage[row+1][col-1]>=1 or zeroimage[row][col]>=1):
                        zeroimage[row][col]=1
        except IndexError as e:
                pass
        if(np.array_equal(auxImage1,zeroimage)):
            filled=True

            
    return zeroimage


def main():

    #image=np.array([[1,1,1,1,1,1,0],[1,1,1,1,1,1,0],[1,1,1,1,1,1,0],[1,1,1,1,1,1,0],[1,1,1,1,1,1,0],[0,0,0,0,0,0,0]])
    #image=np.zeros([128,128])
    #image[64][64]=1;
    #image[9][9]=1;
    #image[3][3] = 1;
    #image[6][6] = 1;
    #image=save_image('image5.png')/255
    image=save_image('grid.png')
    #kernel=np.array([[0.5,0.1,0.7],[0.1,0.7,0.1],[0.1,0.5,0.1]])
    seeds=np.array([[16,7],[16,3]])
    inRange=None
    outRange=[0.3,0.6]
    #imageF=equalizeIntensity(image,256)
    #imageF=gaussianfilter(image,1)#no la dividas entre 256
    #imageF=adjustIntensity(image,None,None)
    #imageF=filterImage(image,kernel)
    #kernel=gaussKernel1D(1)
    #imageF=gaussianfilter(image,1)
    #imageF=medianFilter(image,3)
    #imageF=highBoost(image,2,"gaussian",1)
    #imageF=erode(image,kernel,[0,0])
    #imageF=erode(image,kernel,[0,0])
    #imageF=dilate(image,kernel,[0,0])
    #imageF=opening(image,kernel,[0,0])
    #imageF=closing(image,kernel,[0,0])
    #print(image)
    #imageF=erode(image,kernel,[1,1])
    #imageF1=dilate(image,kernel,[1,1])

    #imageF2=erode(image,kernel1,[1,1])
    #imageF3=dilate(image,kernel1,[1,1])
    #imageF=gradientImage(image,"CentralDiff")
    ##imageF=edgeCanny(image,1,0.2,0.5)
    #print(imageF)
    #print(imageF1)
    #print(imageF2)
    #print(imageF3)
    #cv2.imwrite('erode1.png',imageF*255)
    #cv2.imwrite('dilate1.png',imageF1*255)
    #cv2.imwrite('erode2.png',imageF2*255)
    #cv2.imwrite('dilate2.png',imageF3*255)
    #imageF1=edgeCanny(image,1,0.2,0.2)
    #imageF2=edgeCanny(image,1,0.8,0.8)
    #imageF3=edgeCanny(image,1,0.2,0.8)
    imageF1=medianFilter(image,3)
    imageF2=medianFilter(image,5)
    imageF1= adjustIntensity(imageF1,None,None)
    imageF2= adjustIntensity(imageF2,None,None)
    #show_image(imageF)
    #show_image(imageF2)
    #show_image(imageF3)
    cv2.imwrite('imageF1.png',imageF1*255)
    cv2.imwrite('imageF2.png',imageF2*255)
    #cv2.imwrite('imageF2.png',imageF[1]*255)
    #cv2.imwrite('imageF2.png',imageF2*255)
    #cv2.imwrite('imageF3.png',imageF3*255)
    #hist,nBin=get_histogram(imageF,256)
    #show_histogram(hist,nBin)
   
    cv2.destroyAllWindows()


if __name__== '__main__':
    main()


