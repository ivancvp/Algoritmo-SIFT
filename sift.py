#!/usr/bin/env python
# coding: utf-8

# ## Algoritmo SIFT

# In[9]:


import cv2 
import csv
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

import glob
import re
import os

from pathlib import Path

def path_join(ruta,complemento):
    return os.path.join(ruta, complemento)

#función que aplica la transformación perspectiva
def warpTwoImages(img1, img2, H,url_salida,url_fuente,consecutivo):
    
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin),flags=cv2.INTER_NEAREST)

    """
    blanco=np.zeros((result.shape[0] ,result.shape[1],3), np.uint8)
    
    result1 = result.copy()
    
    blanco[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    
    logo_mask = np.where(result == 0, 255,result)
    
    result1[np.where(logo_mask == 255)] = blanco[np.where(logo_mask == 255)]
    """
    
    
    traslado_x=0
    traslado_y=0
    #translación de las imágenes
    print(t)
    if(t[0]==0 and t[1]==0):
        print("abajo")
        traslado_y=h1
    elif (t[0]==0 and t[1]>0):
        print("adelante")
        traslado_y=ymax-ymin
    elif (t[0]>0 and t[1]==0):
        print("izquierda")
        traslado_y=h1
        traslado_x=-t[0]
    elif (t[0]>0 and t[1]>0):
        print("arriba")
        traslado_y=ymax-ymin
        traslado_x=-t[0]
        
    
    return saveGeoRaster(url_fuente,url_salida,result,traslado_y,traslado_x,consecutivo)
    

salida='salida2'
vuelo='vuelo3'


dir_entrada_imagenes=os.path.join('./test/',os.path.join(salida,vuelo))

sift = cv2.xfeatures2d.SIFT_create()

with open(os.path.join(dir_entrada_imagenes, "mosaico.txt")) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    
    cuenta=0
    
    for row in csv_reader:
        
        print(row)
        
        kp1=[0]
        des1=[0]
        im=[0]
        iteracion=0;
        entrada=row[0]
        imagen=[entrada]
        
        for i in range(1,len(row)):

            file_path=os.path.join(dir_entrada_imagenes,row[i])
            imagen.append(os.path.basename(file_path))
            img_l = cv2.imread(str(file_path))
            im.append(img_l)
            img1_l = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
            kp, des= sift.detectAndCompute(img1_l, None)
            kp1.append(kp)
            des1.append(des)
            iteracion=iteracion+1;


        for p in range(1,iteracion+1): 
            
            print("entrada_imagen:"+str(entrada))
            
            img_l = cv2.imread(os.path.join(dir_entrada_imagenes,entrada)) 
            img1_l = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)    
            kp, des= sift.detectAndCompute(img1_l, None)    

            im[0]=img_l.copy() 
            kp1[0]=kp
            des1[0]=des
            imagen[0]=entrada


            bf = cv2.BFMatcher()

            calidad=0;
            indice=0;

            match=[]

            ite=1

            objetivo=1
            j=1;


            while j<len(imagen):

                matches = bf.knnMatch(des1[j],des1[0], k=2)
                good = []

                for m in matches:
                    if (m[0].distance < 0.7*m[1].distance):
                        good.append(m)
                matches = np.asarray(good)
                match.append(matches)

                if(calidad<matches.shape[0]):
                    objetivo=ite
                    calidad=matches.shape[0]

                j+=1          
                ite+=1
                indice+=1

            calidad=0

            matches=match[objetivo-1]  


            url_fuente=''
            kp_dst=kp1[0]
            kp_src=kp1[objetivo]
            img_=im[objetivo]
            img=im[0]
            url_fuente=imagen[0]


            if (len(matches[:,0]) >= 4):
                src = np.float32([ kp_src[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
                dst = np.float32([ kp_dst[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
                H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            else:
                raise AssertionError('Can’t find enough keypoints.')

            name=str(str(j)+".jpg")
            #hist_match(im[0], im[objetivo],name)


            entrada="out_"+str(cuenta)+str(p)+".jpg"
            consecutivo=str(cuenta)+str(p)

            url_fuente=path_join(dir_entrada_imagenes, url_fuente)
            
            url_salida=path_join(dir_entrada_imagenes, entrada)

            print(url_fuente)
            print(url_salida)

            entrada=warpTwoImages(img,img_, H,url_salida,url_fuente,consecutivo)

            del kp1[objetivo]
            del des1[objetivo]
            del imagen[objetivo]
            del im[objetivo]

            print("-----")
        
        cuenta=cuenta+1;


# In[1]:


import os
import gdal
from xml.etree import ElementTree as ET

#función que almacena la imágen obtenida del proceso de transofrmación perspectiva y aplica la georeferenciación de la imágen de entrada

def saveGeoRaster(url_fuente,url_salida,result,traslado_y,traslado_x,consecutivo):
    
    
    file = url_fuente
    ds = gdal.Open(file)

    ulx, xres, xskew, uly, yskew, yres  = ds.GetGeoTransform()

    yr=ds.RasterYSize

    xi=ulx
    yi=uly+(yr*yres)

    def puntoX(x):
        return xi+(x*xres)
    def puntoY(y):
        return yi-(y*yres)

    [cols, rows] = result[:,:,0].shape

    mov_x=puntoX(traslado_x)
    mov_y=puntoY(traslado_y)
    
    print("mov_x="+str(mov_x))
    print("mov_y="+str(mov_y))
    
    print("y_or="+str(yi))
    print("x_or="+str(xi))

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(url_salida, rows, cols, 3, gdal.GDT_Byte)
    outdata.SetGeoTransform((puntoX(traslado_x), xres, 0.0, puntoY(traslado_y), 0.0, yres))##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(result[:,:,2])
    outdata.GetRasterBand(2).WriteArray(result[:,:,1])
    outdata.GetRasterBand(3).WriteArray(result[:,:,0])
    
    outdata.GetRasterBand(1).SetNoDataValue(0) 
    outdata.GetRasterBand(2).SetNoDataValue(0) 
    outdata.GetRasterBand(3).SetNoDataValue(0) 
    
    outdata.FlushCache() ##saves to disk!!
    outdata = None
    band=None
    print(ds.GetGeoTransform())
    ds=None
    
    #Se crea el mosaico entre la imágen de entrada y el resultado del almacenamiento de la imágen con proyección perspectiva
    salida_vrt=os.path.join('./test/',os.path.join(salida,vuelo))
    salida_vrt=os.path.join(salida_vrt,'mosaic.vrt')
    print("gdalbuildvrt "+salida_vrt+" "+url_fuente+" "+url_salida)
    os.system("gdalbuildvrt "+salida_vrt+" "+url_fuente+" "+url_salida)
    
    #Modificación del fichero vrt para añadir las funcionalidad python para el promedio de imágenes en el traslape para el posterior mosaico.
    tree = ET.parse(salida_vrt)
    root = tree.getroot()

    for nodo in root.findall('VRTRasterBand'):
        nodo.set("subclass","VRTDerivedRasterBand")
        nodo.append(ET.fromstring('<PixelFunctionType>average</PixelFunctionType>'))
        nodo.append(ET.fromstring('<PixelFunctionLanguage>Python</PixelFunctionLanguage>'))
        nodo.append(ET.fromstring("""<PixelFunctionCode><![CDATA[
import numpy as np

def average(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    div = np.zeros(in_ar[0].shape)
    for i in range(len(in_ar)):
        div += (in_ar[i] != 0)
    div[div == 0] = 1

    y = np.sum(in_ar, axis = 0, dtype = 'uint16')
    y = y / div

    np.clip(y,0,255, out = out_ar)
]]>
        </PixelFunctionCode>  """))

    vrt_file=salida_vrt
    tree.write(vrt_file)

    gdal.SetConfigOption('GDAL_VRT_ENABLE_PYTHON','YES')
    
    entrada="mosaico_"+consecutivo+".tif"

    url_salida=path_join(dir_entrada_imagenes, entrada)
    
    ds1 = gdal.Open(salida_vrt)
    ds1 = gdal.Translate(url_salida,ds1)
    ds1 = None
    return entrada
    

