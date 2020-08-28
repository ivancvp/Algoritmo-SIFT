#!/usr/bin/env python
# coding: utf-8

# ### Definicion de la matrix rotacion 

# In[3]:


import numpy as np
def eulerAnglesToRotationMatrix(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
         
         
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R

eulerAnglesToRotationMatrix([90,90,90])


# ### Función para la obtención del GSD

# In[2]:


import math 

#funcion para obtener el GSD (se toma el GSD más alto)
def gsd_get(altura,filas,columnas):
    gsd_h=(altura*100*0.88)/(0.88*filas)
    gsd_w=(altura*100*1.32)/(0.88*columnas)
    if(gsd_h<gsd_w):
        return gsd_w
    else:
        return gsd_h

#funcion de rotación de coordenadas en base a un punto   
def rotar_3d(x,y,x_c,y_c,roll,pitch,yaw):
    
    v_c = np.array([x,y,0])
    m_rot=eulerAnglesToRotationMatrix([abs(roll)*math.pi/180,abs(90+pitch)*math.pi/180,(360-yaw)*math.pi/180])
    new_cord=(m_rot.dot(v_c))
    return new_cord+np.array([x_c,y_c,0])  
     


# ## Función para extraer la altura con base el servicio de Google

# In[4]:


from json import loads
from time import sleep
from urllib.request import Request, urlopen
 
def altura_google(lat,lon):
    try:
        request = Request('https://maps.googleapis.com/maps/api/elevation/json?locations={0},{1}&key=AIzaSyAdt_RZBVsnoST9CQXiRsNvq7CvmwGXfBU'.format(lat,lon))
        response = urlopen(request).read() 
        places = loads(response)
  
        return places['results'][0]['elevation']
        
    except:
        print("error")


# #### Función para enmascarar las rutas con espacios

# In[5]:


import os

def path_espaces(ruta):
    return "\""+ruta+"\""

def path_join(ruta,complemento):
    return os.path.join(ruta, complemento)

def path_create(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# #### Propiedades de una imágen -gdalinfo

# In[6]:


import json

def get_cadena(ruta):

    info=os.popen("gdalinfo -stats -json "+ruta).read()
    
    cadena=""

    j=0
    data = json.loads(info)
    for p in data['bands']:

        minimo=str(p['minimum'])
        maximo=str(p['maximum'])
        media=str(p['mean'])
        desviacion=str(p['stdDev'])
        
        #desde=str(p['mean']-2*p['stdDev'])
        #hasta=str(p['mean']+2*p['stdDev'])
        desde=minimo
        hasta=maximo
        
        if j==0:
            cadena="-scale_1 "+desde+" "+hasta+" 0 255"
        elif j==1:
            cadena=cadena+" -scale_2 "+desde+" "+hasta+" 0 255"
        elif j==2:
            cadena=cadena+" -scale_3 "+desde+" "+hasta+" 0 255"
        j=j+1

    return cadena


# ### Algoritmo para proyección de las imágenes

# Código para obtener exif data con exiftool.exe
# 
# exiftool -n "C:\Users\Ivan Carrillo\Desktop\Tesis Maestria\5. Datos Campo\Fotos dron 1ra salida\Vuelo 1" > out.txt

# ### Forma para determinar la altura basado en coordenadas del modelo de alturas GMTED2010

# In[18]:


import os

salida="salida2"
vuelo="vuelo3"


dir_entrada_imagenes=os.path.join(r'C:\Users\Ivan Carrillo\Desktop\Tesis Maestria\5. Datos Campo',os.path.join(salida,vuelo))


#directorios
dir_salida_exif=os.path.join(dir_entrada_imagenes, "out.txt")

#herramienta de extracción de metadatos de las imágenes
exiftool=os.system("exiftool -n "+path_espaces(dir_entrada_imagenes)+" > "+path_espaces(dir_salida_exif))

import csv

listado_img=[]
with open(os.path.join(dir_entrada_imagenes, "filelist.txt")) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
         listado_img.append(row[0])

print(listado_img)


# In[19]:


import re
import os
import cv2 
import matplotlib.pyplot as plt
from osgeo.osr import SpatialReference, CoordinateTransformation

# Ubicación del fichero con los metadatos de las imágenes.
path = dir_salida_exif
days_file = open(path,'r')

#Lectura del DEM para la extracción de la altura central de las imágenes.
dir_modelo_elev=r"C:\Users\Ivan Carrillo\Desktop\Tesis Maestria\6. Procesamiento\ModeloElevacion\10S090W_20101117_gmted_med075.tif"

#lectura del fichero de los metadatos de las imágenes.
mylines = []                               
with open (dir_salida_exif, 'r') as myfile:
    for myline in myfile:                  
        mylines.append(myline)              

# Definición del sistema de referencia 3116
epsg28992 = SpatialReference()
epsg28992.ImportFromEPSG(3116)
# Definición del sistema de referencia 4326
epsg4326 = SpatialReference()
epsg4326.ImportFromEPSG(4326)
latlon2rd = CoordinateTransformation(epsg4326, epsg28992)        
tolatlon = CoordinateTransformation(epsg28992, epsg4326)       
    

indice=0
#Se reccorre el archivo exif para procesar cada imágen
for x in range(1,100):

    #numero de la imagen
    img=x
    
    indice=(img-1)*135
    
    #nombre del archivo - Imágen
    nombre=mylines[indice+2].split(":",1)[1]
    
    archivo=nombre.split(".",1)[0].split()[0]

    if archivo in listado_img:

        #velocidad del dron en los ejes X y Y
        vel_x = float(mylines[indice+42].split(":",1)[1])
        vel_y = float(mylines[indice+43].split(":",1)[1])
        #Movimientos de la camara del dron
        pitch = float(mylines[indice+48].split(":",1)[1])
        yaw = float(mylines[indice+49].split(":",1)[1])
        roll = float(mylines[indice+50].split(":",1)[1])

        #Centro de la foto calibrado en x
        x_c = float(mylines[indice+95].split(":",1)[1])
        #Centro de la foto calibrado en y
        y_c = float(mylines[indice+96].split(":",1)[1])
        #tamano de la imagen
        columnas=float(mylines[indice+120].split(":",1)[1].split()[0])
        filas=float(mylines[indice+120].split(":",1)[1].split()[1])
        #altitud
        altura_c = float(mylines[indice+125].split(":",1)[1])
        #latitud del centro de la imágen
        latitud_c = float(mylines[indice+126].split(":",1)[1])
        #longitud del centro de la imágen
        longitud_c = float(mylines[indice+127].split(":",1)[1])

        print(latitud_c)
        print(longitud_c)

        #Extracción de la altura con base de un DEM 
        result_elev = os.popen("gdallocationinfo \""+dir_modelo_elev+"\" -wgs84 "+str(longitud_c)+" "+str(latitud_c)+" -xml").read()
        result_elev = re.search('<Value>(.*)</Value>', result_elev)
        elev=float(result_elev.group(1))

        print("altura_dem:"+str(elev))
        print("ältura_drone:"+str(altura_c))

        #altura constante para el vuelo de la primer salida=3309
        #altura constante para el vuelo de la segunda salida=3149

        #Extracción de la altura a partir del servicio de Google 
        altura_constante=3139 #altura constante, tomada de sumar 3014 a 295 metros, altura referencia lago de tota 
        #alt=altura_constante-float(altura_google(latitud_c,longitud_c))
        alt=150

        #alt=125 //ALTURA OK
        #alt=125
        sleep(0.5)
        print("altura_tomada:"+str(alt))

        gsd=gsd_get(alt,filas,columnas)
        print("gsd:"+str(gsd))

        #tamano de la imagen en terreno
        x_t=(gsd/100)*columnas
        y_t=(gsd/100)*filas

        print("x_t:"+str(x_t))
        print("y_t:"+str(y_t))

        # transforma las coordenadas del punto central de la imagen a EPSG:3116
        lonlatz = latlon2rd.TransformPoint(longitud_c, latitud_c)

        if vel_x<0 and vel_y<0:
            x_c=lonlatz[0]-(vel_x/2) #con menos corre la imagen a la izquierda
            y_c=lonlatz[1]-(vel_y/2) #con menos baja la imagen en QGIS
        elif vel_x<0 and vel_y>0:
            x_c=lonlatz[0]-(vel_x/2) #con menos corre la imagen a la izquierda
            y_c=lonlatz[1]+(vel_y/2) #con menos baja la imagen en QGIS
        elif vel_x>0 and vel_y<0:
            x_c=lonlatz[0]+(vel_x/2) #con menos corre la imagen a la izquierda
            y_c=lonlatz[1]-(vel_y/2) #con menos baja la imagen en QGIS
        elif vel_x>0 and vel_y>0:
            x_c=lonlatz[0]-(vel_x/2) #con menos corre la imagen a la izquierda
            y_c=lonlatz[1]-(vel_y/2) #con menos baja la imagen en QGIS

        #puntos de cada vertice de la imágen
        a_x=x_c-(x_t/2)
        a_y=y_c+(y_t/2)
        b_x=x_c+(x_t/2)
        b_y=y_c+(y_t/2)
        c_x=x_c+(x_t/2)
        c_y=y_c-(y_t/2)
        d_x=x_c-(x_t/2)
        d_y=y_c-(y_t/2)

        #coordenadas rotadas en el espacio tridimensional
        #punto a

        a_xr=rotar_3d(-(x_t/2),+(y_t/2),x_c,y_c,roll,pitch,yaw)[0]
        a_yr=rotar_3d(-(x_t/2),+(y_t/2),x_c,y_c,roll,pitch,yaw)[1]

        b_xr=rotar_3d(+(x_t/2),+(y_t/2),x_c,y_c,roll,pitch,yaw)[0]
        b_yr=rotar_3d(+(x_t/2),+(y_t/2),x_c,y_c,roll,pitch,yaw)[1]

        c_xr=rotar_3d(+(x_t/2),-(y_t/2),x_c,y_c,roll,pitch,yaw)[0]
        c_yr=rotar_3d(+(x_t/2),-(y_t/2),x_c,y_c,roll,pitch,yaw)[1]

        d_xr=rotar_3d(-(x_t/2),-(y_t/2),x_c,y_c,roll,pitch,yaw)[0]
        d_yr=rotar_3d(-(x_t/2),-(y_t/2),x_c,y_c,roll,pitch,yaw)[1]

        #construcción de los puntos de control
        gcp="-gcp 0 0 "+str(a_xr)+" "+str(a_yr)+" -gcp "+str(columnas)+" 0 "+str(b_xr)+" "+str(b_yr)+" -gcp "+str(columnas)+" "+str(filas)+" "+str(c_xr)+" "+str(c_yr)+" -gcp 0 "+str(filas)+" "+str(d_xr)+" "+str(d_yr)

        #preparación de los archivos de entrada y salida a los cuales se aplicará la reproyección
        dir_entrada=dir_entrada_imagenes
        archivo=nombre.split(".",1)[0].split()[0]
        filename_entrada=archivo+".JPG" 

        path_create(path_join(dir_entrada_imagenes,"geo"))

        dir_salida=path_join(dir_entrada_imagenes,"geo")

        filename_salida=archivo+"_g.tif" 
        filename_salida_rescale=archivo+".tif" 
        fullpath_entrada = path_join(dir_entrada, filename_entrada)
        fullpath_salida = path_join(dir_salida, filename_salida)
        
        
        dir_salida_rescale=os.path.join('./test',os.path.join(salida,vuelo))
        fullpath_salida_rescale = path_join(dir_salida_rescale, filename_salida_rescale)
        fullpath_salida=path_espaces(fullpath_salida)
        fullpath_salida_rescale=path_espaces(fullpath_salida_rescale)
        fullpath_entrada=path_espaces(fullpath_entrada)

        #se reproyecta la imagen
        os.system("gdal_translate "+gcp+" "+fullpath_entrada+" "+fullpath_salida)
        print("gdal_translate "+gcp+" "+fullpath_entrada+" "+fullpath_salida)
        #se reduce la resolución espacial de la imagen:
        print("gdalwarp -tr 0.3 0.3 -t_srs EPSG:3116 -r cubic -srcnodata \"0 0 0\" "+fullpath_salida+" "+fullpath_salida_rescale)
        try:
            
            os.remove(fullpath_salida_rescale)
        except:
            print("archivo no existe para eliminar")
            
        os.system("gdalwarp -tr 0.15 0.15 -t_srs EPSG:3116 -r cubic -srcnodata \"0 0 0\" "+fullpath_salida+" "+fullpath_salida_rescale)

        print("------------------------------------------------------------")


# ## Ajuste del histograma

# In[40]:


from matplotlib import pyplot as plt
import gdal


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def get_histograma(source,template,banda):
    
    ds1 = gdal.Open(source)
    myarray1 = np.array(ds1.GetRasterBand(banda).ReadAsArray())

    source = myarray1

    ds = gdal.Open(template)
    myarray = np.array(ds.GetRasterBand(banda).ReadAsArray())

    template = myarray

    matched = hist_match(source, template)

    return matched


    
source=r"C:\Users\Ivan Carrillo\Desktop\Tesis Maestria\5. Datos Campo\Fotos dron 2da salida\vuelo 4\geo\DJI_0389_g_1m.tif"
template=r"C:\Users\Ivan Carrillo\Desktop\Tesis Maestria\5. Datos Campo\Fotos dron 2da salida\vuelo 4\geo\DJI_0377_g_1m.tif"

ds1 = gdal.Open(source)
geotransform = ds1.GetGeoTransform()
cols =  ds1.RasterXSize
rows =  ds1.RasterYSize
projection = ds1.GetProjection()

banda1=get_histograma(source,template,1)
banda2=get_histograma(source,template,2)
banda3=get_histograma(source,template,3)


saveRaster(banda1,banda2,banda3,"out.tif",cols,rows,projection)


# In[34]:


def saveRaster(banda1,banda2,banda3,datasetPath,cols,rows,projection):
    rasterSet = gdal.GetDriverByName('GTiff').Create(datasetPath, cols, rows,3,gdal.GDT_Byte)
    rasterSet.SetProjection(projection)
    rasterSet.SetGeoTransform(geotransform)
    rasterSet.GetRasterBand(1).WriteArray(banda1)
    rasterSet.GetRasterBand(1).SetNoDataValue(-999)
    rasterSet.GetRasterBand(2).WriteArray(banda2)
    rasterSet.GetRasterBand(2).SetNoDataValue(-999)
    rasterSet.GetRasterBand(3).WriteArray(banda3)
    rasterSet.GetRasterBand(3).SetNoDataValue(-999)
    rasterSet = None

