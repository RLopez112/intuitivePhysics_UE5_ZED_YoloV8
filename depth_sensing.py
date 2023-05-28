########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    This sample demonstrates how to capture a live 3D point cloud   
    with the ZED SDK and display the result in an OpenGL window.    
"""

import sys
import ogl_viewer.viewer as gl
import pyzed.sl as sl
import math
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import csv
import cv2

model = YOLO("yolov8n-seg.pt")
  

# define a video capture object
vid = cv2.VideoCapture(1)
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
height= vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

print("Running Depth Sensing sample ... Press 'q' to quit")
    
init = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720,
                             depth_mode=sl.DEPTH_MODE.PERFORMANCE,
                             coordinate_units=sl.UNIT.METER,
                             coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)


zed = sl.Camera()
status = zed.open(init)
init.depth_maximum_distance=400

if status != sl.ERROR_CODE.SUCCESS:
    print(repr(status))
    exit()

res = sl.Resolution()
res.width = 720
res.height = 404

image = sl.Mat()
depth = sl.Mat()

camera_model = zed.get_camera_information().camera_model
mirror_ref = sl.Transform()
mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
tr_np = mirror_ref.m

# Create OpenGL viewer // for pointcloud display.

#viewer = gl.GLViewer()
#viewer.init(len(sys.argv), sys.argv, camera_model, res)

point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

header = ['---', 'boundingBoxLocation', 'boundingBoxRotation', 'boundingBoxScale','SHAPE','fbxPath','pointCloudPath','SimulatePhysicsFromStart?','GravitiAffectsIt?','id','initialSpeedVector']

def inferAndMakeBox(source,show):
    results = model.predict(source=source,show = show, conf = 0.8) 
    boxCoordinates=results[0].boxes.xyxy.numpy()
    return boxCoordinates
    
def getBoxCenter(boxCoordinates):
    if boxCoordinates.size > 0:
            
            x_depth=math.trunc(boxCoordinates.item(0))
            y_depth=math.trunc(boxCoordinates.item(1))
            z_depth=depth.get_value(x_depth,y_depth)[1]
            
            point3D = point_cloud.get_value(x_depth, y_depth)

            x_cloud = point3D[1][0]*100
            y_cloud = point3D[1][1]*100
            z_cloud = point3D[1][2]*100

            #print("center",x_cloud,y_cloud,z_depth)

            location = "(X="+str(x_cloud)+"Y="+str(y_cloud)+"Z="+str(z_cloud)+")"

            data = ['NewRow',location,'(Pitch=1.261254,Yaw=1.521589,Roll=1.111004)','(X=5.000000,Y=5.000000,Z=5.000000)',1]

    else:
            print('The array has a size of 0')



def printFPS():
    currentFPS = zed.get_current_fps()
    print(currentFPS)

def writeToCSV(path,header,data):
    with open(path,mode= 'w', encoding='UTF8') as f:
                writer = csv.writer(f)

                # write the header
                writer.writerow(header)

                # write the data
                writer.writerow(data)

while(True):

    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, res)
        
        depth_stream = depth.get_data()
        
        stream = image.get_data()
        image_without_alpha = stream[:,:,:3] #removes alpha chanel      ESTO CUESTA UN MONTÓN!!!!!   VER COMO RESOLVERLO

        inferAndMakeBox(image_without_alpha,True)

        printFPS()     
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()