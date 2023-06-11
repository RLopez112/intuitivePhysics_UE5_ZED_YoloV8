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
from ultralytics import YOLO
import math
import cv2
import csv

def parseArg(argLen, argv, param):
    if(argLen>1):
        if(".svo" in argv):
            # SVO input mode
            param.set_from_svo_file(sys.argv[1])
            print("Sample using SVO file input "+ sys.argv[1])
        elif(len(argv.split(":")) == 2 and len(argv.split(".")) == 4):
            #  Stream input mode - IP + port
            l = argv.split(".")
            ip_adress = l[0] + '.' + l[1] + '.' + l[2] + '.' + l[3].split(':')[0]
            port = int(l[3].split(':')[1])
            param.set_from_stream(ip_adress,port)
            print("Stream input mode")
        elif (len(argv.split(":")) != 2 and len(argv.split(".")) == 4):
            #  Stream input mode - IP
            param.set_from_stream(argv)
            print("Stream input mode")
        elif("HD2K" in argv):
            param.camera_resolution = sl.RESOLUTION.HD2K
            print("Using camera in HD2K mode")
        elif("HD1200" in argv):
            param.camera_resolution = sl.RESOLUTION.HD1200
            print("Using camera in HD1200 mode")
        elif("HD1080" in argv):
            param.camera_resolution = sl.RESOLUTION.HD1080
            print("Using camera in HD1080 mode")
        elif("HD720" in argv):
            param.camera_resolution = sl.RESOLUTION.HD720
            print("Using camera in HD720 mode")
        elif("SVGA" in argv):
            param.camera_resolution = sl.RESOLUTION.SVGA
            print("Using camera in SVGA mode")
        elif("VGA" in argv and "SVGA" not in argv):
            param.camera_resolution = sl.RESOLUTION.VGA
            print("Using camera in VGA mode")

def inferAndMakeBox(source,show):

    results = model.predict(source=source,show = show, conf = 0.45) 
    
    return results

def getBoxCenter(box_coordinate):

    if box_coordinate.size > 0:
            x_0 = math.trunc(box_coordinate.item(0)/2)
            y_0 = math.trunc(box_coordinate.item(1)/2)
            x_med = math.trunc(box_coordinate.item(2)/2)
            y_med= math.trunc(box_coordinate.item(3)/2)
            x_depth = x_0 + x_med
            y_depth = y_0 + y_med
            location=(x_depth,y_depth)
            return location

    else:
            
            print('The array has a size of 0')
            location=(100,100)
            return location

def get3dPoint(x,y,point_cloud,depth):
    point3D = point_cloud.get_value(x, y)
    x_cloud = point3D[1][0]*100
    y_cloud = point3D[1][1]*100
    z_cloud = point3D[1][2]*100
    z_depth=depth.get_value(x,y)[1]
    return (x_cloud,y_cloud,z_cloud)

def FPS():
    currentFPS = str(zed.get_current_fps())
    return currentFPS

def saveCSV(x,y,z,object):
    location = "(X="+str(x)+"Y="+str(y)+"Z="+str(z)+")"
    header = ['---', 'boundingBoxLocation', 'boundingBoxRotation', 'boundingBoxScale','SHAPE','fbxPath','pointCloudPath','SimulatePhysicsFromStart?','GravitiAffectsIt?','id','initialSpeedVector']
    data = ['NewRow',location,'(Pitch=0,Yaw=0,Roll=0)','(X=1,Y=1,Z=1)',0]

    with open(f'F:/_WORKSPACE_PERSONAL/intuitivePhysics/_files/active/{object}-test.csv',mode= 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        writer.writerow(data)

model = YOLO("yolov8N-seg.pt")


if __name__ == "__main__":
    print("Running Depth Sensing sample ... Press 'Esc' to quit\nPress 's' to save the point cloud")

    init = sl.InitParameters(   
                                depth_mode=sl.DEPTH_MODE.ULTRA,
                                coordinate_units=sl.UNIT.METER,
                                coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
    

    trail=[]
    if (len(sys.argv) > 1):
        parseArg(len(sys.argv), sys.argv[1], init)
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    res = sl.Resolution()
    res.width = 1280
    res.height = 720

    camera_model = zed.get_camera_information().camera_model

    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(1, sys.argv, camera_model, res)

    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    image = sl.Mat()
    depth = sl.Mat()

    while viewer.is_available():

        if zed.grab() == sl.ERROR_CODE.SUCCESS:

            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, res)
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            stream = image.get_data()
            
            image_without_alpha = stream[:,:,:3] #esto consume 
            results=inferAndMakeBox(image_without_alpha,False)
            
            print(FPS())
            
            for result in results:
                #prin (item)   
                #print(result)
                boxes=result.boxes
                
                for box in boxes:
                    box_coordinates=box.xyxy.numpy()
                    location=getBoxCenter(box_coordinates)
                    #print(box_coordinates,location)
                    cv2.circle(stream, location, 5, (0,255,100), -1)
                    saveCSV(location[0],location[1],location[2])

            #viewer.updateData(point_cloud)
            point3D=get3dPoint(location[0],location[1],point_cloud,depth)
            print(point3D)
            #trail.append(point3D)
            #print(point_cloud)
            #print("sd")
            
            cv2.imshow('video',stream)

    viewer.exit()
    print(trail)
    zed.close()