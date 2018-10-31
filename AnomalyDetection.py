import cv2
import os
import sys
import numpy as np
from keras.models import model_from_yaml
from keras.preprocessing import image
import pandas as pd
from imageai.Detection import ObjectDetection
exe = os.getcwd()

 
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(exe , "yolo.h5"))
detector.loadModel()
custom_objects = detector.CustomObjects(bicycle = True, car = True, motorcycle = True, airplane = True, bus = True, train = True, truck = True, skateboard = True)
# load YAML and create model
yaml_file = open(os.path.join(exe , "model.yaml"), 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
classifier = model_from_yaml(loaded_model_yaml)
# load weights into new model
classifier.load_weights(os.path.join(exe , "model.h5"))
print("----------TIP: PUT testing dataset in \"images\" folder for Anomaly Detections-------------")   
print("-----------------------Loaded Trained model from disk----------------------")
print("----------------------------ANOMALY DETECTION--------------------")

folderName = "images"                                                        
folderPath = os.path.join(os.path.dirname(os.path.realpath('__file__')), folderName)
if not os.path.exists(folderPath):
    os.makedirs(folderPath)
                                              
while True:
    if not os.listdir(folderPath) :
     print("Folder empty")   
     sys.exit(0)
    # read frames from dataset folder
    for filename in os.listdir(folderPath):
     key = cv2.waitKey(1) & 0xFF 
     if key == ord("q"):
        cv2.destroyAllWindows()
        sys.exit(0)
     fi = (os.path.join(folderPath,filename))
     detections = detector.detectCustomObjectsFromImage(input_image=fi, output_image_path=os.path.join(exe,"Output_image" + ".jpg"), custom_objects=custom_objects, minimum_percentage_probability=65)
     c=0
     for eachObject in detections:
         c+=1
         print(eachObject["name"]+" : "+str(eachObject["percentage_probability"]))
        #print("--------------------------------")
     
        
     test_image = image.load_img(fi, target_size = (64, 64))
     
     test_image = image.img_to_array(test_image)
     
     test_image = np.expand_dims(test_image, axis = 0)
     
     result = classifier.predict(test_image)
     print("Executed:")
     if result[0][0] == 1:
        prediction = 'normal'
        #print("roadwalk")
        
     else:
       prediction = 'grasswalk'
       
     if c>0:
       if prediction=='grasswalk':
         print("==========Object and Grass-walk ANOMALY==============")  
         out_text="Object and Grass-walk anomaly"
         
       else:  
         out_text="Object anomaly"
         print("==========Object ANOMALY==============")  
     else:
        if prediction=='grasswalk':  
         out_text="Grass-walk anomaly" 
         print("==========Grass-walk ANOMALY==============")  
        else:  
         out_text="No anomalies"     
         print("==========NO ANOMALY==============")  
     x = cv2.imread(os.path.join(folderPath,filename))
     cv2.putText(x,out_text, (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)  
    
     img_path = cv2.imread(exe+ "\\Output_image" +".jpg")
     
     cv2.imshow("FRAME 2",img_path)
     cv2.waitKey(100)
     cv2.imshow("FRAME 1", x)
     cv2.waitKey(100)
     
     os.remove(exe+ "\\Output_image" +".jpg")   
   

    else:
        continue

    key = cv2.waitKey(1) & 0xFF 

    if key == ord("q"):
        cv2.destroyAllWindows()
        sys.exit(0)
    
# cleanup
cv2.destroyAllWindows()
sys.exit(0)
 