import cv2
import os

import numpy as np
from keras.models import model_from_yaml
from keras.preprocessing import image

from imageai.Detection import ObjectDetection
exe = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(exe, "yolo.h5"))
detector.loadModel()
custom_objects = detector.CustomObjects(bicycle=True,car=True,motorcycle=True, airplane=True, bus=True, train=True, truck=True, skateboard=True,sports_ball=True)
# load YAML and create model
yaml_file = open(os.path.join(exe , "model.yaml"), 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
classifier = model_from_yaml(loaded_model_yaml)
# load weights into new model
classifier.load_weights(os.path.join(exe , "model.h5"))
print("----------TIP: PUT testing dataset in \"images\" folder for Anomaly Detections-------------")   
print("-----------------------Loaded Trained model from disk----------------------")
print("----------------------------ANOMALY DETECTION OPENS--------------------")
stream = cv2.VideoCapture(0)



while True:
    # read frames from live web cam stream
    (grabbed, frame) = stream.read(0)
    frame = cv2.flip(frame, 1)
    if grabbed==True:
    # resize the frames to be smaller and switch to gray scale
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
   
     output = frame.copy()
     
     cv2.imwrite(exe + "\\User" + ".jpg",
                    output)
     
    
     
     detections = detector.detectCustomObjectsFromImage(input_image=os.path.join(exe , "User" +".jpg") , output_image_path=os.path.join(exe, "Output_image" +".jpg"), custom_objects=custom_objects, minimum_percentage_probability=65)
     c=0
     for eachObject in detections:
         c+=1
         print(eachObject["name"]+" : "+str(eachObject["percentage_probability"]))
     
     #cv2.imshow("Face Detection",test_image1 )'''

     
    
     test_image = image.load_img(exe + "\\User" +".jpg", target_size = (64, 64))
     
     test_image = image.img_to_array(test_image)
     
     test_image = np.expand_dims(test_image, axis = 0)
     
     result = classifier.predict(test_image)
     print("Executed:")
     if result[0][0] == 1:
        prediction = 'normal'
        #print("roadwalk")
        
     else:
       prediction = 'grasswalk'
       #print("grasswalk")
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
     cv2.putText(output,out_text, (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  
     
     img_path=exe+"\\Output_image" +".jpg"
     img = cv2.imread(img_path)
     # show the frame
     cv2.imshow("FRAME 1",img)
     cv2.imshow("FRAME 2",output)
     os.remove(exe+ "\\Output_image" +".jpg")     
     os.remove(exe+ "\\User" +".jpg")  
    else:
        continue
    
        # draw a fancy border around the faces
    
    
    key = cv2.waitKey(1) & 0xFF 
    # press q to break out of the loop
    if key == ord("q"):
        break
    
    
# cleanup
cv2.destroyAllWindows()
 