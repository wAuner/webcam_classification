import numpy as np
import cv2
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

cap = cv2.VideoCapture(0)

model = ResNet50()

while True:
    ret, frame = cap.read()
    

    # crop image
    roi = frame[:,80:560]    
    res = cv2.resize(roi, (224,224))
    
    res = np.expand_dims(res, axis=0)
    

    prediction = model.predict(res)
    decoded = decode_predictions(prediction, top=1)
    

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, decoded[0][0][1], (10,400), font, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow("window",frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 