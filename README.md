
# Counting People in a Public Area Using a Camera Feed

This project detects and counts people using YOLOv8 and DeepSORT.

## Features
- Real-time people detection
- Entry & exit counting
- Video output generation

## Technologies
- Python
- YOLOv8
- OpenCV
- DeepSORT

## Run
pip install -r requirements.txt  
python main.py
# YOLOv8 Person Detection

This project utilizes the YOLOv8 object detection model to detect and count people in a given video or live stream. It employs the Ultralytics YOLO library, which is based on the YOLOv8 models.


Run the application:

```
python main.py
```

The application will open a window showing the video stream with people bounding boxes and counts. Press Esc to exit.

The final output video will be saved as Final_output.mp4 in the project directory.

## Performance
The application uses object tracking and centroid-based counting to track people and count their entry and exit. The counting results are displayed in the video window and logged to the console.



