
#  Instant Segmentation

In this lesson, we will talk about an interesting algorithm. Mask R-CNN is an instant segmentation algorithm which means that it can detect the object in the image but also a mask on each object. This means that on a person you not only have the box but the coordinates surrounding the person.

Mask R-CNN was developed in 2017, which means that it is old, in computer vision software development is very fast, but still today it is an excellent algorithm to be used even in commercial projects.

# How to use Mask R-CNN with OpenCV

First of all you have to make sure you have OpenCV installed, if not run this command from the terminal:







To run tests, run the following command

```bash
 pip install opencv-python
```

If everything is installed correctly, you can download the files for the dnn modules from this site

1-frozen_inference_graph_coco.pb

2-mask_rcnn_inception_v2_coco_2018_01_28.pbtxt

When we have all the material to proceed we set up the python file in the usual way by calling the opencv library

```bash
import cv2
import numpy as np
...
cv2.waitKey(0)
```
and we’re ready to go ahead and upload the model.

# Load the model

To make the model work with OpenCV on our python file we use the function cv2.dnn.readNetFromTensorflow being careful to enter the correct path of the files to be load.

```bash
# Loading Mask RCNN
net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                    "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

```
# Detect object (draw the box)
We need to convert the image to model-readable format.

```bash
# Detect objects
blob = cv2.dnn.blobFromImage(img, swapRB=True)
net.setInput(blob)
```

The image size must be scaled, the color format must change, and much more, otherwise, you will get errors.
We put everything in the network and the model is already working
```bash
boxes, masks = net.forward(["detection_out_final", "detection_masks"])
detection_count = boxes.shape[2]
```

We have all the coordinates of the objects and all that remains is to do the extraction. Being an array we use a simple for loop and as a limit the total number of objects found detection_count

```bash
for i in range(detection_count):
    box = boxes[0, 0, i]
    class_id = box[1]
    score = box[2]
    if score < 0.5:
        continue
    # Get box Coordinates
    x = int(box[3] * width)
    y = int(box[4] * height)
    x2 = int(box[5] * width)
    y2 = int(box[6] * height)
    roi = black_image[y: y2, x: x2]
    roi_height, roi_width, _ = roi.shape
```
# Detect object (take out the mask)
For each box, in the same position, there is an associated mask that perfectly overlaps the object. All coordinate data has been extracted and is now contained in the arrays.
```bash
        # Get the mask
    mask = masks[i, int(class_id)]
    mask = cv2.resize(mask, (roi_width, roi_height))
    _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
    cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)
```

We must now draw the coordinates with the function fillPoly()

```bash
    # Get mask coordinates
    contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = colors[int(class_id)]
    for cnt in contours:
        cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))
```
If we now try to apply all these operations to the image, we get the result in the picture 1 and picture 2.



