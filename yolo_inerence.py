from ultralytics import YOLO

# model = YOLO('yolov8x')

# result = model.predict("input_videos\image.png", save=True)
# print("the result is **********")
# print(result) ## has th class names, masks, boxes
# print("The End o result **************")
# print(f"result data type is a{type(result)}") ## has the class, conf rate, original shhape, coordinates, etc
# print("boxes")

# ## uses the min/max representation of boundary boxes positions
# for box in result[0].boxes:
#     print(box)



# ## running on a video is similar
# ## copy pasting or clarity
# ### notice   video 1/1 (frame 1/214) in the terminal
# ## notice that tennis ball prediction is so poor and we want a better model to fine-tune
# ## we are doing that in the training older under the tennis ball dec nnotebook
# result = model.predict("input_videos\input_video.mp4", save=True)
# print("the result is **********")
# print(result) 
# print("The End o result **************")
# print(f"result data type is a{type(result)}") 
# print("boxes")


# for box in result[0].boxes:
#     print(box)



## after fine-tunnninng, inspecting the models we trianed
## last shown better results than best
# model = YOLO('models/yolo11_last.pt')
# result = model.predict("input_videos\input_video.mp4", conf=0.2, save=True)
# print(result)
# print("boxes")
# for box in result[0].boxes:
#     print(box)


## now we need "object matching" to kee track  of the same object
model = YOLO('yolov8x')
result = model.track("input_videos\input_video.mp4", conf=0.2, save=True)
## check runs/track for results


## now we want a dataset and a model for the key points detector
