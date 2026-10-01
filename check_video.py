import cv2

video = cv2.VideoCapture("traffic2.mp4")

fps = video.get(cv2.CAP_PROP_FPS)
frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

duration = frames / fps

print("FPS:", fps)
print("Total Frames:", frames)
print("Resolution:", width, "x", height)
print("Duration:", duration, "seconds")

video.release()