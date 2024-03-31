"""
import cv2 as cv
import sys, itertools
img = cv.imread("/home/creditizens/Downloads/perrots.jpg")
if img is None:
 sys.exit("Could not read the image.")
cv.imshow("Display window", img)
k = cv.waitKey(0)
if k == ord("s"):
 cv.imwrite("starry_night.png", img)
"""



"""
# video capture and write frame to .mp4
import numpy as np
import cv2 as cv
import time

# webcam video 0, -1 or 1, 2 depending on the video streams if many
# cap = cv.VideoCapture(0)
# from url video on youtube here
cap = cv.VideoCapture("https://www.youtube.com/watch?v=q6njK5acv-A")

# Define the codec and create VideoWriter object
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    break
    frame = cv.flip(frame, 0)

    # write the flipped frame
    out.write(frame)

    cv.imshow('frame', frame)
    time.sleep(10)
    if cv.waitKey(1) == ord('q'):
      break

# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()
"""



"""
# here blends 2 images with one having lower opacity on top of the other
import cv2 as cv
# Load the images
img1 = cv.imread('/home/creditizens/Downloads/image1.png')
img2 = cv.imread('/home/creditizens/Downloads/image2.png')
# Resize img2 to match img1's dimensions
img2_resized = cv.resize(img2, (img1.shape[1], img1.shape[0]))
# Blend the images, need to have same dimension
dst = cv.addWeighted(img1, 0.7, img2_resized, 0.3, 0)
# Show the blended image, shows also coordinates when moving mouse
cv.imshow('Blended Image', dst)
cv.waitKey(0)
cv.destroyAllWindows()
"""


"""
# to find corners on an images using goodFeaturesToTrack, but image have to be changed to a greyscale one, we also get coordinated while moving mouse and rgb color
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('/home/creditizens/Downloads/boxes.jpg')
# changed to greyscale image
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# gooFeaturesToTrack with parameters: greyscale_image, nbrs of corners, precision % (under it is not detected), eucledian distance
corners = cv.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int64(corners)

for i in corners:
 x,y = i.ravel()
 cv.circle(img,(x,y),3,255,-1)

plt.imshow(img),plt.show()
"""


"""
# using SIFT(has a patent so have to pay to use it so use next one which is not: ORB and more Faster and better) to  detect keypoint or structure, then we can use this compute those points and be able to find the same structure in other images by comparing keypoints
import numpy as np
import cv2 as cv

img = cv.imread('/home/creditizens/Downloads/abstract_art.jpg')
img_gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

kp = sift.detect(img_gray,None)

# this flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS will also show the orientation of keypoints inside the circle drawn
img_keypoints_drawn = cv.drawKeypoints(img_gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imwrite('abstract_art_sift_keypoints.jpg',img_keypoints_drawn)

# this to compute descriptor vector
# Keypoint Matching: Keypoints between two images are matched by identifying their nearest neighbours.
# But in some cases, the second closest-match may be very near to the first. It may happen due to noise or some other reasons.
# In that case, ratio of closest-distance to second-closest distance is taken. If it is greater than 0.8, they are rejected.
#It eliminates around 90% of false matches while discards only 5% correct matches, as per the paper.
kp, des = sift.compute(img_gray, kp) # kp will be a list of keypoints and des is a numpy array of shape: (nbr of keypoints)x128
print("Compute only KP: ", kp, "Conpute only DES: ", des)

# or you can do those two steps in one using detectAndCompute function  to get keypoints and descriptors vectors
sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(img_gray, None)
print("Detect and Compute kp: ", kp, "Dectect and Compute des: ", des)
"""


"""
# ORD - Oriented FAST (Keypoints detector) Rotated BRIEF (descriptor vectors). so here will try to find corner as well but no point traced but multiple circles for zones
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('/home/creditizens/Downloads/boxes.jpg', cv.IMREAD_GRAYSCALE)

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img, None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()
"""


"""
# find matching images in another image
# we can use a function from calib3d module, ie cv.findHomography().
# If we pass the set of points from both the images, it will find the perspective transformation of that object.
# Then we can use cv.perspectiveTransform() to find the object. It needs at least four correct points to find the transformation.
# algorithm uses RANSAC or LEAST_MEDIAN (which can be decided by the flags).
# So good matches which provide correct estimation are called inliers and remaining are called outliers.
# cv.findHomography() returns a mask which specifies the inlier and outlier points.
# so we find first SIFT features and apply best ratio
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv.imread('/home/creditizens/Downloads/abstract_art.jpg', cv.IMREAD_GRAYSCALE) # queryImage
img2 = cv.imread('/home/creditizens/Downloads/abstract_art_hidden_in_image.jpg', cv.IMREAD_GRAYSCALE) # trainImage

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good_matches = [m for m,n in matches if m.distance < 0.7*n.distance]
print("GOOD Mathcing points: ", good_matches)

# Now, we need to extract the keypoints that passed the ratio test
# queryIdx (query descriptor index), trainIdx (train descriptor index), imgIdx (train image index), and distance (distance between descriptors).
# These attributes help in identifying and utilizing matched keypoints between two sets of images.
# Note: m.queryIdx for img1 (queryImage), m.trainIdx for img2 (trainImage)
good_kp2 = [kp2[m.trainIdx] for m in good_matches]

# Draw good keypoints on img2
img2_with_matches = cv.drawKeypoints(img2, good_kp2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0,255,0))

# Display the image with keypoints
plt.imshow(img2_with_matches), plt.show()

# OR we can use as decided earlier to check if at least 10 points are matching with the parameter MIN_MATCH_COUNT amd use findHomography function to find those point even if in the train image the queryImage is rotated or distorded, then plot it
if len(good_matches) > MIN_MATCH_COUNT:
  src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
  dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

  M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
  matchesMask = mask.ravel().tolist()

  h, w = img1.shape
  pts = np.float32([ [0,0], [0,h-1], [w-1, h-1], [w-1,0] ]).reshape(-1,1,2)
  dst = cv.perspectiveTransform(pts, M)

  img2_matched_polylines_with_img1 = cv.polylines(img2, [np.int32(dst)],True,255,3, cv.LINE_AA)
  print("Img2 new with mathced points downloaded to PWD folder")
  # here we get a new image created with a rectangle showing where is the object that we query for: CAN BE USED TO RECOGNIZE HAND OR SIGN FROM VIDEO FRAME THAT WILL START A BACKGROUND ACTION
  cv.imwrite('abstract_art_sift_keypoints_matched.jpg', img2_matched_polylines_with_img1)

else:
 print(f"Not enough matches are found - {len(good_matches)}/{MIN_MATCH_COUNT}")
 matchesMask = None

# OR can use this to plot mathcing points with line connecting those points having the images side by side queryImage|trainImage
# flag=2 draw only inliers(matching) and not outliers
draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask=matchesMask, flags=2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

# gray is a valid value for cmap (there so many other colors but we use that one as in documentation)
# this will show the two images with the matching points joined by grenn lines to justify which point is mathcing which one from one image to the other
plt.imshow(img3, 'gray'), plt.show()
"""
