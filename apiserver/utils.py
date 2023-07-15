import cv2
import numpy as np
import scipy.spatial.distance
import math
import base64
import imutils
from PIL import Image

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-left, and the fourth is the bottom-right
    rect = np.zeros((4, 2), dtype="float32")
    total = pts.sum(axis=1)
    rect[0] = pts[np.argmin(total)]
    rect[3] = pts[np.argmax(total)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    (rows, cols, _) = image.shape
    # image center
    u0 = (cols)/2.0
    v0 = (rows)/2.0
    p = order_points(pts)
    widthA = scipy.spatial.distance.euclidean([0], p[1])
    widthB = scipy.spatial.distance.euclidean(p[2], p[3])
    maxWidth = max(int(widthA), int(widthB))
    heightA = scipy.spatial.distance.euclidean(p[0], p[2])
    heightB = scipy.spatial.distance.euclidean(p[1], p[3])
    maxHeight = max(int(heightA), int(heightB))

    # visible aspect ratio
    ar_vis = float(maxWidth)/float(maxHeight)

    # make numpy arrays and append 1 for linear algebra
    m1 = np.array((p[0][0], p[0][1], 1)).astype('float32')
    m2 = np.array((p[1][0], p[1][1], 1)).astype('float32')
    m3 = np.array((p[2][0], p[2][1], 1)).astype('float32')
    m4 = np.array((p[3][0], p[3][1], 1)).astype('float32')

    # calculate the focal disrance
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21 = n2[0]
    n22 = n2[1]
    n23 = n2[2]

    n31 = n3[0]
    n32 = n3[1]
    n33 = n3[2]
    n = 0

    try:
        f = math.sqrt(np.abs((1.0/(n)) * ((n21*n31 - (n21*n33 + n23*n31) * u0 +
                                           n23*n33*u0*u0) + (n22*n32 - (n22*n33+n23*n32)*v0 + n23*n33*v0*v0))))
    except:
        f = 1.0
    A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]]).astype('float32')
    At = np.transpose(A)
    Ati = np.linalg.inv(At)
    Ai = np.linalg.inv(A)

    # calculate the real aspect ratio
    ar_real = math.sqrt(np.dot(np.dot(np.dot(n2, Ati), Ai), n2) /
                        np.dot(np.dot(np.dot(n3, Ati), Ai), n3))
    if ar_real < ar_vis:
        W = int(maxWidth)
        H = int(W / ar_real)
    else:
        H = int(maxHeight)
        W = int(ar_real * H)
     # construct the set of destination points to obtain a "birds eye view",
    BEV = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
    M = cv2.getPerspectiveTransform(p, BEV)
    warped = cv2.warpPerspective(image, M, (W, H))
    return warped


def process_image(request):
    # Get the image data from the request
    image_data = request.POST.get('image')
    # Decode the data URL to a binary string
    image_binary = base64.b64decode(image_data.split(',')[1])
    if is_heic_base64(image_binary):
        # Open the HEIC image using pillow-heif
        heic_image = Image.frombytes("RGB", (1, 1), image_binary, "raw", "RGB", 0, 1)
        # Convert the HEIC image to OpenCV format
        img = cv2.cvtColor(np.array(heic_image), cv2.COLOR_RGB2BGR)
    else:
        # Convert the binary string to a NumPy array
        image_np = np.frombuffer(image_binary, np.uint8)
        # Decode the NumPy array to an OpenCV image
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return img

def is_heic_base64(base64_string):
    signature = base64_string[:8]
    return signature == b'ftypheic' or signature == b'ftypheix'

def get_4_corner_points_grabcut(img):
    # Resize image to workable size
    dim_limit = 1080
    max_dim = max(img.shape)
    # Create a copy of resized original image for later use
    orig_img = img.copy()
    resize_scale = 1
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)
        max_dim = max(orig_img .shape)
    
    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
    # GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    # Edge Detection.
    canny = cv2.Canny(gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
 
    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
 
    # Detecting Edges through Contour approximation.
    # Loop over the contours.
    for c in page:
        # Approximate the contour.
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points.
        if len(corners) == 4:
            print("got it")
            approx = corners
            print(approx)
            # get the coordinates of the original image as we perform action on resized image
            if max_dim > dim_limit:
                approx = approx / resize_scale
            approx_list = approx.tolist()
            for item in approx_list:
                item[0][0] = int(item[0][0])
                item[0][1] = int(item[0][1])
            # Sorting the corners and converting them to desired shape.
            return approx_list

def get_4_corner_points_traditional_way(image):
    orig = image.copy()
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(image, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    # find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(corners) == 4:
            approx = corners
            return approx.tolist()
    return None

def clahe_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_planes = list(lab_planes)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab_planes = tuple(lab_planes)
    lab = cv2.merge(lab_planes)
    clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return clahe_bgr








