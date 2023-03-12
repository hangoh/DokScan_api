import cv2
import numpy as np
import scipy.spatial.distance
import math
import base64


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
    print("ar_real:  "+str(ar_real))
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

    # Convert the binary string to a NumPy array
    image_np = np.frombuffer(image_binary, np.uint8)

    # Decode the NumPy array to an OpenCV image
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return img


def reconstruct_points(p):
    n_p = []
    i = 0
    for j in range(4):
        n_p.append([[p[i], p[i+1]]])
        i = i+2
    return n_p
