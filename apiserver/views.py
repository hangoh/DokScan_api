from django.shortcuts import render
from apiserver.utils import order_points, four_point_transform, process_image, reconstruct_points
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils
from django.http import JsonResponse, HttpResponse
import json
from django.middleware.csrf import get_token
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.csrf import csrf_exempt
import base64
# Create your views here.


@ensure_csrf_cookie
def test(request):

    return HttpResponse(json.dumps({"response": "new csrfCookie", "token": request.META['CSRF_COOKIE']}), content_type='application/json')


@csrf_exempt
def scan_for_points(request):

    image = process_image(request)

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
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            return JsonResponse({"result": "positive", "points": screenCnt.tolist()})
    return JsonResponse({"result": "negative"})


@csrf_exempt
def return_scaned_doc(request):
    img = process_image(request)
    points = request.POST.get("points")
    points = np.array(json.loads(points))
    print(points)
    warped = four_point_transform(img, points.reshape(4, 2))

    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset=10, method="gaussian")
    warped = (warped > T).astype("uint8") * 255
    # Convert the processed image back to a byte string
    success, buffer = cv2.imencode('.jpeg', warped)
    image_bytes = base64.b64encode(buffer)

    # Return the byte string as the API response

    return HttpResponse(image_bytes, content_type='image/jpeg')
