from django.shortcuts import render
from apiserver.utils import  four_point_transform, process_image, get_4_corner_points_traditional_way, get_4_corner_points_grabcut, clahe_image
from skimage.filters import threshold_local
import numpy as np
import cv2
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
    points = None
    # different kind of method can help to detect corners in different kind of image, therefore multiple method were use to retrieve points
    # some might fail and some might success to retrieve points. this depends on image contrast, the complicated level of the background, angle, light refraction and much more
    
    # use grabcut method
    points = get_4_corner_points_grabcut(image) 
    if points is not None:    
        return JsonResponse({"result": "positive", "points":points})
    # traditional way of turn it grayscale bur it and find the edges but with clahe img
    points = get_4_corner_points_traditional_way(clahe_image(image))
    if points is not None:    
        return JsonResponse({"result": "positive", "points":points})
    # traditional way of turn it grayscale bur it and find the edges
    points =  get_4_corner_points_traditional_way(image)
    if points is not None:    
        return JsonResponse({"result": "positive", "points":points})
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
    warped_1 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped_1, 21, offset=10, method="gaussian")
    warped_2 = (warped_1 > T).astype("uint8") * 255
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    warped_3 = cv2.filter2D(warped_2, -1, kernel)
    warped_3 = cv2.cvtColor(warped_3, cv2.COLOR_GRAY2BGR)
    c = clahe_image(warped_3)
    warped_c = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    w = cv2.bitwise_and(warped_3, warped)
    # Convert the processed image back to a byte string
    success, buffer = cv2.imencode('.jpeg', warped_3)
    image_bytes = base64.b64encode(buffer)

    # Return the byte string as the API response

    return HttpResponse(image_bytes, content_type='image/jpeg')
