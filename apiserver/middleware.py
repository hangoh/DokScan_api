import pytz
from datetime import datetime

import geocoder

class TimezoneMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        print("ip_address for request: {}".format(request.META.get('HTTP_X_FORWARDED_FOR')))
        print("ip_address for request: {}".format(request.META.get('REMOTE_ADDR')))
        g = geocoder.ip(request.META.get('HTTP_X_FORWARDED_FOR'))

        if g and g.timezone:
            timezone_name = g.timezone
            timezone_obj = pytz.timezone(timezone_name)
        else:
            # Default timezone if geolocation or timezone not found
            timezone_obj = pytz.timezone('UTC')

        request.timezone = timezone_obj
        request.time = datetime.now().astimezone(request.timezone)
        print("client timezone: {}".format(request.time))
        response = self.get_response(request)

        return response
