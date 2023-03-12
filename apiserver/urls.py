from django.urls import path
from apiserver.views import scan_for_points, return_scaned_doc, test

urlpatterns = [
    path('', test),
    path("scan_for_points", scan_for_points),
    path("return_scaned_doc", return_scaned_doc),

]
