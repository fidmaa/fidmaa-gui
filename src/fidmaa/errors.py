from PySide6.QtCore import QObject

tr = QObject.tr

NO_DEPTH_DATA_ERROR = tr(
    """<p>
Looks like this image has no depth data. Make sure you took the photo  without any
'Move furthrer from the subject' message on the phone.
</p><p>
This application currently supports selfies (photos taken with the front-facing camera)
taken on the iPhone in **portrait** mode. Other kinds of pictures probably contain no usable data.
</p><p>
If instead of JPEG your iPhone Xs transfers a HEIC/HEIF file, this means you took the
photo too close or too far away. Make sure there are no "Move away from the subject" messages.
</p>"""
)

NO_FRONT_CAMERA_NOTIFICATION = tr(
    """<p>
Looking at the file description, it does not look like it was taken using the front camera
of the iPhone (the TrueDepth camera). Chances are it probably does not contain proper depth
data to use with this sofware.
</p><p>
Please consider re-taking this picture with front ("selfie") camera in portrait mode.
</p><p>
If this image was taken with the front iPhone camera in portrait mode, chances are it was
shared to you via an incompatible method. Please remember, that sharing the image via AirDrop,
via iCloud or via Photos app is safe. If you recevied this image via Messages app, it was probably
stripped out of important information and thus cannot be used for measurements.
</p>"""
)

FACE_NOT_DETECTED = tr(
    """<p>
Face was not detected in this image.
</p><p>
In case this is an image with the neck extended, you're probalby okay and you can take the
measurements.
</p><p>
In case this is face image, it means that it is probably hardly readable. In this case,
please re-take the picture.
</p>"""
)

FACE_TOO_SMALL = tr(
    """<p>
Face was detected and it looks like it is too small. You should probably re-take the
picture and make sure it is close enough so that the face area takes at least {minimum_width:.2f}% of
width of the photo and the height of the face is at least {minimum_height:.2f}% of the photo.
</p><p>
In case this is an image with the neck extended, you're probalby okay and you can take
the measurements.
</p><p>
Current face measurements: width&nbsp;is&nbsp;{percent_width:.2f}&nbsp;%, height&nbsp;is&nbsp;{percent_height:.2f}&nbsp;%.
</p>"""
)

MULTIPLE_FACES_DETECTED = tr(
    """<p>
Multiple faces detected on the picture. You should probably re-take the picture and make sure that there is only one face and it is close enough, so the face area takes at least 60-75% of the photo.
</p><p>
In case this is an image with the neck extended, you're probalby okay and you can take the measurements.
</p>"""
)
