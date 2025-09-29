from picamera2 import Picamera2
camera = Picamera2()
print(camera.sensor_modes)
camera.close()
