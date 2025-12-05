import cv2

class TapoCamera:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            raise Exception("❌ Could not open RTSP stream")

    def get_frame(self):
        if not self.cap:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def close(self):
        if self.cap:
            self.cap.release()
