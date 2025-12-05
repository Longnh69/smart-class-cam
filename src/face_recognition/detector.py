import insightface

class FaceDetector:
    def __init__(self):
        self.model = insightface.app.FaceAnalysis()
        self.model.prepare(ctx_id=0)

    def detect(self, frame):
        return self.model.get(frame)
