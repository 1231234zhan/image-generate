import cv2
import dlib
import pickle
import numpy as np

pdctdir = 'data/shape_predictor_68_face_landmarks.dat'
ref3dir = 'data/ref3d.pkl'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(pdctdir)
with open(ref3dir, 'rb') as f:
    ref = pickle.load(f)

refA   = ref['outA']
p3d    = ref['p3d']
R0     = ref['R0']
refimg = ref['refimg']

refimg = (refimg*255).astype(np.uint8)
gray = cv2.cvtColor(refimg, cv2.COLOR_BGR2GRAY)
det = detector(gray, 1)[0]
refw = det.width()

def get_pose(_det, _p2d):
    w = _det.width()
    p2d = _p2d * (refw / w)
    p3_ = p3d.reshape((-1,3,1)).astype(np.float)
    p2_ = p2d.reshape((-1,2,1)).astype(np.float)
    distCoeffs = np.zeros((5, 1))    # distortion coefficients
    succ,rvec,_ = cv2.solvePnP(p3_, p2_, refA, distCoeffs)
    if not succ:
        return np.inf

    R1, _ = cv2.Rodrigues(rvec)
    R = np.matmul(R1, R0)
    rvec, _ = cv2.Rodrigues(R)
    theta = np.linalg.norm(rvec)
    theta *= (180 / np.pi)
    return theta