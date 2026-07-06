import sys, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import numpy as np
from matsz.predictor import MATPredictor
ckpt = 'models/MAT_Places512_G_fp16.safetensors'
pred = MATPredictor(ckpt, 1234, 0.0, 255.0)
print('device:', pred.device)
recon = np.zeros((3,512,512), np.float32)
known = np.zeros((512,512), bool)
known[:16,:16] = True
t0 = time.time()
out = pred.predict(recon, known)
print(f'forward: {time.time()-t0:.2f}s, shape={out.shape}')
