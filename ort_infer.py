import onnxruntime
import numpy as np

model = "model/model_bn.onnx"
session = onnxruntime.InferenceSession(model, providers=['CPUExecutionProvider'])

# generate input
input_npa = np.full((1, 3, 32, 32), 1.0, dtype=np.float32)

# inference
output_npa = session.run(["output"], {"input": input_npa})

print(output_npa[0])
