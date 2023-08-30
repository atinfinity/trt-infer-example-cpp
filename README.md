# Example inference code of TensorRT(C++)

This is example inference code of TensorRT.  
I checked on the following environment.

- reComputer J4012(Jetson Orin NX 16GB)
- JetPack 5.1.2
- TensorRT 8.5.2

And, I used `onnxruntime` to compare the result between ONNX Runtime and TensorRT.

- onnxruntime-gpu 1.15.1
  - <https://elinux.org/Jetson_Zoo#ONNX_Runtime>

## Preparation

### create ONNX model

I created `model/model_bn.onnx`. This model was generated using the following steps.  
<https://github.com/NVIDIA-AI-IOT/jetson_dla_tutorial>

![](image/model_bn.onnx.svg)

### Build TensorRT Engine

Please build engine by TensorRT.

```shell
trtexec --verbose --profilingVerbosity=detailed --buildOnly --memPoolSize=workspace:8192MiB --onnx=model/model_bn.onnx --saveEngine=model/model_bn.onnx.engine > model_bn.onnx.engine.build.log
```

If you use DLA(Deep Learning Accelerator), please add `--useDLACore` option.

```shell
trtexec --verbose --profilingVerbosity=detailed --buildOnly --memPoolSize=workspace:8192MiB --onnx=model/model_bn.onnx --saveEngine=model/model_bn.onnx.engine --useDLACore=0 --allowGPUFallback > model_bn.onnx.engine.build.log
```

## Inference

I created `trt_infer.cpp` to infer using TensorRT Engine.

### include NvInfer.h

```cpp
#include <NvInfer.h>
```

### deserialize TensorRT Engine

```cpp
nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine((const void*)engine_data.get(), engine_size);
```

### create context

```cpp
nvinfer1::IExecutionContext* context = engine->createExecutionContext();
```

### inference

```cpp
context->setTensorAddress("input", d_input);
context->setTensorAddress("output", d_output);
bool status = context->enqueueV3(stream);
```

## Build

```shell
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release
cp -r model build
cd build
make
```

## Result

### ONNX Runtime(CPUExecutionProvider)

```shell
$ python3 ort_infer.py
[[-0.00628578 -0.02112402 -0.00283293  0.01181907  0.02438403  0.00028906
  -0.03561208  0.02654092  0.0145703   0.00279154]]
```

### TensorRT(without DLA)

```shell
$ ./trt_infer
[TRT] Loaded engine size: 6 MiB
[TRT] Deserialization required 4967 microseconds.
[TRT] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +5, now: CPU 0, GPU 5 (MiB)
[TRT] Total per-runner device persistent memory is 0
[TRT] Total per-runner host persistent memory is 19872
[TRT] Allocated activation device memory of size 131584
[TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +1, now: CPU 0, GPU 6 (MiB)
-0.00628827 -0.0211218 -0.00284093 0.0118219 0.0243848 0.000285783 -0.0356103 0.0265424 0.014573 0.00278646
```

### TensorRT(with DLA)

```shell
$ ./trt_infer
[TRT] Loaded engine size: 3 MiB
[TRT] Deserialization required 2182 microseconds.
[TRT] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +3, GPU +0, now: CPU 3, GPU 0 (MiB)
[TRT] Total per-runner device persistent memory is 0
[TRT] Total per-runner host persistent memory is 1472
[TRT] Allocated activation device memory of size 12800
[TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 3, GPU 0 (MiB)
-0.00628281 -0.0211182 -0.00282669 0.0118103 0.0243835 0.000305653 -0.0355835 0.0265503 0.0145798 0.00279236
```

## Reference

- <https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html>
- <https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/index.html>
- <https://github.com/dusty-nv/jetson-inference/blob/master/c/tensorNet.cpp>
- <https://github.com/MrLaki5/TensorRT-onnx-dockerized-inference/blob/main/src/trt_engine.cpp>
- <https://github.com/NVIDIA-AI-IOT/NVIDIA-Optical-Character-Detection-and-Recognition-Solution/blob/main/src/TRTEngine.cpp>
- <https://github.com/cyrusbehr/tensorrt-cpp-api>
