#include "logger.hpp"
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <fstream>
#include <string>
#include <memory>
#include <vector>
#include <iostream>

nvinfer1::ICudaEngine* createDeserializeCudaEngine(nvinfer1::IRuntime* runtime, const std::string model_file)
{
    std::ifstream in_file(model_file, std::ios::binary | std::ios::in);
    std::streampos begin, end;
    begin = in_file.tellg();
    in_file.seekg(0, std::ios::end);
    end = in_file.tellg();
    const std::size_t engine_size = end - begin;
    in_file.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> engine_data(new char[engine_size]);
    in_file.read((char*)engine_data.get(), engine_size);
    in_file.close();
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine((const void*)engine_data.get(), engine_size);
    return engine;
}

int main(int argc, const char* argv[])
{
    // deserialize TensorRT Engine
    Logger logger;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    const std::string model_file = "model/model_bn.onnx.engine";
    nvinfer1::ICudaEngine* engine = createDeserializeCudaEngine(runtime, model_file);

    // create context
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // input and output data
    size_t input_elem_num = 1 * 3 * 32 * 32;
    size_t output_elem_num = 1 * 10;
    std::vector<float> input(input_elem_num, 1.0f);
    std::vector<float> output(output_elem_num, 0.0f);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // allocate device memory
    float *d_input = nullptr;
    float *d_output = nullptr;
    cudaMallocAsync(&d_input, input_elem_num * sizeof(float), stream);
    cudaMallocAsync(&d_output, output_elem_num * sizeof(float), stream);

    // copy HtoD
    cudaMemcpyAsync(d_input, input.data(), input_elem_num * sizeof(float), cudaMemcpyHostToDevice, stream);

    // inference
    context->setTensorAddress("input", d_input);
    context->setTensorAddress("output", d_output);
    bool status = context->enqueueV3(stream);

    cudaStreamSynchronize(stream);

    // copy DtoH
    cudaMemcpyAsync(output.data(), d_output, output_elem_num * sizeof(float), cudaMemcpyDeviceToHost, stream);

    // print output data
    for (auto const &i: output)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    // release device memory
    cudaFree(d_input);
    cudaFree(d_output);

    cudaStreamDestroy(stream);

    return 0;
}
