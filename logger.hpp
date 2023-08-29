
#include <NvInfer.h>
#include <iostream>

class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        std::cout << "[TRT] " << std::string(msg) << std::endl;
    }
};
