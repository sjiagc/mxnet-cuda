#include <mxnet-cpp/MxNetCpp.h>

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

using namespace mxnet::cpp;

static const int BATCH_SIZE = 128;

float TEST_DATA[1][28][28] =
{ {{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.1373, 0.2980, 0.2824, 0.0000, 0.0000, 0.0000, 0.0000,
0.3176, 0.2980, 0.0078, 0.0706, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.3843, 0.8118, 0.9412, 0.7137, 0.3765, 0.5098, 0.5412, 0.4235,
0.5882, 0.7490, 0.7569, 0.6745, 0.3059, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2314,
0.6118, 0.5882, 0.8745, 0.7608, 0.8078, 0.5294, 0.5098, 0.2588,
0.0392, 0.3529, 0.6000, 0.7020, 0.8941, 0.1804, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4471,
0.6314, 0.6118, 0.8314, 0.6980, 0.7843, 0.7255, 0.2941, 0.5098,
0.7529, 0.2471, 0.4549, 0.4314, 0.6392, 0.2275, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.4824, 1.0000, 0.8000, 0.6157, 0.3725, 0.5529, 0.1725, 0.2510,
0.3529, 0.1922, 0.4784, 0.0588, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0824, 0.8588, 0.7373, 0.6157, 0.6353, 0.2549, 0.4667,
0.2471, 0.4706, 0.2431, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.6314, 0.6549, 0.8078, 0.7961, 0.3020, 0.2471,
0.2275, 0.4314, 0.2078, 0.0000, 0.0039, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.4314, 0.3176, 0.7098, 0.7569, 0.6353, 0.2784,
0.1412, 0.2745, 0.2196, 0.0000, 0.0039, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0039,
0.0000, 0.0000, 0.4078, 0.3373, 0.8784, 0.8196, 0.7255, 0.4118,
0.1294, 0.5569, 0.1451, 0.0000, 0.0078, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0039,
0.0000, 0.0000, 0.4275, 0.3098, 0.8549, 0.0706, 0.3725, 0.7098,
0.1333, 0.7255, 0.2196, 0.0000, 0.0039, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0314, 0.4471, 0.4196, 0.3843, 0.2510, 0.2431, 0.1725,
0.5686, 0.8000, 0.3529, 0.0000, 0.0078, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0392, 0.4275, 0.4510, 0.4314, 0.4510, 0.1333, 0.4314,
0.8039, 0.8039, 0.3373, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.2745, 0.7255, 0.5686, 0.5451, 0.5647, 0.1843, 0.9412,
0.7843, 0.7176, 0.5569, 0.0078, 0.0000, 0.0078, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.4039, 0.6157, 0.4275, 0.2118, 0.5843, 1.0000, 0.7608,
0.2157, 0.5882, 0.7804, 0.1647, 0.0000, 0.0196, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.2980, 0.2863, 0.5922, 0.7686, 0.9294, 0.8706, 0.2353,
0.4667, 0.4235, 0.4471, 0.1137, 0.0000, 0.0118, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.3216, 0.2510, 0.4706, 0.6118, 0.4863, 0.9843, 0.5333,
0.1412, 0.3216, 0.6706, 0.0431, 0.0000, 0.0039, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0549, 0.2980, 0.1490, 0.0902, 0.1529, 0.8824, 0.8039,
0.5765, 0.6667, 0.9765, 0.0078, 0.0000, 0.0078, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0039,
0.0000, 0.3255, 0.6196, 0.1176, 0.1608, 0.2627, 0.9333, 0.8706,
0.8392, 0.8588, 0.7059, 0.0000, 0.0000, 0.0039, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0118,
0.0000, 0.4627, 0.8275, 0.2745, 0.6902, 0.2471, 0.7490, 0.0314,
0.5529, 0.8431, 0.4941, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0118,
0.0000, 0.2235, 0.9176, 0.6784, 0.5686, 0.4824, 0.3882, 0.5020,
0.3569, 0.8275, 0.4784, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0078,
0.0000, 0.1059, 0.8471, 0.6510, 0.3255, 0.2392, 0.6588, 0.5725,
0.4706, 0.5804, 0.4549, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0078,
0.0000, 0.0118, 0.8627, 0.7020, 0.3569, 0.3569, 0.6471, 0.6588,
0.6314, 0.6235, 0.3020, 0.0000, 0.0078, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.6784, 0.5882, 0.8039, 0.8863, 0.6627, 0.7843,
0.7412, 0.6941, 0.1882, 0.0000, 0.0078, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.6706, 0.7882, 0.9294, 0.7176, 0.5333, 0.7725,
0.7373, 0.4353, 0.1373, 0.0000, 0.0078, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0039,
0.0000, 0.0000, 0.6000, 0.8196, 0.9020, 0.7843, 0.7529, 0.4902,
0.3686, 0.1216, 0.0667, 0.0000, 0.0078, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.7137, 0.9216, 0.8667, 0.7843, 0.8549, 0.5765,
0.4627, 0.7098, 0.0000, 0.0000, 0.0039, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.5961, 0.8392, 0.9373, 0.9020, 0.8549, 0.7647,
0.5843, 0.6980, 0.0118, 0.0000, 0.0039, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0039,
0.0000, 0.0000, 0.0000, 0.1294, 0.5255, 0.5765, 0.4824, 0.3765,
0.3686, 0.1255, 0.0000, 0.0039, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000 }} };

static const size_t TEST_DATA_SIZE = sizeof(TEST_DATA);
static const size_t TEST_DATA_ELEMENT_COUNT = sizeof(TEST_DATA) / sizeof(float);

double ms_now() {
    double ret;
    auto timePoint = std::chrono::high_resolution_clock::now().time_since_epoch();
    ret = std::chrono::duration<double, std::milli>(timePoint).count();
    return ret;
}


// define the data type for NDArray, aliged with the definition in mshadow/base.h
enum TypeFlag {
    kFloat32 = 0,
    kFloat64 = 1,
    kFloat16 = 2,
    kUint8 = 3,
    kInt32 = 4,
    kInt8 = 5,
    kInt64 = 6,
};

/*
 * class Predictor
 *
 * This class encapsulates the functionality to load the model, prepare dataset and run the forward pass.
 */

class Predictor {
public:
    Predictor(const std::string& model_json_file,
        const std::string& model_params_file);
    ~Predictor();
    std::vector<NDArray> Score(NDArray &inData);

private:
    void LoadModel(const std::string& model_json_file);
    void LoadParameters(const std::string& model_parameters_file);
    void SplitParamMap(const std::map<std::string, NDArray>& paramMap,
        std::map<std::string, NDArray>* argParamInTargetContext,
        std::map<std::string, NDArray>* auxParamInTargetContext,
        Context targetContext);

    inline bool FileExists(const std::string& name) {
        std::ifstream fhandle(name.c_str());
        return fhandle.good();
    }

    std::map<std::string, NDArray> args_map_;
    std::map<std::string, NDArray> aux_map_;
    Symbol net_;
    Executor* executor_;
    Context global_ctx_ = Context::gpu();
};

Predictor::Predictor(const std::string& model_json_file,
    const std::string& model_params_file)
{
    // Load the model
    LoadModel(model_json_file);
    LoadParameters(model_params_file);

    int dtype = kFloat32;
    args_map_["data"] = NDArray(Shape(BATCH_SIZE, 1, 28, 28), global_ctx_, false, dtype);
    std::vector<NDArray> arg_arrays;
    std::vector<NDArray> grad_arrays;
    std::vector<OpReqType> grad_reqs;
    std::vector<NDArray> aux_arrays;

    // infer and create ndarrays according to the given input ndarrays.
    net_.InferExecutorArrays(global_ctx_, &arg_arrays, &grad_arrays, &grad_reqs,
        &aux_arrays, args_map_, std::map<std::string, NDArray>(),
        std::map<std::string, OpReqType>(), aux_map_);
    for (auto& i : grad_reqs) i = OpReqType::kNullOp;

    // Create an executor after binding the model to input parameters.
    executor_ = new Executor(net_, global_ctx_, arg_arrays, grad_arrays, grad_reqs, aux_arrays);
}

Predictor::~Predictor() {
    if (executor_) {
        delete executor_;
    }
    MXNotifyShutdown();
}

/*
 * The following function loads the model from json file.
 */
void Predictor::LoadModel(const std::string& model_json_file) {
    if (!FileExists(model_json_file)) {
        LG << "Model file " << model_json_file << " does not exist";
        throw std::runtime_error("Model file does not exist");
    }
    LG << "Loading the model from " << model_json_file << std::endl;
    net_ = Symbol::Load(model_json_file);
}

/*
 * The following function loads the model parameters.
 */
void Predictor::LoadParameters(const std::string& model_parameters_file) {
    if (!FileExists(model_parameters_file)) {
        LG << "Parameter file " << model_parameters_file << " does not exist";
        throw std::runtime_error("Model parameters does not exist");
    }
    LG << "Loading the model parameters from " << model_parameters_file << std::endl;
    std::map<std::string, NDArray> parameters;
    NDArray::Load(model_parameters_file, 0, &parameters);
    SplitParamMap(parameters, &args_map_, &aux_map_, global_ctx_);
    /*WaitAll is need when we copy data between GPU and the main memory*/
    NDArray::WaitAll();
}

/*
 * The following function split loaded param map into arg parm
 *   and aux param with target context
 */
void Predictor::SplitParamMap(const std::map<std::string, NDArray>& paramMap,
    std::map<std::string, NDArray>* argParamInTargetContext,
    std::map<std::string, NDArray>* auxParamInTargetContext,
    Context targetContext) {
    for (const auto& pair : paramMap) {
        std::string type = pair.first.substr(0, 4);
        std::string name = pair.first.substr(4);
        if (type == "arg:") {
            (*argParamInTargetContext)[name] = pair.second.Copy(targetContext);
        }
        else if (type == "aux:") {
            (*auxParamInTargetContext)[name] = pair.second.Copy(targetContext);
        }
    }
}

std::vector<NDArray>
Predictor::Score(NDArray &inData) {
    inData.CopyTo(&args_map_["data"]);
    executor_->Forward(false);
    return executor_->outputs;
}

int main(int argc, char** argv) {
    static const int TEST_ROUND = 1000;
    LG << "BATCH_SIZE = " << BATCH_SIZE << ", TEST_ROUND = " << TEST_ROUND;
    LG << "Preparing data";
    float* testDataBatch_d;
    cudaMalloc(&testDataBatch_d, TEST_DATA_SIZE * BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; ++i)
        cudaMemcpy(testDataBatch_d + TEST_DATA_ELEMENT_COUNT * i, TEST_DATA, TEST_DATA_SIZE, cudaMemcpyHostToDevice);
    NDArray td = NDArray((mx_float*)testDataBatch_d, Shape(BATCH_SIZE, 1, 28, 28), Context::gpu());
    float* testDataBatch = new float[TEST_DATA_ELEMENT_COUNT * BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE; ++i)
        memcpy_s(testDataBatch + TEST_DATA_ELEMENT_COUNT * i, TEST_DATA_SIZE, TEST_DATA, TEST_DATA_SIZE);
    NDArray t = NDArray((mx_float*)testDataBatch, Shape(BATCH_SIZE, 1, 28, 28), Context::cpu());
    try {
        Predictor predict("net-trained-symbol.json", "net-trained-0010.params");
        double ms = 0;
        // Warm up
        LG << "Warming up";
        for (int i = 0; i < 5; ++i) {
            auto predsGPU = predict.Score(td);
            //LG << "preds GPU: " << predsGPU.size();
            //for (auto pred : predsGPU) {
            //    LG << pred << ", ";
            //}

            auto predsCPU = predict.Score(t);
            //LG << "preds CPU: " << predsCPU.size();
            //for (auto pred : predsCPU) {
            //    LG << pred << ", ";
            //}
        }
        NDArray::WaitAll();
        // GPU
        LG << "Running with GPU input";
        ms = ms_now();
        for (int i = 0; i < TEST_ROUND; ++i) {
            predict.Score(td);
        }
        NDArray::WaitAll();
        ms = ms_now() - ms;
        LG << "Time for GPU: " << ms;
        // CPU
        LG << "Running with CPU input";
        ms = ms_now();
        for (int i = 0; i < TEST_ROUND; ++i) {
            predict.Score(t);
        }
        NDArray::WaitAll();
        ms = ms_now() - ms;
        LG << "Time for CPU: " << ms;
        // CPU 2
        LG << "Running with CPU input, round 2";
        ms = ms_now();
        for (int i = 0; i < TEST_ROUND; ++i) {
            predict.Score(t);
        }
        NDArray::WaitAll();
        ms = ms_now() - ms;
        LG << "Time for CPU: " << ms;
        // GPU 2
        LG << "Running with GPU input, round 2";
        ms = ms_now();
        for (int i = 0; i < TEST_ROUND; ++i) {
            predict.Score(td);
        }
        NDArray::WaitAll();
        ms = ms_now() - ms;
        LG << "Time for GPU: " << ms;
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        delete[] testDataBatch;
        cudaFree(testDataBatch_d);
        return -1;
    }
    delete[] testDataBatch;
    cudaFree(testDataBatch_d);
    return 0;
}
