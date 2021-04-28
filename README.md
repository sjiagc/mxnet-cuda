# mxnet-cuda
mxnet example to consume CUDA input data directly.

## Training

Training is done by python code inside folder training. The dataset used is Fashion MNIST. With a successful training, there will be net-trained-symbol.json and net-trained-0010.params created.

## mxnet inference

To build inference executable:

1. Open the VS2019 solution;
2. Set mxnet include and lib paths;
3. Create a folder named WorkDir in solution directory;
4. Copy net-trained-symbol.json and net-trained-0010.params into WorkDir;
5. Copy all mxnet dll files into WorkDir (only if mxnet dll directory is not in the system PATH);
6. Set VS2019 debugging working directory to WorkDir.

