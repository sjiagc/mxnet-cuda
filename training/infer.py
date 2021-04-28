import mxnet as mx
from collections import namedtuple
from mxnet.gluon.data.vision import datasets, transforms


device = mx.gpu()

sym, arg_params, aux_params = mx.model.load_checkpoint('net-trained', 10)
mod = mx.mod.Module(symbol=sym, context=device)
mod.bind(for_training=False, data_shapes=[('data', (10, 1, 28, 28))], label_shapes=None)
mod.set_params(arg_params, aux_params, allow_missing=True)

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.13, 0.31)])

mnist_valid = datasets.FashionMNIST(train=False)
x, y = mnist_valid[:10]
x = transformer(x)
print('x shape: ', x.shape, 'x dtype', x.dtype, 'y:', y)

batch = namedtuple('Batch', ['data'])
x = x.copyto(device)

mod.forward(batch([x]))
output = mod.get_outputs()
preds = [o.argmax(axis=1).astype('int') for o in output]
print(preds)
print(y)
