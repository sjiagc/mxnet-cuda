from mxnet import nd, cpu, gpu, gluon, init, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
import time

device = gpu()


def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return (output.argmax(axis=1) ==
            label.astype('float32')).mean().asscalar()


def train():
    mnist_train = datasets.FashionMNIST(train=True)
    X, y = mnist_train[0]
    print('X shape: ', X.shape, 'X dtype', X.dtype, 'y:', y)

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.13, 0.31)])
    mnist_train = mnist_train.transform_first(transformer)

    batch_size = 256
    train_data = gluon.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)

    for data, label in train_data:
        print(data.shape, label.shape)
        break

    mnist_valid = gluon.data.vision.FashionMNIST(train=False)
    valid_data = gluon.data.DataLoader(
        mnist_valid.transform_first(transformer),
        batch_size=batch_size, num_workers=4)

    net = nn.HybridSequential()
    net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Flatten(),
            nn.Dense(120, activation="relu"),
            nn.Dense(84, activation="relu"),
            nn.Dense(10))
    net.initialize(init=init.Xavier(), ctx=device)
    net.hybridize()

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

    for epoch in range(10):
        train_loss, train_acc, valid_acc = 0., 0., 0.
        tic = time.time()
        for data, label in train_data:
            data = data.copyto(device)
            label = label.copyto(device)
            # forward + backward
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            # update parameters
            trainer.step(batch_size)
            # calculate training metrics
            train_loss += loss.mean().asscalar()
            train_acc += acc(output, label)
        # calculate validation accuracy
        for data, label in valid_data:
            data = data.copyto(device)
            label = label.copyto(device)
            valid_acc += acc(net(data), label)
        print("Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec" % (
                epoch, train_loss/len(train_data), train_acc/len(train_data),
                valid_acc/len(valid_data), time.time()-tic))

    net.export('net-trained', epoch=10)


if __name__ == '__main__':
    train()
